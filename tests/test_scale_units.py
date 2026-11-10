"""单元生成 + split 可复现（字节一致）+ 8:2 + 原子写用例（T02）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from stock_ai.config.schema import SampleConfig
from stock_ai.samples.builder import SampleBuilder
from stock_ai.scale.atomic import atomic_write_json
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.units import SPLIT_ALGORITHM, SplitPlan, Unit, UnitScanner


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _make_df(start: str, periods: int, seed: int = 1):
    """构造确定性日线 DataFrame。"""
    import pandas as pd

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, periods)))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + rng.normal(0, 0.002, periods)),
            "high": close * (1 + np.abs(rng.normal(0, 0.004, periods))),
            "low": close * (1 - np.abs(rng.normal(0, 0.004, periods))),
            "close": close,
            "volume": np.abs(rng.normal(2e6, 3e5, periods)),
        }
    )


class _FakeCache:
    def __init__(self, data: dict) -> None:
        self._data = data

    def exists(self, symbol: str) -> bool:
        return symbol in self._data

    def read(self, symbol: str):
        return self._data[symbol]


class _FakeLoader:
    def __init__(self, data: dict) -> None:
        self.cache = _FakeCache(data)


def _builder() -> SampleBuilder:
    return SampleBuilder(SampleConfig(window=5, features=["close", "volume", "ma5", "momentum"]))


def _scanner(tmp_path: Path, data: dict) -> UnitScanner:
    paths = ScalePaths(tmp_path)
    paths.ensure()
    return UnitScanner(paths, _builder(), _FakeLoader(data))


# --------------------------------------------------------------------------- #
# Unit
# --------------------------------------------------------------------------- #
def test_unit_roundtrip():
    u = Unit.make("600519", 2015)
    assert u.unit_id == "600519_2015"
    assert u.start == "20150101" and u.end == "20151231"
    u2 = Unit.parse("600519_2015")
    assert u2 == u
    assert Unit.from_dict(u.to_dict()) == u


def test_unit_parse_last_underscore():
    """以最后一个 ``_`` 切分，兼容含下划线 symbol。"""
    u = Unit.parse("600519_SH_2015")
    assert u.symbol == "600519_SH"
    assert u.year == 2015


# --------------------------------------------------------------------------- #
# 扫描 + 划分
# --------------------------------------------------------------------------- #
def test_scan_statuses_available_insufficient_missing(tmp_path):
    data = {
        "600519": _make_df("2014-01-02", 520),   # 覆盖 2014~2015
        "000001": _make_df("2015-12-01", 5),      # 太短 → insufficient
    }
    scanner = _scanner(tmp_path, data)
    units = scanner.scan_units(["600519", "000001", "300750"], (2014, 2015))
    assert len(units) == 6  # 3 代码 × 2 年
    st = scanner.last_status
    assert st["600519_2014"] == "available"
    assert st["600519_2015"] == "available"
    assert st["000001_2015"] == "insufficient"
    assert st["300750_2014"] == "missing"
    assert st["300750_2015"] == "missing"


def test_write_units_distinguishes_statuses(tmp_path):
    data = {"600519": _make_df("2014-01-02", 520)}
    scanner = _scanner(tmp_path, data)
    units = scanner.scan_units(["600519", "000001"], (2014, 2015))
    p = scanner.write_units(units, scanner.last_status, scanner.last_steps)
    payload = json.loads(Path(p).read_text(encoding="utf-8"))
    assert payload["counts"]["available"] >= 1
    assert payload["counts"]["missing"] >= 1
    statuses = {r["unit_id"]: r["status"] for r in payload["units"]}
    assert statuses["000001_2014"] == "missing"


def test_write_units_reason_field(tmp_path):
    """非 available 单元显式给出 reason：missing→no_cache / insufficient→too_few_steps（P2-2）。"""
    data = {
        "600519": _make_df("2014-01-02", 520),
        "000001": _make_df("2014-01-02", 3),   # 太短 → insufficient
    }
    scanner = _scanner(tmp_path, data)
    units = scanner.scan_units(["600519", "000001", "300750"], (2014, 2015))
    p = scanner.write_units(units, scanner.last_status, scanner.last_steps)
    payload = json.loads(Path(p).read_text(encoding="utf-8"))
    reasons = {r["unit_id"]: r["reason"] for r in payload["units"]}

    assert reasons["600519_2014"] == ""               # available → 无原因
    assert reasons["000001_2014"] == "too_few_steps"  # insufficient
    assert reasons["300750_2014"] == "no_cache"       # missing


def test_split_reproducible_bytes_and_ratio(tmp_path):
    """同种子两次 ``split.json`` sha256 相同；train+detect == available；≈8:2。"""
    data = {
        "600519": _make_df("2013-01-02", 1700, seed=1),
        "000001": _make_df("2013-01-02", 1700, seed=2),
    }
    scanner = _scanner(tmp_path, data)
    units = scanner.scan_units(["600519", "000001"], (2013, 2018))  # 12 单元
    available = [u for u in units if scanner.last_status[u.unit_id] == "available"]
    assert len(available) == 12

    def _run() -> Path:
        plan = scanner.build_split(
            available, 42, {"train": 0.8, "detect": 0.2}, "sha256:x",
            "configs/symbols_hs300_full.yaml", (2013, 2018), len(units), 2,
        )
        return scanner.write_split(plan)

    p1 = _run()
    h1 = hashlib.sha256(Path(p1).read_bytes()).hexdigest()
    p2 = _run()
    h2 = hashlib.sha256(Path(p2).read_bytes()).hexdigest()
    assert h1 == h2

    d = json.loads(Path(p1).read_text(encoding="utf-8"))
    assert "created_at" not in d  # 无时钟字段
    assert d["algorithm"] == SPLIT_ALGORITHM
    train, detect = d["train_units"], d["detect_units"]
    assert set(train) | set(detect) == {u.unit_id for u in available}
    assert set(train) & set(detect) == set()
    assert abs(len(train) - round(12 * 0.8)) <= 1


def test_split_plan_stage_units():
    plan = SplitPlan(train_units=["a_2020"], detect_units=["a_2021"])
    assert plan.stage_units("train") == ["a_2020"]
    assert plan.stage_units("detect") == ["a_2021"]


# --------------------------------------------------------------------------- #
# 原子写
# --------------------------------------------------------------------------- #
def test_atomic_write_json_no_tmp_left(tmp_path):
    p = tmp_path / "sub" / "x.json"
    ret = atomic_write_json({"a": 1, "中文": "值"}, p)
    assert Path(ret) == p
    assert json.loads(p.read_text(encoding="utf-8"))["中文"] == "值"
    assert list(p.parent.glob("*.tmp")) == []
