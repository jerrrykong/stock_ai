"""``--adjust``（复权口径）开关与 P1-2 registry 单一真源的离线测试。

覆盖：

A. 复权值域：
   1. ``qfq`` / ``hfq`` / ``none`` 三个合法值可用；
   2. 非法值抛 :class:`ConfigError`；历史空串 ``""`` 归一化为 ``none``。

B. 各数据源 map（离线 mock，不触网）：
   3. 腾讯 ``qfq/hfq/none -> qfqday/hfqday/day`` 字段与 param 复权标识；
   4. AKShare ``none -> ""``（mock ``akshare`` 模块）；
   5. 东方财富 ``none -> 0``（fqt）。

C. 缓存一致性（A5）：
   6. ``meta.json`` 记录 ``adjust``，``data info`` 回读；
   7. 复权口径变更 -> 旧缓存不命中、强制重拉（计数源证明）；
   8. refresh 口径不一致 -> **不做增量拼接**，退化为全量重拉。

D. P1-2 registry 单一真源：
   9. ``SUPPORTED_SOURCES`` 由 ``_SOURCE_FACTORIES`` 唯一推导；
  10. 未注册名（含合法子类名）构造 loader 明确报错；
  11. 大小写不一致（``--source TENCENT``）仍被正确识别。

E. CLI：``--adjust`` 非法值 -> 非零退出。
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from stock_ai.config.schema import DataConfig, SUPPORTED_ADJUSTS, normalize_adjust
from stock_ai.data.loader import (
    SUPPORTED_SOURCES,
    StockDataLoader,
    _SOURCE_FACTORIES,
)
from stock_ai.data.sources.akshare_source import AkShareSource
from stock_ai.data.sources.base import DataSource
from stock_ai.data.sources.eastmoney_source import EastMoneySource
from stock_ai.data.sources.synthetic_source import SyntheticSource
from stock_ai.data.sources.tencent_source import TencentSource
from stock_ai.utils.errors import ConfigError, DataSourceError

# 复用既有测试的数据构造器，避免重复维护
from tests.test_data_loader import _make_df

#: 仓库内默认配置的绝对路径（CLI 会 chdir，需绝对路径）
_CFG = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")

#: 项目 ``src`` 目录（供子进程 ``-m`` 导入）
_SRC = str(Path(__file__).resolve().parents[1] / "src")


# --------------------------------------------------------------------------- #
# 计数/记录型自定义源（经注入进入 loader，不借用 --source 名）
# --------------------------------------------------------------------------- #
class _RecordingSource(DataSource):
    """记录每次 ``fetch`` 的 ``(start, end)``，并返回固定 DataFrame。"""

    name = "recordsrc"

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.calls: list[tuple[str, str]] = []

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        self.calls.append((start, end))
        return self._df


# =========================================================================== #
# A. 复权值域
# =========================================================================== #
def test_supported_adjusts_constant():
    """白名单常量恰好三个合法值。"""
    assert SUPPORTED_ADJUSTS == frozenset({"qfq", "hfq", "none"})


@pytest.mark.parametrize("adjust", ["qfq", "hfq", "none"])
def test_adjust_legal_values(adjust: str):
    """三个合法值可构造 DataConfig 且原样保留。"""
    cfg = DataConfig(adjust=adjust)
    assert cfg.adjust == adjust
    assert normalize_adjust(adjust) == adjust


@pytest.mark.parametrize("bad", ["bogus", "fq", "pre", "hqf", 1, 2.5])
def test_adjust_illegal_raises(bad):
    """非法值（含数字）抛 ConfigError。"""
    with pytest.raises(ConfigError):
        normalize_adjust(bad)


def test_adjust_whitespace_and_case_normalized():
    """带空格 / 大写输入被 strip+lower 后合法。"""
    assert normalize_adjust("QFQ ") == "qfq"
    assert normalize_adjust("  hfq") == "hfq"


def test_adjust_illegal_raises_dataconfig():
    """DataConfig 构造期即校验非法复权值。"""
    with pytest.raises(ConfigError):
        DataConfig(adjust="bogus")


@pytest.mark.parametrize("raw", ["", None, "  ", "None"])
def test_adjust_empty_normalizes_to_none(raw):
    """历史"不复权"空串写法归一化为 ``none``，而非拒绝。"""
    assert normalize_adjust(raw) == "none"
    assert DataConfig(adjust=raw).adjust == "none"


def test_adjust_uppercase_normalized():
    """大写输入被归一化为小写 canonical 值。"""
    assert normalize_adjust("HFQ") == "hfq"
    assert DataConfig(adjust="QFQ").adjust == "qfq"


# =========================================================================== #
# B. 各数据源 map（离线 mock）
# =========================================================================== #
def test_tencent_adjust_map_constant():
    """腾讯 map：``qfq/hfq/none -> (字段, param 标识)``。"""
    from stock_ai.data.sources.tencent_source import _ADJUST_MAP

    assert _ADJUST_MAP == {
        "qfq": ("qfqday", "qfq"),
        "hfq": ("hfqday", "hfq"),
        "none": ("day", ""),
    }


class _TencentFakeResp:
    """腾讯 mock 响应：仅包含指定字段。"""

    def __init__(self, node: dict) -> None:
        self.text = json.dumps({"data": {"sh600519": node}})

    def raise_for_status(self) -> None:
        return None


@pytest.mark.parametrize(
    "adjust,field,api_suffix",
    [("qfq", "qfqday", ",qfq"), ("hfq", "hfqday", ",hfq"), ("none", "day", ",")],
)
def test_tencent_adjust_field_and_param(monkeypatch, adjust, field, api_suffix):
    """腾讯 fetch 依据 adjust 读取正确字段，并在 param 末尾带上复权标识。"""
    import requests

    captured: list[str] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        captured.append(params["param"])
        node = {field: [["2023-01-03", 10.0, 10.5, 11.0, 9.5, 1000.0]]}
        return _TencentFakeResp(node)

    monkeypatch.setattr(requests, "get", fake_get)

    src = TencentSource(adjust=adjust, retry_times=1)
    df = src.fetch("600519", "20230103", "20230103")
    assert len(df) == 1
    assert captured, "应至少发起一次请求"
    assert captured[-1].endswith(api_suffix), captured[-1]


def test_tencent_unknown_adjust_raises():
    """腾讯源非法 adjust 抛 ConfigError（不静默）。"""
    with pytest.raises(ConfigError):
        TencentSource(adjust="bogus")


@pytest.mark.parametrize("adjust,expected", [("qfq", "qfq"), ("hfq", "hfq"), ("none", "")])
def test_akshare_adjust_arg(monkeypatch, adjust, expected):
    """AKShare：``none -> ""``（不复权），不依赖真实 akshare。"""
    captured: dict[str, object] = {}

    fake = types.ModuleType("akshare")

    def stock_zh_a_hist(symbol, period, start_date, end_date, adjust):  # noqa: ANN001
        captured["adjust"] = adjust
        return pd.DataFrame(
            {
                "日期": ["2023-01-03"],
                "开盘": [10.0],
                "收盘": [10.5],
                "最高": [11.0],
                "最低": [9.5],
                "成交量": [1000.0],
                "成交额": [1_000_000.0],
            }
        )

    fake.stock_zh_a_hist = stock_zh_a_hist  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "akshare", fake)

    src = AkShareSource(adjust=adjust, retry_times=1)
    df = src.fetch("600519", "20230103", "20230103")
    assert captured["adjust"] == expected
    assert len(df) == 1


def test_eastmoney_adjust_map_constant():
    """东方财富 map：``qfq/hfq/none -> 1/2/0``。"""
    from stock_ai.data.sources.eastmoney_source import _ADJUST_MAP

    assert _ADJUST_MAP == {"qfq": 1, "hfq": 2, "none": 0}


@pytest.mark.parametrize("adjust,expected_fqt", [("qfq", "1"), ("hfq", "2"), ("none", "0")])
def test_eastmoney_adjust_fqt(monkeypatch, adjust, expected_fqt):
    """东方财富 fetch 的 ``fqt`` 参数随 adjust 变化。"""
    import requests

    captured: dict[str, str] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": {
                    "klines": ["2023-01-03,10.0,10.5,11.0,9.5,1000,1000000"]
                }
            }

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["fqt"] = params["fqt"]
        return _Resp()

    monkeypatch.setattr(requests, "get", fake_get)

    src = EastMoneySource(adjust=adjust, retry_times=1)
    df = src.fetch("600519", "20230103", "20230103")
    assert captured["fqt"] == expected_fqt
    assert len(df) == 1


# =========================================================================== #
# C. 缓存一致性（A5）
# =========================================================================== #
def test_meta_records_adjust(tmp_path):
    """fetch 后 ``meta.json`` 与 ``data info`` 均带复权口径。"""
    cfg = DataConfig(
        symbols=["600519"], source="synthetic", cache_dir=tmp_path,
        start="20230101", end="20230601", adjust="hfq",
    )
    loader = StockDataLoader(cfg)
    loader.fetch_all()

    meta = loader.cache.read_meta()
    assert meta["symbols"]["600519"]["adjust"] == "hfq"

    infos = loader.info()
    assert len(infos) == 1
    assert infos[0].adjust == "hfq"


def test_adjust_change_refetches_not_hit(tmp_path):
    """换复权口径 -> 旧口径缓存**不得命中**，必须重拉。"""
    df = _make_df("2023-01-02", 120)  # 覆盖 20230101~20230601
    src_qfq = _RecordingSource(df)
    cfg = DataConfig(
        symbols=["600519"], source="synthetic", cache_dir=tmp_path,
        start="20230101", end="20230601", adjust="qfq",
    )
    loader = StockDataLoader(cfg, sources=[src_qfq])
    loader.fetch_all(force=True)
    assert len(src_qfq.calls) == 1
    assert loader.cache.read_meta()["symbols"]["600519"]["adjust"] == "qfq"

    # 同口径二次调用 -> 命中缓存，不再调用数据源
    loader.fetch_all()
    assert len(src_qfq.calls) == 1, "同口径应命中缓存"

    # 换 hfq -> 不得命中 qfq 旧缓存，必须重拉
    loader.config.adjust = "hfq"
    src_hfq = _RecordingSource(df)
    loader.sources = [src_hfq]
    loader.fetch_all()
    assert len(src_hfq.calls) == 1, "口径不一致必须重拉"
    assert loader.cache.read_meta()["symbols"]["600519"]["adjust"] == "hfq"


def test_refresh_adjust_mismatch_no_incremental_mix(tmp_path):
    """refresh 口径不一致 -> 全量重拉（fetch 覆盖完整区间），绝不增量拼接。"""
    src = _RecordingSource(_make_df("2023-01-02", 120))
    cfg = DataConfig(
        symbols=["600519"], source="synthetic", cache_dir=tmp_path,
        start="20230101", end="20230601", adjust="qfq",
    )
    loader = StockDataLoader(cfg, sources=[src])
    loader.fetch_all(force=True)

    # 换 hfq 后刷新：应请求 [config.start, today] 的全量区间，而非 (last+1, today)
    loader.config.adjust = "hfq"
    src2 = _RecordingSource(_make_df("2023-01-02", 130))
    loader.sources = [src2]
    loader.refresh_symbol("600519", today="20230601")

    assert src2.calls == [("20230101", "20230601")], src2.calls
    assert loader.cache.read_meta()["symbols"]["600519"]["adjust"] == "hfq"


def test_refresh_same_adjust_uses_incremental(tmp_path):
    """口径一致时 refresh 仍走增量（起始 > config.start），证明未误伤正常路径。"""
    src = _RecordingSource(_make_df("2023-01-02", 120))
    cfg = DataConfig(
        symbols=["600519"], source="synthetic", cache_dir=tmp_path,
        start="20230101", end="20230601", adjust="qfq",
    )
    loader = StockDataLoader(cfg, sources=[src])
    loader.fetch_all(force=True)

    src2 = _RecordingSource(_make_df("2023-06-01", 10))
    loader.sources = [src2]
    loader.refresh_symbol("600519", today="20230630")
    # 增量起点应晚于 config.start
    assert src2.calls, "增量应调用数据源"
    assert src2.calls[0][0] > "20230101"


def test_refresh_adjust_mismatch_preserves_range(tmp_path, caplog):
    """P3：口径不一致全量重拉时，起点取 ``min(缓存起日, config.start)``，不静默收缩。

    构造：缓存 hfq 区间自 ``20120104`` 起、``config.start=20180101``；请求 qfq。
    期望：重拉起点 == ``20120104``（min 生效，而非缩到 2018），最终缓存仍覆盖 2012。
    """
    cached = _make_df("2012-01-04", 200)  # 起日 20120104
    src = _RecordingSource(cached)
    cfg = DataConfig(
        symbols=["600519"], source="synthetic", cache_dir=tmp_path,
        start="20180101", end="20241231", adjust="hfq",
    )
    loader = StockDataLoader(cfg, sources=[src])
    loader.fetch_all(force=True)
    assert loader.cache.read("600519")["date"].min().strftime("%Y%m%d") == "20120104"
    assert loader.cache.read_meta()["symbols"]["600519"]["adjust"] == "hfq"

    # 切 qfq（config.start=20180101 晚于缓存起日）-> 全量重拉起点应为 20120104
    loader.config.adjust = "qfq"
    src2 = _RecordingSource(_make_df("2012-01-04", 210))
    loader.sources = [src2]
    with caplog.at_level("WARNING"):
        loader.refresh_symbol("600519", today="20241231")

    # ① 只发生一次全量拉取，区间为 [min_start, end]（证明未做增量拼接）
    assert src2.calls == [("20120104", "20241231")], src2.calls
    # ② 区间未收缩，最终缓存仍覆盖 2012
    assert loader.cache.read("600519")["date"].min().strftime("%Y%m%d") == "20120104"
    # ③ 口径已切换
    assert loader.cache.read_meta()["symbols"]["600519"]["adjust"] == "qfq"
    # ④ WARNING 显式说明区间变化
    assert any("复权口径变更" in r.message for r in caplog.records)


def test_refresh_adjust_mismatch_expands_when_config_start_earlier(tmp_path):
    """P3 反向：``config.start`` 早于缓存起日时，min 取更早的 config.start（扩展区间）。"""
    cached = _make_df("2020-01-02", 60)  # 起日 20200102
    src = _RecordingSource(cached)
    cfg = DataConfig(
        symbols=["600519"], source="synthetic", cache_dir=tmp_path,
        start="20200101", end="20241231", adjust="hfq",
    )
    loader = StockDataLoader(cfg, sources=[src])
    loader.fetch_all(force=True)

    loader.config.adjust = "qfq"
    loader.config.start = "20120104"  # 比缓存起日更早
    src2 = _RecordingSource(_make_df("2012-01-04", 300))
    loader.sources = [src2]
    loader.refresh_symbol("600519", today="20241231")
    assert src2.calls == [("20120104", "20241231")], src2.calls


# =========================================================================== #
# D. P1-2 registry 单一真源
# =========================================================================== #
def test_supported_sources_derived_from_registry():
    """``SUPPORTED_SOURCES`` 与 ``_SOURCE_FACTORIES`` 唯一同源。"""
    assert SUPPORTED_SOURCES == frozenset(_SOURCE_FACTORIES)
    assert SUPPORTED_SOURCES == frozenset(
        {"akshare", "tencent", "eastmoney", "sina", "synthetic"}
    )


def test_no_legacy_subclass_scanning():
    """确认旧的"子类名扫描"真源已被删除（P1-2 单源化）。"""
    import stock_ai.data.loader as loader_mod

    assert not hasattr(loader_mod, "_known_source_names")


class _FreshCustomSource(DataSource):
    """合法子类，但 ``name`` 未注册进内置 registry。"""

    name = "fresh_custom"

    def fetch(self, symbol, start, end):  # noqa: ANN001
        raise DataSourceError("noop")


@pytest.mark.parametrize("name", ["fresh_custom", "unregistered", "nosuch"])
def test_unregistered_source_name_errors(tmp_path, name):
    """未注册名（含合法子类名）构造 loader 必须明确报错，且信息含源名。"""
    cfg = DataConfig(symbols=["600519"], source=name, cache_dir=tmp_path)
    with pytest.raises(ConfigError) as ei:
        StockDataLoader(cfg)
    assert name in str(ei.value)


def test_custom_source_via_injection_ok(tmp_path):
    """自定义源经注入可用（扩展路径）。"""
    cfg = DataConfig(symbols=["600519"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_FreshCustomSource()])
    assert any(s.name == "fresh_custom" for s in loader.sources)


def test_source_name_case_insensitive(tmp_path):
    """``--source TENCENT``（大小写不一致）仍被正确识别。"""
    cfg = DataConfig(symbols=["600519"], source="TENCENT", cache_dir=tmp_path)
    loader = StockDataLoader(cfg)
    assert any(getattr(s, "name", "") == "tencent" for s in loader.sources)


# =========================================================================== #
# E. CLI：非法 --adjust 非零退出
# =========================================================================== #
def test_cli_invalid_adjust_nonzero(tmp_path):
    """``data fetch --adjust bogus`` 应在进入网络前报错并非零退出。"""
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-m", "stock_ai.data.fetch_cli", "fetch",
        "--config", _CFG, "--symbols", "600519",
        "--adjust", "bogus", "--cache-dir", str(tmp_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=120, cwd=str(tmp_path)
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "复权" in (proc.stdout + proc.stderr)
