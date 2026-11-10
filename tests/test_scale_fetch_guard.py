"""``scale fetch`` 数据获取防护的回归测试（防污染 / 容错 / 节流）。

背景（真实缺陷，2026-10-10 大规模拉取时暴露）：

1. ``fetch_all`` 里 ``fetch_symbol`` **未被 try 包裹** —— 任一标的在全部数据源
   失败时抛 ``DataSourceError``，会**中断整批**（长名单场景下前功尽弃）。
2. 无节流：行情源（腾讯 ``web.ifzq.gtimg.cn``）存在窗口速率风控，连续快速拉
   ~150 只即返回 ``501``（WAF 风控页）。
3. 隐式 synthetic 兜底会**落盘假数据**：首次拉取（无既有真实缓存）时
   ``_should_protect_cache`` 不生效，合成数据被写入并被后续运行"缓存命中"，
   假数据永久污染训练集。

修复后新增能力：``fetch_all(..., throttle=, on_error="skip")``；``scale fetch``
默认**禁用隐式 synthetic 兜底**（``--allow-synthetic`` 才启用）。
"""

from __future__ import annotations

import time

import pytest

from stock_ai.config.schema import DataConfig
from stock_ai.data.loader import StockDataLoader
from stock_ai.utils.errors import ConfigError, DataSourceError
from tests.test_data_loader import _make_df


class _SelectiveSource:
    """对**指定** symbol 抛错、其余返回合法日线的测试源（用于模拟部分失败）。"""

    name = "selective"

    def __init__(self, bad: set[str] | None = None) -> None:
        self.bad = set(bad or ())

    def fetch(self, symbol: str, start: str, end: str):  # noqa: ANN201
        if symbol in self.bad:
            raise RuntimeError(f"boom {symbol}")
        return _make_df("2023-01-02", 10)

    def probe(self, symbol: str) -> bool:  # pragma: no cover - trivial
        return symbol not in self.bad


# --------------------------------------------------------------------------- #
# on_error 语义
# --------------------------------------------------------------------------- #
def test_on_error_skip_continues_after_failure(tmp_path):
    """``on_error="skip"``：单只失败不中断整批，失败记入 last_failures 且不落盘。"""
    cfg = DataConfig(symbols=["600000", "600001", "600002"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_SelectiveSource({"600001"})])

    frames = loader.fetch_all(start="20230101", end="20231231", force=True, on_error="skip")

    assert set(frames) == {"600000", "600002"}
    assert set(loader.last_failures) == {"600001"}
    assert not loader.cache.exists("600001")          # 失败绝不写假数据
    assert loader.cache.exists("600000")


def test_on_error_raise_is_default_and_aborts(tmp_path):
    """默认 ``on_error="raise"``：保持既有行为——失败即上抛（不静默吞掉）。"""
    cfg = DataConfig(symbols=["600000", "600001", "600002"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_SelectiveSource({"600002"})])

    with pytest.raises(DataSourceError):
        loader.fetch_all(start="20230101", end="20231231", force=True)


def test_on_error_invalid_value_raises_config_error(tmp_path):
    """非法 ``on_error`` 必须前置报错（白名单约定），不得静默退化。"""
    cfg = DataConfig(symbols=["600000"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_SelectiveSource()])

    with pytest.raises(ConfigError):
        loader.fetch_all(start="20230101", end="20231231", force=True, on_error="bogus")


# --------------------------------------------------------------------------- #
# throttle 节流
# --------------------------------------------------------------------------- #
def test_throttle_sleeps_between_symbols_only(tmp_path, monkeypatch):
    """``throttle``：N 只之间只 sleep N-1 次，且首次请求前不睡眠。"""
    cfg = DataConfig(symbols=["600000", "600001", "600002"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_SelectiveSource()])

    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(float(s)))

    loader.fetch_all(start="20230101", end="20231231", force=True, throttle=1.5)

    assert slept == [1.5, 1.5]


def test_throttle_default_zero_does_not_sleep(tmp_path, monkeypatch):
    """默认 ``throttle=0``：完全不引入睡眠（既有行为零影响）。"""
    cfg = DataConfig(symbols=["600000", "600001"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_SelectiveSource()])

    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(float(s)))

    loader.fetch_all(start="20230101", end="20231231", force=True)

    assert slept == []


# --------------------------------------------------------------------------- #
# scale fetch CLI：默认禁用隐式 synthetic 兜底
# --------------------------------------------------------------------------- #
def test_scale_fetch_cli_filters_synthetic_by_default(tmp_path, monkeypatch):
    """``scale fetch`` 默认把 synthetic 从源链剔除（不写假数据），失败记 FAIL。"""
    from click.testing import CliRunner

    from stock_ai.cli.scale_cmd import scale_fetch

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "data:\n"
        f"  cache_dir: {tmp_path.as_posix()}/raw\n"
        "  symbols: ['600000']\n"
        "  start: '20230101'\n"
        "  end: '20231231'\n"
        "paths:\n"
        f"  root: {tmp_path.as_posix()}\n",
        encoding="utf-8",
    )

    captured: dict = {}

    def _fake_fetch_all(self, syms, start, end, force=False, throttle=0.0, on_error="raise"):
        captured["sources"] = [getattr(s, "name", "?") for s in self.sources]
        captured["throttle"] = throttle
        captured["on_error"] = on_error
        self.last_failures = {s: "simulated" for s in syms}
        self._symbol_sources = {}
        return {}

    monkeypatch.setattr(StockDataLoader, "fetch_all", _fake_fetch_all)

    res = CliRunner().invoke(
        scale_fetch, ["--config", str(cfg_file), "--symbols", "600000", "--throttle", "2.5"]
    )

    assert res.exit_code == 0, res.output
    assert "synthetic" not in captured["sources"]      # 默认剔除合成源
    assert captured["throttle"] == 2.5
    assert captured["on_error"] == "skip"              # 逐只容错
    assert "失败 1 只" in res.output
    assert (tmp_path / "raw" / "fetch_failed.txt").exists()


def test_scale_fetch_cli_explicit_synthetic_not_filtered(tmp_path, monkeypatch):
    """显式 ``--source synthetic`` 尊重用户意图，不剔除合成源（离线能力保留）。"""
    from click.testing import CliRunner

    from stock_ai.cli.scale_cmd import scale_fetch

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "data:\n"
        f"  cache_dir: {tmp_path.as_posix()}/raw\n"
        "  symbols: ['600000']\n"
        "paths:\n"
        f"  root: {tmp_path.as_posix()}\n",
        encoding="utf-8",
    )

    captured: dict = {}

    def _fake_fetch_all(self, syms, start, end, force=False, throttle=0.0, on_error="raise"):
        captured["sources"] = [getattr(s, "name", "?") for s in self.sources]
        self.last_failures = {}
        self._symbol_sources = {}
        return {}

    monkeypatch.setattr(StockDataLoader, "fetch_all", _fake_fetch_all)

    res = CliRunner().invoke(
        scale_fetch,
        ["--config", str(cfg_file), "--symbols", "600000", "--source", "synthetic"],
    )

    assert res.exit_code == 0, res.output
    assert captured["sources"] == ["synthetic"]
