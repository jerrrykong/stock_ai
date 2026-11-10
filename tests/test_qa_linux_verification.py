"""QA 独立验证用例（Linux / Bash 移植轮次）。

本文件由 QA 独立编写，**不复用**工程师的自测脚本，用于覆盖可能被遗漏的场景：

1. 隐式 synthetic 兜底不得覆盖真实缓存（独立实现）。
2. meta.json 丢失但真实缓存仍在时的保护语义（健壮性）。
3. 损坏 parquet 缓存的优雅处理。
4. ``tencent_source`` 纯函数：``_load_json`` / ``_prev_day`` / ``_estimate_bars``。
5. ``TencentSource.fetch`` 的分页装配：多页合并后无重复、严格升序、覆盖 start。
6. 参数校验：非法 ``--backend`` / ``--source`` 应明确报错（回归缺陷）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_ai.config.schema import DataConfig
from stock_ai.data.loader import StockDataLoader
from stock_ai.data.schema import COLUMNS
from stock_ai.data.sources.base import DataSource
from stock_ai.data.sources.synthetic_source import SyntheticSource
from stock_ai.data.sources.tencent_source import (
    TencentSource,
    _estimate_bars,
    _load_json,
    _prev_day,
)
from stock_ai.utils.dates import to_day_key, validate_range
from stock_ai.utils.errors import ConfigError, DataSourceError


# --------------------------------------------------------------------------- #
# 辅助：构造可控数据源
# --------------------------------------------------------------------------- #
def _make_df(days: int = 8, start: str = "2023-01-02") -> pd.DataFrame:
    """构造一段合法日线（工作日近似）。"""
    dates = pd.bdate_range(start=start, periods=days)
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0 + i for i in range(n)],
            "high": [11.0 + i for i in range(n)],
            "low": [9.0 + i for i in range(n)],
            "close": [10.5 + i for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


class _RealSource(DataSource):
    """模拟一个『真实』数据源。"""

    name = "realsrc"

    def fetch(self, symbol: str, start: str, end: str):  # noqa: D401
        return _make_df()


class _FailSource(DataSource):
    """模拟拉取必定失败的源。"""

    name = "failsrc"

    def fetch(self, symbol: str, start: str, end: str):
        raise DataSourceError("模拟网络失败")


# --------------------------------------------------------------------------- #
# 1. 隐式 synthetic 兜底不得覆盖真实缓存（独立实现）
# --------------------------------------------------------------------------- #
def test_independent_implicit_synthetic_not_overwrite_real_cache(tmp_path: Path):
    # P1-2 修复后：模拟真实源通过 **注入** 进入 loader
    cfg = DataConfig(symbols=["ZZZ"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_RealSource()])

    # 预写一份真实缓存
    loader.fetch_all(start="20230101", end="20230301", force=True)
    real = loader.cache.read("ZZZ")
    bytes_before = loader.cache.path_of("ZZZ").read_bytes()

    # 真实源全挂 + synthetic 兜底
    loader.sources = [_FailSource(), SyntheticSource(seed=42)]
    frames = loader.fetch_all(start="20230101", end="20240601", force=True)

    # 返回旧真实数据，缓存字节完全未变
    pd.testing.assert_frame_equal(frames["ZZZ"], real)
    assert loader.cache.path_of("ZZZ").read_bytes() == bytes_before
    assert loader._symbol_sources["ZZZ"] == "realsrc"


def test_independent_explicit_synthetic_overwrites(tmp_path: Path):
    cfg = DataConfig(symbols=["ZZZ"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_RealSource()])
    loader.fetch_all(start="20230101", end="20230301", force=True)

    loader.config.source = "synthetic"
    loader.sources = [SyntheticSource(seed=42)]
    frames = loader.fetch_all(start="20230101", end="20240601", force=True)
    assert len(frames["ZZZ"]) > 100  # 合成约一年多交易日，显著多于 8 行


def test_refresh_increment_from_synthetic_keeps_real_history(tmp_path: Path):
    cfg = DataConfig(symbols=["ZZZ"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_RealSource()])
    loader.fetch_all(start="20230101", end="20230301", force=True)
    real = loader.cache.read("ZZZ")
    real_last = real["date"].max()

    loader.sources = [_FailSource(), SyntheticSource(seed=42)]
    out = loader.refresh_symbol("ZZZ", today="20241231")
    pd.testing.assert_frame_equal(out, real)
    assert loader.cache.read("ZZZ")["date"].max() == real_last


def test_meta_missing_still_protects_real_cache(tmp_path: Path):
    """meta.json 丢失但真实 parquet 仍在时，隐式 synthetic 兜底仍不应覆盖。

    这是 QA 额外加的健壮性场景（工程师用例未覆盖）。
    """
    cfg = DataConfig(symbols=["ZZZ"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_RealSource()])
    loader.fetch_all(start="20230101", end="20230301", force=True)
    bytes_before = loader.cache.path_of("ZZZ").read_bytes()

    loader.cache.meta_path.unlink()  # 模拟 meta.json 丢失
    loader.sources = [_FailSource(), SyntheticSource(seed=42)]
    loader.fetch_all(start="20230101", end="20240601", force=True)

    assert loader.cache.path_of("ZZZ").read_bytes() == bytes_before


# --------------------------------------------------------------------------- #
# 2. 损坏缓存优雅处理
# --------------------------------------------------------------------------- #
def test_corrupt_parquet_fetch_refetches(tmp_path: Path):
    cfg = DataConfig(symbols=["ZZZ"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_RealSource()])
    p = loader.cache.path_of("ZZZ")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a parquet")

    frames = loader.fetch_all(start="20230101", end="20230301", force=False)
    assert "ZZZ" in frames
    assert len(frames["ZZZ"]) == len(_make_df())  # 成功重拉


# --------------------------------------------------------------------------- #
# 3. tencent_source 纯函数
# --------------------------------------------------------------------------- #
def test_load_json_plain_and_jsonp():
    assert _load_json('{"a": 1}')["a"] == 1
    # JSONP 前缀
    assert _load_json('kline_day={"a": 2};')["a"] == 2
    # 结尾多余字符
    assert _load_json('{"a": 3}\n;')["a"] == 3


def test_load_json_invalid():
    with pytest.raises(DataSourceError):
        _load_json("")
    with pytest.raises(DataSourceError):
        _load_json("not json at all")
    with pytest.raises(DataSourceError):
        _load_json("[1, 2, 3]")  # 顶层非对象


def test_prev_day_cross_month_and_year():
    assert _prev_day("2024-03-01") == "20240229"  # 闰年跨月
    assert _prev_day("20240101") == "20231231"  # 跨年
    assert _prev_day("2024-01-01") == "20231231"


def test_estimate_bars_bounds():
    assert _estimate_bars("20240101", "20241231") <= 800  # 上限
    assert _estimate_bars("20240101", "20240102") == 320  # 下界
    assert _estimate_bars("garbage", "x") == 320  # 无法解析回退


# --------------------------------------------------------------------------- #
# 4. fetch 分页装配（用假 requests 复现腾讯『返回区间尾部』语义）
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, rows):
        self._rows = rows
        self.text = json.dumps({"data": {"sh600000": {"qfqday": rows}}})

    def raise_for_status(self):
        return None


def _install_fake_requests(monkeypatch, master_rows):
    """让 requests.get 返回『以 end 结尾、最多 count 根』的尾部数据。

    ``master_rows`` 为 ``[date, open, close, high, low, volume]`` 行列表，
    与腾讯 ``qfqday`` 结构一致。
    """
    import requests

    def fake_get(url, params=None, headers=None, timeout=None):
        param = params["param"]
        _, _, s, e, count, _adj = param.split(",")
        count = int(count)
        e_s = e.replace("-", "")
        s_s = s.replace("-", "")
        window = [r for r in master_rows if s_s <= r[0].replace("-", "") <= e_s]
        return _FakeResp(window[-count:])

    monkeypatch.setattr(requests, "get", fake_get)


def test_fetch_pagination_no_dup_sorted_covers_start(monkeypatch):
    # 主序列须显著长于单页上限（800 根），否则一次请求即可拿全，无法验证分页装配。
    master = pd.bdate_range(start="2021-01-04", periods=1200)  # > 800 -> 必分页
    master_rows = [
        [d.strftime("%Y-%m-%d"), 10.0, 10.5, 11.0, 9.5, 1000.0] for d in master
    ]
    _install_fake_requests(monkeypatch, master_rows)

    src = TencentSource(adjust="qfq", retry_times=1)
    start = "20210104"
    end = master_rows[-1][0].replace("-", "")
    df = src.fetch("600000", start, end)

    dates = df["date"].dt.strftime("%Y%m%d")
    assert dates.is_monotonic_increasing
    assert dates.duplicated().sum() == 0
    # 覆盖 start（首日 <= start）
    assert dates.iloc[0] <= start
    # 强制多页：单页最多 800 根，1200 行必须分页装配才能拿全
    assert len(df) >= 1150, f"仅 {len(df)} 行 —— 未真正分页装配"


# --------------------------------------------------------------------------- #
# 5. 参数校验回归（当前源码存在缺陷 -> 这些用例会失败，作为缺陷证据）
# --------------------------------------------------------------------------- #
def test_invalid_backend_is_rejected(monkeypatch):
    """非法 --backend 必须明确报错，而非静默按 auto 处理。"""
    from stock_ai.config.schema import TrainConfig
    from stock_ai.train.factory import TrainerFactory

    class _Env:  # 占位，不应被真正使用
        pass

    cfg = TrainConfig(algo="ppo", backend="bogus")
    with pytest.raises(ConfigError):
        TrainerFactory.create(cfg, _Env(), preference="bogus")


def test_invalid_source_is_rejected():
    """非法 --source 必须明确报错，而非静默使用默认源链。"""
    cfg = DataConfig(symbols=["600519"], source="nosuch")
    with pytest.raises(ConfigError):
        StockDataLoader(cfg)


# --------------------------------------------------------------------------- #
# 第 2 轮新增：P2-1 反向语义 / P2-6 边界 / P1-2 白名单残余漏洞审计
# --------------------------------------------------------------------------- #
class _UnregisteredSource(DataSource):
    """自定义源：``name`` 是合法子类名，但**未注册**进 loader 内置 registry。

    用于审计 P1-2 修复采用「静态白名单 ∪ 所有 DataSource 子类名」后，
    是否留下"校验通过但被静默忽略"的残余漏洞。
    """

    name = "unregistered_custom"

    def fetch(self, symbol: str, start: str, end: str):
        return _make_df()


def test_p1_2_no_cache_implicit_synthetic_still_writes(tmp_path: Path):
    """反向语义①：首次无缓存时，隐式 synthetic 兜底应正常写入（不被过度保护误伤）。"""
    cfg = DataConfig(symbols=["ZZZ"], source="tencent", cache_dir=tmp_path)
    loader = StockDataLoader(cfg)
    loader.sources = [_FailSource(), SyntheticSource(seed=42)]
    frames = loader.fetch_all(start="20230101", end="20230601", force=True)
    assert loader.cache.exists("ZZZ")
    assert loader._symbol_sources["ZZZ"] == "synthetic"
    assert len(frames["ZZZ"]) > 100


def test_p1_2_explicit_synthetic_still_writes(tmp_path: Path):
    """反向语义②：用户显式 --source synthetic 仍应正常写入。"""
    cfg = DataConfig(symbols=["ZZZ"], source="synthetic", cache_dir=tmp_path)
    loader = StockDataLoader(cfg)
    frames = loader.fetch_all(start="20230101", end="20230601", force=True)
    assert loader.cache.exists("ZZZ")
    assert len(frames["ZZZ"]) > 100


@pytest.mark.parametrize("name", ["unregistered_custom", "plugintest", "nosuch"])
def test_p1_2_residual_unregistered_source_must_error(tmp_path: Path, name: str):
    """P1-2 残余漏洞（已修复）：未注册进 registry 的源名必须**明确报错**。

    修复后 ``SUPPORTED_SOURCES`` 与运行时 registry 同源于 ``_SOURCE_FACTORIES``，
    ``_build_sources`` 对不在 registry 的 ``config.source`` 直接抛 :class:`ConfigError`，
    因此即便 ``name`` 是合法 ``DataSource`` 子类名，也不会被"校验通过后静默忽略"。
    自定义/插件源需改用 ``StockDataLoader(..., sources=[...])`` 注入。
    """
    cfg = DataConfig(symbols=["600519"], source=name, cache_dir=tmp_path)
    with pytest.raises(ConfigError):
        StockDataLoader(cfg)


def test_p1_2_unregistered_source_via_injection_works(tmp_path: Path):
    """扩展路径：自定义/插件源经 **注入** 仍可正常进入 loader（可扩展性保留）。"""
    cfg = DataConfig(symbols=["ZZZ"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_UnregisteredSource()])
    assert any(s.name == "unregistered_custom" for s in loader.sources)


def test_p1_2_source_name_case_insensitive(tmp_path: Path):
    """大小写不一致（--source TENCENT）应被正确识别。"""
    cfg = DataConfig(symbols=["600519"], source="TENCENT", cache_dir=tmp_path)
    loader = StockDataLoader(cfg)  # 不应抛错
    assert any(getattr(s, "name", "") == "tencent" for s in loader.sources)


def test_p2_6_validate_range_boundaries():
    """P2-6：start>end 报错；start==end / 单日 / 归一后相等 / 不可解析 均不误伤。"""
    with pytest.raises(ConfigError):
        validate_range("20241231", "20120101", label="区间", hint="x")
    validate_range("20240101", "20240101")          # 相等
    validate_range("2024-01-01", "20240101")        # 归一后相等
    validate_range("", "20240101")                  # 不可解析 -> 不误伤
    validate_range("garbage", "20240101")           # 不可解析 -> 不误伤


def test_p2_6_to_day_key_normalization():
    """to_day_key 归一边界。"""
    assert to_day_key("2024-01-01") == "20240101"
    assert to_day_key("2024/01/01") == "20240101"
    assert to_day_key("20240101") == "20240101"
    assert to_day_key("") == ""
    assert to_day_key("2024") == ""  # 位数不足

