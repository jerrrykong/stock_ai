"""隐式 synthetic 兜底不得覆盖真实缓存的回归测试。

背景：真实源（akshare/tencent/eastmoney）全部失败时，fallback 链会落到
synthetic 兜底。修复前 fetch_all / refresh_symbol 会把合成数据**静默覆盖**
已有的真实缓存。本文件验证保护逻辑：

1. 隐式兜底 + 已有真实缓存 -> 拒绝写入，返回旧缓存；
2. 用户显式 --source synthetic -> 尊重意图，允许写入；
3. 缓存本来就没有 / 来源本来就是 synthetic -> 正常写入（离线兜底语义）；
4. refresh 增量来自隐式兜底 -> 跳过合并写入，保留真实历史。
"""

from __future__ import annotations

import pandas as pd

from stock_ai.config.schema import DataConfig
from stock_ai.data.loader import StockDataLoader
from tests.test_data_loader import _FailingSource, _OkSource, _make_df


def _prefill_real_cache(tmp_path, symbol: str = "600519") -> StockDataLoader:
    """用『真实源』预写一份缓存（来源标记为非 synthetic）。

    P1-2 修复后，模拟真实源通过 **注入** 进入 loader（而非借用 ``--source`` 名）。
    """
    cfg = DataConfig(symbols=[symbol], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_OkSource()])
    loader.fetch_all(start="20230101", end="20231231", force=True)
    assert loader.cache.exists(symbol)
    return loader


def test_implicit_synthetic_does_not_overwrite_real_cache(tmp_path):
    """真实源全挂 -> synthetic 兜底被拒绝写入，返回真实缓存。"""
    loader = _prefill_real_cache(tmp_path)
    real = loader.cache.read("600519")

    # 模拟网络全挂：链上只剩失败源 + synthetic
    loader.sources = [_FailingSource()]
    from stock_ai.data.sources.synthetic_source import SyntheticSource
    loader.sources.append(SyntheticSource(seed=42))

    frames = loader.fetch_all(start="20230101", end="20240601", force=True)

    # 返回的是旧的真实数据，缓存未被覆盖
    assert len(frames["600519"]) == len(real)
    assert loader._symbol_sources["600519"] == "ok"
    after = loader.cache.read("600519")
    pd.testing.assert_frame_equal(after, real)


def test_explicit_synthetic_source_allows_write(tmp_path):
    """显式 --source synthetic 是明确意图，允许覆盖。"""
    loader = _prefill_real_cache(tmp_path)
    real_rows = len(loader.cache.read("600519"))

    loader.config.source = "synthetic"          # 模拟用户显式指定
    from stock_ai.data.sources.synthetic_source import SyntheticSource
    loader.sources = [SyntheticSource(seed=42)]

    frames = loader.fetch_all(start="20230101", end="20240601", force=True)
    assert loader._symbol_sources["600519"] == "synthetic"
    # 允许写入：合成区间约一年多交易日，行数必然不同于 8 行真实样本
    assert len(frames["600519"]) > 100


def test_synthetic_fallback_writes_when_no_cache(tmp_path):
    """缓存本不存在时，synthetic 兜底正常写入（离线可用性语义）。"""
    from stock_ai.data.sources.synthetic_source import SyntheticSource

    cfg = DataConfig(symbols=["600519"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(
        cfg, sources=[_FailingSource(), SyntheticSource(seed=42)]
    )

    frames = loader.fetch_all(start="20230101", end="20230601", force=True)
    assert "600519" in frames
    assert loader.cache.exists("600519")
    assert loader._symbol_sources["600519"] == "synthetic"


def test_refresh_with_synthetic_increment_keeps_real_history(tmp_path):
    """增量来自隐式 synthetic 兜底时跳过写入，真实历史保持不变。"""
    loader = _prefill_real_cache(tmp_path)
    real = loader.cache.read("600519")
    real_last = real["date"].max()

    loader.sources = [_FailingSource()]
    from stock_ai.data.sources.synthetic_source import SyntheticSource
    loader.sources.append(SyntheticSource(seed=42))

    out = loader.refresh_symbol("600519", today="20241231")
    # 返回旧缓存，且缓存文件未混入合成数据
    pd.testing.assert_frame_equal(out, real)
    assert loader.cache.read("600519")["date"].max() == real_last
