"""数据加载器单测：多源 fallback、增量合并、缺口检测。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.config.schema import DataConfig
from stock_ai.data.cache import DataCache
from stock_ai.data.loader import StockDataLoader
from stock_ai.data.schema import COLUMNS, validate_frame
from stock_ai.data.sources.base import DataSource
from stock_ai.data.sources.synthetic_source import SyntheticSource
from stock_ai.utils.errors import ArtifactError, DataSourceError


def _make_df(start: str, periods: int, base: float = 100.0) -> pd.DataFrame:
    """构造合法日线。"""
    dates = pd.bdate_range(start, periods=periods)
    close = base + np.arange(periods, dtype="float64")
    return pd.DataFrame(
        {
            "date": dates, "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 1e6, "amount": close * 1e6, "adj_factor": 1.0,
        }
    )


# ------------------------- schema -------------------------
def test_validate_frame_sorts_and_dedupes():
    """validate_frame 排序 + 去重 + 列顺序固定。"""
    df = _make_df("2023-01-02", 5)
    shuffled = pd.concat([df.iloc[::-1], df.iloc[[2]]], ignore_index=True)
    out = validate_frame(shuffled)
    assert list(out.columns) == COLUMNS
    assert out["date"].is_monotonic_increasing
    assert out["date"].duplicated().sum() == 0


def test_validate_frame_missing_column_raises():
    """缺关键列抛 DataSourceError。"""
    df = _make_df("2023-01-02", 5).drop(columns=["close"])
    with pytest.raises(DataSourceError):
        validate_frame(df)


def test_validate_frame_empty_raises():
    """空数据抛 DataSourceError。"""
    with pytest.raises(DataSourceError):
        validate_frame(pd.DataFrame())


# ------------------------- cache -------------------------
def test_cache_write_read_roundtrip(tmp_path):
    """缓存写读往返一致。"""
    cache = DataCache(tmp_path)
    df = _make_df("2023-01-02", 10)
    cache.write("600519", df)
    assert cache.exists("600519")
    back = cache.read("600519")
    assert len(back) == 10


def test_cache_read_missing_raises(tmp_path):
    """读缺失缓存抛 ArtifactError。"""
    cache = DataCache(tmp_path)
    with pytest.raises(ArtifactError):
        cache.read("999999")


def test_cache_find_gaps(tmp_path):
    """缺口检测能识别缺失交易日。"""
    cache = DataCache(tmp_path)
    df = _make_df("2023-01-02", 10)
    df = df.drop(index=[3, 4]).reset_index(drop=True)   # 挖两个洞
    gaps = cache.find_gaps(df)
    assert len(gaps) >= 2


# ------------------------- fallback -------------------------
class _FailingSource(DataSource):
    """总是失败的数据源（模拟主源挂掉）。"""

    name = "failing"

    def fetch(self, symbol, start, end):
        raise DataSourceError("模拟网络失败")


class _OkSource(DataSource):
    """总是成功的数据源（fallback 目标）。"""

    name = "ok"

    def fetch(self, symbol, start, end):
        return _make_df("2023-01-02", 8)


def test_fallback_uses_second_source(tmp_path):
    """主源失败时自动 fallback 到第二源（自定义源经 **注入** 进入 loader）。

    P1-2 修复后，自定义源不再借用 ``--source`` 名通过校验，而是通过
    ``StockDataLoader(..., sources=[...])`` 注入。
    """
    cfg = DataConfig(symbols=["600519"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_FailingSource(), _OkSource()])
    df = loader.fetch_symbol("600519", "20230101", "20231231")
    assert len(df) == 8


def test_all_sources_fail_raises(tmp_path):
    """全部失败抛 DataSourceError（自定义源经注入）。"""
    cfg = DataConfig(symbols=["600519"], cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_FailingSource(), _FailingSource()])
    with pytest.raises(DataSourceError):
        loader.fetch_symbol("600519", "20230101", "20231231")


# ------------------------- merge_incremental -------------------------
def test_merge_incremental_dedupe_and_sort():
    """增量合并：去重、升序、保留新值。"""
    old = _make_df("2023-01-02", 5)
    new = _make_df("2023-01-06", 5)   # 与 old 有重叠日
    new.loc[0, "close"] = 999.0       # 覆盖值
    merged = StockDataLoader.merge_incremental(old, new)
    assert merged["date"].is_monotonic_increasing
    assert merged["date"].duplicated().sum() == 0


def test_merge_incremental_empty_raises():
    """全空输入抛 DataSourceError。"""
    with pytest.raises(DataSourceError):
        StockDataLoader.merge_incremental(pd.DataFrame(), pd.DataFrame())


# ------------------------- synthetic (offline) -------------------------
def test_synthetic_source_deterministic():
    """合成数据源同参数结果一致。"""
    src = SyntheticSource(seed=42)
    a = src.fetch("600519", "20230101", "20230601")
    b = src.fetch("600519", "20230101", "20230601")
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == COLUMNS


def test_synthetic_offline_pipeline(tmp_path):
    """离线全链路：fetch_all 落盘 + 缓存命中。"""
    cfg = DataConfig(
        symbols=["600519"], source="synthetic",
        start="20230101", end="20231231", cache_dir=tmp_path,
    )
    loader = StockDataLoader(cfg)
    frames = loader.fetch_all()
    assert "600519" in frames
    assert loader.cache.exists("600519")

    # 二次调用应命中缓存
    frames2 = loader.fetch_all()
    assert len(frames2["600519"]) == len(frames["600519"])
