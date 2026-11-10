"""walk-forward 切分单测：零重叠 + embargo。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_ai.samples.split import date_to_int, walk_forward_split


def _bdate_strings(start: str, periods: int) -> "np.ndarray":
    """生成交易日字符串数组。"""
    return pd.bdate_range(start, periods=periods).strftime("%Y%m%d").to_numpy()


def test_no_overlap_and_order():
    """train/test 索引零重叠，且 train 整体早于 test。"""
    dates = _bdate_strings("2020-01-02", 900)   # 约 3.5 年，覆盖到 2023
    train_idx, test_idx = walk_forward_split(
        dates, ("20200102", "20211231"), ("20220103", "20230101"), embargo_days=5
    )
    assert train_idx.size > 0
    assert test_idx.size > 0
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert train_idx.max() < test_idx.min()


def test_embargo_gap():
    """train 末端与 test 首端之间至少隔 embargo_days 个交易日。"""
    dates = _bdate_strings("2020-01-02", 900)
    embargo = 10
    train_idx, test_idx = walk_forward_split(
        dates, ("20200102", "20211231"), ("20220103", "20230101"), embargo_days=embargo
    )
    gap = test_idx.min() - train_idx.max()
    assert gap >= embargo


def test_embargo_zero_allows_adjacency():
    """embargo=0 时允许紧邻。"""
    dates = _bdate_strings("2020-01-02", 900)
    train_idx, test_idx = walk_forward_split(
        dates, ("20200102", "20211231"), ("20220103", "20230101"), embargo_days=0
    )
    assert test_idx.min() - train_idx.max() >= 1


def test_invalid_range_raises():
    """start > end 抛 ConfigError。"""
    from stock_ai.utils.errors import ConfigError

    dates = _bdate_strings("2020-01-02", 100)
    with pytest.raises(ConfigError):
        walk_forward_split(dates, ("20220101", "20200101"), ("20230101", "20240101"), 5)


def test_date_to_int():
    """日期格式归一化。"""
    assert date_to_int("2023-01-03") == 20230103
    assert date_to_int("20230103") == 20230103
    assert date_to_int(20230103) == 20230103


def test_empty_when_no_data_in_range():
    """区间内无数据时返回空索引。"""
    dates = _bdate_strings("2020-01-02", 100)
    train_idx, test_idx = walk_forward_split(
        dates, ("20300101", "20301231"), ("20310101", "20311231"), embargo_days=5
    )
    assert train_idx.size == 0
    assert test_idx.size == 0
