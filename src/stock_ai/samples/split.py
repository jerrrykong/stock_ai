"""walk-forward 时间切分 + embargo（纯函数，K14）。

严禁打乱顺序、严禁 train/test 重叠、边界必须留 embargo 隔离带。
"""

from __future__ import annotations

import numpy as np

from stock_ai.utils.errors import ConfigError
from stock_ai.utils.logging import LoggerFactory

__all__ = ["walk_forward_split", "date_to_int", "merge_masks"]

logger = LoggerFactory.get(__name__)


def date_to_int(value: str) -> int:
    """``"20230101"`` / ``"2023-01-01"`` -> ``20230101``（int）。

    Args:
        value: 日期字符串。

    Returns:
        8 位整数日期。
    """
    s = str(value).replace("-", "").strip()
    return int(s)


def walk_forward_split(
    dates: "np.ndarray",
    train_range: tuple[str, str],
    test_range: tuple[str, str],
    embargo_days: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    """按时间顺序切分，返回 train / test 的**位置索引**。

    规则：
        - ``train_idx``：``date ∈ [train_start, train_end]``
        - ``test_idx``：``date ∈ [test_start, test_end]``
        - 二者不重叠；且在 train 之后、test 之前至少保留 ``embargo_days`` 个交易日
          的隔离带（即训练区间末端回退 ``embargo_days`` 个交易日，避免边界泄漏）。

    Args:
        dates: ``(T,)`` 的日期数组（元素为 ``YYYYMMDD`` 字符串或整数）。
        train_range: ``(start, end)``，格式 ``YYYYMMDD``。
        test_range: ``(start, end)``，格式 ``YYYYMMDD``。
        embargo_days: embargo 交易日数（>=0）。

    Returns:
        ``(train_idx, test_idx)``，均为 ``int64`` 位置索引数组，升序。

    Raises:
        ConfigError: 区间参数非法（start > end）。
    """
    ts, te = date_to_int(train_range[0]), date_to_int(train_range[1])
    ss, se = date_to_int(test_range[0]), date_to_int(test_range[1])
    if ts > te:
        raise ConfigError(f"train_range 非法: {train_range}")
    if ss > se:
        raise ConfigError(f"test_range 非法: {test_range}")

    d = np.asarray([date_to_int(x) for x in dates], dtype="int64")
    order = np.argsort(d, kind="stable")
    d_sorted = d[order]

    # train 区间
    train_mask_sorted = (d_sorted >= ts) & (d_sorted <= te)
    train_positions = np.nonzero(train_mask_sorted)[0]

    # 应用 embargo：从 train 尾部裁掉 embargo_days 个交易日
    if embargo_days > 0 and len(train_positions) > embargo_days:
        train_positions = train_positions[:-embargo_days]
    elif embargo_days > 0 and len(train_positions) > 0:
        train_positions = np.array([], dtype="int64")

    train_idx = order[train_positions]

    # test 区间
    test_mask_sorted = (d_sorted >= ss) & (d_sorted <= se)
    test_positions = np.nonzero(test_mask_sorted)[0]
    test_idx = order[test_positions]

    train_idx = np.sort(train_idx).astype("int64")
    test_idx = np.sort(test_idx).astype("int64")

    # 硬校验：零重叠
    if train_idx.size and test_idx.size:
        overlap = np.intersect1d(train_idx, test_idx)
        if overlap.size:
            raise ConfigError(
                f"train/test 索引重叠（不应发生）: {overlap[:10]}",
                context={"n_overlap": int(overlap.size)},
            )

    logger.debug(
        "walk_forward_split: train=%d, test=%d, embargo=%d",
        train_idx.size, test_idx.size, embargo_days,
    )
    return train_idx, test_idx


def merge_masks(*masks: "np.ndarray") -> "np.ndarray":
    """按位或合并多个布尔掩码（长度需一致）。

    Args:
        *masks: 布尔数组。

    Returns:
        合并后的布尔数组。
    """
    if not masks:
        return np.array([], dtype=bool)
    out = np.zeros_like(np.asarray(masks[0]), dtype=bool)
    for m in masks:
        out |= np.asarray(m, dtype=bool)
    return out
