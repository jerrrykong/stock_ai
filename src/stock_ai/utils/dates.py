"""日期区间工具：轻量规范化 + ``start <= end`` 前置校验（P2-6）。

设计目标：在**进入任何网络/计算逻辑之前**，对用户给出的日期区间做一次
廉价且明确的校验，避免非法区间被后续逻辑（如"所有数据源均失败"）以
不直观的错误间接暴露。
"""

from __future__ import annotations

import re

from stock_ai.utils.errors import ConfigError

__all__ = ["to_day_key", "validate_range"]

#: 非数字字符（用于把 ``2024-01-01`` / ``2024/01/01`` 归一为 ``20240101``）
_NON_DIGIT = re.compile(r"[^0-9]")


def to_day_key(value: object) -> str:
    """把日期归一为 8 位数字键（``YYYYMMDD``）。

    支持 ``YYYYMMDD`` / ``YYYY-MM-DD`` / ``YYYY/MM/DD`` 等常见写法；
    位数不足或无法解析时返回空串（交由调用方决定是否报错，避免误伤）。

    Args:
        value: 任意可转字符串的日期表示。

    Returns:
        8 位数字键，或空串。
    """
    digits = _NON_DIGIT.sub("", str(value or ""))
    return digits[:8] if len(digits) >= 8 else ""


def validate_range(
    start: object,
    end: object,
    *,
    label: str = "日期区间",
    hint: str = "",
) -> None:
    """校验 ``start <= end``；起始晚于结束时抛出明确的 :class:`ConfigError`。

    Args:
        start: 起始日期。
        end: 结束日期。
        label: 用于错误信息的区间名称（如 ``"数据拉取区间"``）。
        hint: 附加提示（如 ``"请检查 --start/--end"``）。

    Raises:
        ConfigError: 当两侧均可解析且 ``start > end``。
    """
    s = to_day_key(start)
    e = to_day_key(end)
    if s and e and s > e:
        extra = f"（{hint}）" if hint else ""
        raise ConfigError(
            f"{label}非法：起始日期 {start} 晚于结束日期 {end}{extra}",
            context={"start": str(start), "end": str(end)},
        )
