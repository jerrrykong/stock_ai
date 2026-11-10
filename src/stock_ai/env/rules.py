"""A 股交易规则纯函数（K14 —— 必须无副作用，便于单测）。

覆盖：T+1、涨跌停、停牌、滑点、佣金（双边 0.025%，最低 5 元）、印花税（卖出 0.05%）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from stock_ai.env.account import EnvState

__all__ = [
    "is_limit_up",
    "is_limit_down",
    "is_halted",
    "can_buy",
    "can_sell",
    "apply_slippage",
    "trade_cost",
    "max_affordable_lots",
    "round_lots",
    "limit_price",
]

#: 默认涨跌停幅度（主板 10%）
DEFAULT_LIMIT: float = 0.10

#: 浮点比较容差
_EPS: float = 1e-6


def limit_price(prev_close: float, side: str = "up", limit: float = DEFAULT_LIMIT) -> float:
    """计算涨停/跌停价（四舍五入到 0.01 元）。

    Args:
        prev_close: 上一交易日收盘价。
        side: ``"up"`` 涨停，``"down"`` 跌停。
        limit: 涨跌停幅度（默认 0.10）。

    Returns:
        涨/跌停价格。
    """
    factor = 1.0 + limit if side == "up" else 1.0 - limit
    return round(float(prev_close) * factor + 1e-9, 2)


def is_limit_up(prev_close: float, high: float, limit: float = DEFAULT_LIMIT) -> bool:
    """判断当日是否涨停（最高价触及涨停价）。

    Args:
        prev_close: 上一交易日收盘价。
        high: 当日最高价。
        limit: 涨跌停幅度。

    Returns:
        涨停返回 ``True``。
    """
    if prev_close is None or prev_close <= 0 or high is None:
        return False
    return float(high) >= limit_price(prev_close, "up", limit) - _EPS


def is_limit_down(prev_close: float, low: float, limit: float = DEFAULT_LIMIT) -> bool:
    """判断当日是否跌停（最低价触及跌停价）。

    Args:
        prev_close: 上一交易日收盘价。
        low: 当日最低价。
        limit: 涨跌停幅度。

    Returns:
        跌停返回 ``True``。
    """
    if prev_close is None or prev_close <= 0 or low is None:
        return False
    return float(low) <= limit_price(prev_close, "down", limit) + _EPS


def is_halted(volume: float, prev_volume: float, ratio: float = 0.05) -> bool:
    """判断当日是否停牌（成交量极低或为 0）。

    Args:
        volume: 当日成交量。
        prev_volume: 上一交易日成交量。
        ratio: 视为停牌的成交量比例阈值。

    Returns:
        停牌返回 ``True``。
    """
    try:
        v = float(volume)
    except (TypeError, ValueError):
        return True
    if v <= 0:
        return True
    try:
        pv = float(prev_volume)
    except (TypeError, ValueError):
        return False
    if pv <= 0:
        return False
    return v < pv * float(ratio)


def can_buy(
    state: "EnvState",
    price: float,
    prev_close: float,
    halted: bool,
    t1_blocked: bool = False,
    limit: float = DEFAULT_LIMIT,
) -> bool:
    """判断当前是否可买入。

    条件：未停牌、非涨停、现金充足（至少能买 1 手）。

    Args:
        state: 账户状态。
        price: 当前价（不含滑点）。
        prev_close: 上一交易日收盘价。
        halted: 是否停牌。
        t1_blocked: 是否被 T+1 冻结（买入不受 T+1 限制，仅保留参数语义）。
        limit: 涨跌停幅度。

    Returns:
        可买返回 ``True``。
    """
    if halted:
        return False
    if state.cash <= 0:
        return False
    # 涨停不可买入
    if is_limit_up(prev_close, price, limit):
        return False
    return True


def can_sell(
    state: "EnvState",
    price: float,
    prev_close: float,
    halted: bool,
    held_days: int = 1,
    limit: float = DEFAULT_LIMIT,
) -> bool:
    """判断当前是否可卖出。

    条件：有持仓、未停牌、非跌停、非 T+1 冻结（``held_days >= 1``）。

    Args:
        state: 账户状态。
        price: 当前价（不含滑点）。
        prev_close: 上一交易日收盘价。
        halted: 是否停牌。
        held_days: 买入后经过的交易日数（0 表示当日买入，T+1 拒绝）。
        limit: 涨跌停幅度。

    Returns:
        可卖返回 ``True``。
    """
    if halted:
        return False
    if state.shares <= 0:
        return False
    if held_days < 1:  # T+1：当日买入不可卖
        return False
    if is_limit_down(prev_close, price, limit):
        return False
    return True


def apply_slippage(price: float, side: str, slippage: float) -> float:
    """按方向施加滑点。

    买入价上浮、卖出价下浮。

    Args:
        price: 原始成交价。
        side: ``"buy"`` 或 ``"sell"``。
        slippage: 滑点比例（如 0.001）。

    Returns:
        滑点后的成交价。
    """
    p = float(price)
    s = float(slippage)
    if side == "buy":
        return p * (1.0 + s)
    return p * (1.0 - s)


def trade_cost(
    notional: float,
    side: str,
    commission_rate: float,
    stamp_tax_rate: float,
    min_commission: float = 5.0,
) -> float:
    """计算交易成本。

    规则：
        - ``buy``:  ``max(notional * commission_rate, min_commission)``
        - ``sell``: ``max(notional * commission_rate, min_commission) + notional * stamp_tax_rate``

    Args:
        notional: 成交金额（价格 × 股数）。
        side: ``"buy"`` 或 ``"sell"``。
        commission_rate: 佣金率（双边）。
        stamp_tax_rate: 印花税率（仅卖出）。
        min_commission: 最低佣金。

    Returns:
        成本金额（元）。``notional <= 0`` 时返回 0。
    """
    n = float(notional)
    if n <= 0:
        return 0.0
    commission = max(n * float(commission_rate), float(min_commission))
    if side == "sell":
        return commission + n * float(stamp_tax_rate)
    return commission


def round_lots(shares: float, lot_size: int = 100) -> int:
    """向下取整到整手。

    Args:
        shares: 目标股数。
        lot_size: 每手股数（A 股 100）。

    Returns:
        整手股数。
    """
    lot = max(1, int(lot_size))
    return int(max(0.0, float(shares)) // lot) * lot


def max_affordable_lots(cash: float, price: float, lot_size: int = 100) -> int:
    """在给定现金下最多可买多少股（含滑点前近似，整手）。

    Args:
        cash: 可用现金。
        price: 成交价。
        lot_size: 每手股数。

    Returns:
        可买股数（整手）。
    """
    if cash <= 0 or price <= 0:
        return 0
    return round_lots(float(cash) / float(price), lot_size)
