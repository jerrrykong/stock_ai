"""账户状态与账面计算。"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["EnvState"]


@dataclass
class EnvState:
    """交易账户的瞬时状态。

    Attributes:
        cash: 可用现金。
        shares: 持仓股数。
        cost_basis: 持仓均价（用于浮动盈亏）。
        prev_close: 上一交易日收盘价（用于涨跌停判断）。
        last_buy_idx: 最近一次买入的步索引（-1 表示无），T+1 判定用。
        portfolio_value: 组合总市值（mark-to-market 后更新）。
        position_ratio: 当前仓位比例（持股市值 / 总市值）。
        cash_ratio: 现金比例。
        unrealized_pnl: 浮动盈亏。
    """

    cash: float = 0.0
    shares: int = 0
    cost_basis: float = 0.0
    prev_close: float = 0.0
    last_buy_idx: int = -1
    portfolio_value: float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def position_ratio(self) -> float:
        """持股市值 / 总市值（总市值为 0 时返回 0）。"""
        if self.portfolio_value <= 0:
            return 0.0
        return float(self.shares * self.cost_basis) / self.portfolio_value

    @property
    def cash_ratio(self) -> float:
        """现金 / 总市值。"""
        if self.portfolio_value <= 0:
            return 0.0
        return float(self.cash) / self.portfolio_value

    @property
    def unrealized_pnl(self) -> float:
        """浮动盈亏 = (现价 - 持仓均价) × 股数（需外部先更新 ``cost_basis``）。"""
        if self.shares == 0:
            return 0.0
        return float(self.shares) * (float(self.prev_close) - float(self.cost_basis))

    @property
    def held(self) -> bool:
        """是否持有股票。"""
        return self.shares > 0

    def mark_to_market(self, price: float) -> float:
        """按给定价格更新组合市值并返回。

        Args:
            price: 用于估值的现价。

        Returns:
            更新后的 ``portfolio_value``。
        """
        self.portfolio_value = float(self.cash) + float(self.shares) * float(price)
        return self.portfolio_value

    def snapshot(self) -> dict:
        """返回状态快照（供 info / 日志使用）。"""
        return {
            "cash": float(self.cash),
            "shares": int(self.shares),
            "cost_basis": float(self.cost_basis),
            "prev_close": float(self.prev_close),
            "portfolio_value": float(self.portfolio_value),
            "position_ratio": self.position_ratio,
            "cash_ratio": self.cash_ratio,
        }
