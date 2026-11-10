"""组合模拟器：按动作序列模拟组合演进（复用 env 成本规则）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from stock_ai.config.schema import EnvConfig
from stock_ai.env.rules import (
    apply_slippage,
    is_limit_down,
    is_limit_up,
    max_affordable_lots,
    trade_cost,
)
from stock_ai.utils.logging import LoggerFactory

__all__ = ["PortfolioSimulator"]

logger = LoggerFactory.get(__name__)


class PortfolioSimulator:
    """与 :class:`~stock_ai.env.trading_env.TradingEnv` 同规则的成本模拟器。

    用于对 ``actions`` 数组做向量化回放（如已有动作序列时的离线评估）。

    Args:
        env_cfg: 环境配置（费率/滑点/整手）。
    """

    def __init__(self, env_cfg: EnvConfig) -> None:
        """初始化。"""
        self.env_cfg: EnvConfig = env_cfg

    # ------------------------------------------------------------------ #
    def apply_costs(self, notional: float, side: str) -> float:
        """计算单笔成交成本（委托 :func:`stock_ai.env.rules.trade_cost`）。

        Args:
            notional: 成交金额。
            side: ``buy`` / ``sell``。

        Returns:
            成本金额。
        """
        return trade_cost(
            notional,
            side,
            self.env_cfg.commission_rate,
            self.env_cfg.stamp_tax_rate,
            self.env_cfg.min_commission,
        )

    # ------------------------------------------------------------------ #
    def run(
        self,
        actions: "np.ndarray",
        prices: "np.ndarray",
        dates: "np.ndarray",
    ) -> tuple["np.ndarray", list[dict]]:
        """按动作序列模拟组合。

        Args:
            actions: ``(T,)`` 离散动作（0=持有,1=买入,2=卖出）。
            prices: ``(T,)`` 结算价（close）。
            dates: ``(T,)`` 日期。

        Returns:
            ``(equity (T,), trades)``。
        """
        acts = np.asarray(actions, dtype="int64").reshape(-1)
        px = np.asarray(prices, dtype="float64").reshape(-1)
        ds = np.asarray(dates)
        t = min(len(acts), len(px), len(ds))
        if t == 0:
            return np.array([], dtype="float64"), []

        cfg = self.env_cfg
        cash = float(cfg.initial_cash)
        shares = 0
        cost_basis = 0.0
        last_buy_idx = -1
        equity = np.zeros(t, dtype="float64")
        trades: list[dict] = []

        prev_close = float(px[0])

        for i in range(t):
            price = float(px[i])
            # 样本契约不含成交量，无法精确判定停牌；此处按"未停牌"处理
            # （训练环境 TradingEnv 有成交量代理，可精确判定）
            halted = False

            act = int(acts[i])
            if act == 1 and shares == 0:
                if not is_limit_up(prev_close, price):
                    exec_price = apply_slippage(price, "buy", cfg.slippage_rate)
                    budget = cash * (1.0 - cfg.commission_rate)
                    qty = max_affordable_lots(budget, exec_price, cfg.lot_size)
                    if qty > 0:
                        notional = qty * exec_price
                        fee = self.apply_costs(notional, "buy")
                        total = notional + fee
                        if total <= cash + 1e-6:
                            cash -= total
                            shares = qty
                            cost_basis = total / qty
                            last_buy_idx = i
                            trades.append(
                                {
                                    "date": str(ds[i]), "side": "buy",
                                    "price": exec_price, "shares": qty,
                                    "notional": notional, "fee": fee,
                                }
                            )
            elif act == 2 and shares > 0:
                held_days = i - last_buy_idx
                if held_days >= 1 and not is_limit_down(prev_close, price):
                    exec_price = apply_slippage(price, "sell", cfg.slippage_rate)
                    notional = shares * exec_price
                    fee = self.apply_costs(notional, "sell")
                    proceeds = notional - fee
                    realized = proceeds - cost_basis * shares
                    cash += proceeds
                    trades.append(
                        {
                            "date": str(ds[i]), "side": "sell",
                            "price": exec_price, "shares": shares,
                            "notional": notional, "fee": fee,
                            "realized_pnl": realized,
                        }
                    )
                    shares = 0
                    cost_basis = 0.0
                    last_buy_idx = -1

            equity[i] = cash + shares * price
            prev_close = price

        return equity, trades

    # ------------------------------------------------------------------ #
    def mark_to_market(self, cash: float, shares: int, price: float) -> float:
        """按市价估值。

        Args:
            cash: 现金。
            shares: 持股数。
            price: 现价。

        Returns:
            组合总市值。
        """
        return float(cash) + float(shares) * float(price)
