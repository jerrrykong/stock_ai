"""A 股交易环境（Gymnasium.Env）。

内建规则：T+1、涨跌停不可成交、停牌、佣金双边 0.025%、印花税卖出 0.05%、滑点 0.1%。
行为由 :mod:`stock_ai.env.rules` 的纯函数驱动，便于单测。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env

from stock_ai.config.schema import EnvConfig
from stock_ai.env.account import EnvState
from stock_ai.env.rewards import RewardCalculator
from stock_ai.env.rules import (
    apply_slippage,
    can_buy,
    can_sell,
    is_halted,
    is_limit_down,
    is_limit_up,
    max_affordable_lots,
    trade_cost,
)
from stock_ai.env.spaces import build_action_space, build_observation_space
from stock_ai.samples.schema import SampleSplit
from stock_ai.utils.logging import LoggerFactory

__all__ = ["TradingEnv"]

logger = LoggerFactory.get(__name__)


class TradingEnv(Env):
    """单标的多步交易环境。

    State: 最近 ``window`` 步的归一化特征张量 + 账户状态标量（仓位/现金比）。
    Action: 离散 0=持有 / 1=买入 / 2=卖出（或连续目标仓位）。
    Reward: :class:`~stock_ai.env.rewards.RewardCalculator` 的多目标结构化奖励。

    Args:
        sample: 单切分样本（obs/dates/prices）。
        env_cfg: 环境配置（费率、滑点、奖励系数等）。
        reward_cfg: 奖励计算器（若为 None 则用 ``env_cfg`` 新建）。
        seed: 随机种子。
        action_mode: ``"discrete"`` 或 ``"continuous"``。
        limit: 涨跌停幅度（主板 0.10）。
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        sample: SampleSplit,
        env_cfg: EnvConfig,
        reward_cfg: RewardCalculator | None = None,
        seed: int = 42,
        action_mode: str = "discrete",
        limit: float = 0.10,
    ) -> None:
        """初始化环境。"""
        super().__init__()
        if sample is None or sample.n_steps == 0:
            raise ValueError("TradingEnv 需要非空 SampleSplit")

        self.sample: SampleSplit = sample
        self.cfg: EnvConfig = env_cfg
        self.reward_calc: RewardCalculator = reward_cfg or RewardCalculator(env_cfg)
        self.action_mode: str = str(action_mode).lower()
        self.limit: float = float(limit)

        self.n_steps: int = int(sample.n_steps)
        self.window: int = int(sample.obs.shape[1]) if sample.obs.ndim == 3 else 1
        self.n_features: int = int(sample.obs.shape[2]) if sample.obs.ndim == 3 else 0

        # 市场原始价量（用于涨跌停/停牌/结算）
        self.obs: "np.ndarray" = np.asarray(sample.obs, dtype="float32")
        self.dates: "np.ndarray" = np.asarray(sample.dates)
        self.symbols: "np.ndarray" = np.asarray(sample.symbols)
        self.prices: "np.ndarray" = np.asarray(sample.prices, dtype="float64")
        # 由 close 序列近似 OHLC（样本契约只保证 prices）
        # 注意：多标的下 prices 是按日期拼接的，prev_close 必须【在标的内部】取
        # 上一日收盘，跨标的处用当日本身（下一条 symbol_boundary 逻辑会重置），
        # 否则会把 A 股票的收盘价当成 B 股票的昨收，导致涨跌停判定完全失真。
        self.prev_closes: "np.ndarray" = self._build_prev_closes()
        # 标的分段边界标记：True 表示该 step 是新标的的第一天（独立 episode 起点）
        self.symbol_boundary: "np.ndarray" = self._build_symbol_boundary()
        # 涨跌停幅度由相邻收盘推得的隐含 high/low 近似：用 |close 变动| 判定
        self.highs: "np.ndarray" = np.maximum(self.prices, self.prev_closes)
        self.lows: "np.ndarray" = np.minimum(self.prices, self.prev_closes)
        # 成交量近似：用价格波动强度作为活跃度代理（停牌判定）
        self.volumes: "np.ndarray" = np.abs(np.diff(self.prices, prepend=self.prices[0])) + 1e-9

        # 空间
        self.observation_space = build_observation_space(self.n_features, self.window)
        self.action_space = build_action_space(self.action_mode)

        # 运行时状态
        self.state: EnvState = EnvState(cash=float(env_cfg.initial_cash))
        self.idx: int = 0
        self._trades: list[dict[str, Any]] = []
        self._equity: list[float] = []
        self._np_random: np.random.Generator = np.random.default_rng(int(seed))

        # SB3 check_env 会校验 action_space.contains；continuous 模式下 action 需被裁到 [0,1]
        self._render_data: list[float] = []

    # ------------------------------------------------------------------ #
    def _build_prev_closes(self) -> "np.ndarray":
        """构造逐日的 ``prev_close`` 数组（仅在**同一标的内部**取上一日收盘）。

        多标的样本按日期拼接，若直接用 ``prices[:-1]`` 前移，会把前一标的的
        收盘价误当作当前标的的昨收，导致涨跌停判定失真。因此在标的分界处
        令 ``prev_close == prices[idx]``（等价于"当日无涨跌幅约束"），保证
        新标的的第一天不会被错误地判为涨跌停。
        """
        n = len(self.prices)
        prev = np.empty(n, dtype="float64")
        if n == 0:
            return prev
        prev[0] = float(self.prices[0])
        for i in range(1, n):
            if self._same_symbol(i, i - 1):
                prev[i] = float(self.prices[i - 1])
            else:
                prev[i] = float(self.prices[i])
        return prev

    def _build_symbol_boundary(self) -> "np.ndarray":
        """标记每个 step 是否为**新标的的第一天**（独立 episode 边界）。"""
        n = len(self.prices)
        boundary = np.zeros(n, dtype=bool)
        for i in range(1, n):
            boundary[i] = not self._same_symbol(i, i - 1)
        return boundary

    def _same_symbol(self, i: int, j: int) -> bool:
        """判断第 ``i`` 与第 ``j`` 步是否属于同一标的（越界或空 symbols 视为同标的）。"""
        if self.symbols.size == 0:
            return True
        if i < 0 or j < 0 or i >= self.symbols.size or j >= self.symbols.size:
            return True
        return str(self.symbols[i]) == str(self.symbols[j])

    # ------------------------------------------------------------------ #
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple["np.ndarray", dict]:
        """重置环境到起点。

        Args:
            seed: 覆盖随机种子。
            options: 预留（支持 ``start_idx`` 指定起点）。

        Returns:
            ``(obs, info)``。
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(int(seed))

        start_idx = 0
        if options and "start_idx" in options:
            start_idx = int(options["start_idx"])
        start_idx = max(0, min(start_idx, max(0, self.n_steps - 1)))

        self.idx = start_idx
        self.state = EnvState(
            cash=float(self.cfg.initial_cash),
            shares=0,
            cost_basis=0.0,
            prev_close=float(self.prev_closes[start_idx]),
            last_buy_idx=-1,
        )
        self.state.mark_to_market(float(self.prices[start_idx]))
        self._trades = []
        self._equity = [self.state.portfolio_value]
        self.reward_calc.reset(self.state.portfolio_value)
        self._render_data = [self.state.portfolio_value]

        return self._obs(), self._info(turnover=0.0, trade_ok=False, blocked_reason="reset")

    # ------------------------------------------------------------------ #
    def step(
        self,
        action: int | "np.ndarray",
    ) -> tuple["np.ndarray", float, bool, bool, dict]:
        """执行一步。

        Args:
            action: 离散动作 (0/1/2) 或连续目标仓位数组。

        Returns:
            ``(obs, reward, terminated, truncated, info)``。
        """
        prev_value = float(self.state.portfolio_value)
        price = float(self.prices[self.idx])
        prev_close = float(self.prev_closes[self.idx])

        # ---- 标的分界处理（U6：v1 单标的独立训练）----
        # 多标的样本按日期拼接，进入新标的第一天时必须：
        #   1) 强平上一标的遗留持仓（按上一日收盘价结算，含卖出成本），
        #      避免把 A 股票的持仓带到 B 股票上造成跨标的套利失真；
        #   2) 重置 T+1 计数与成本基准，使新标的是独立 episode 起点。
        boundary = bool(self.symbol_boundary[self.idx]) if self.idx < self.symbol_boundary.size else False
        if boundary and self.state.shares > 0:
            self._force_liquidate(float(self.prices[self.idx - 1]))
            prev_value = float(self.state.portfolio_value)

        halted = is_halted(
            float(self.volumes[self.idx]),
            float(self.volumes[self.idx - 1]) if self.idx > 0 and not boundary else 0.0,
        )
        if boundary:
            # 新标的的第一天：无昨收约束，也不存在 T+1 冻结
            held_days = 0
        else:
            held_days = self.idx - self.state.last_buy_idx if self.state.last_buy_idx >= 0 else 0

        trade_ok = False
        blocked_reason = ""
        turnover = 0.0

        act = self._normalize_action(action)

        if act == 1:  # 买入
            if can_buy(self.state, price, prev_close, halted):
                exec_price = apply_slippage(price, "buy", self.cfg.slippage_rate)
                lot = int(self.cfg.lot_size)
                # 预留手续费
                budget = self.state.cash * (1.0 - self.cfg.commission_rate)
                shares = max_affordable_lots(budget, exec_price, lot)
                if shares > 0:
                    notional = shares * exec_price
                    fee = trade_cost(
                        notional, "buy",
                        self.cfg.commission_rate, self.cfg.stamp_tax_rate,
                        self.cfg.min_commission,
                    )
                    total = notional + fee
                    if total <= self.state.cash + 1e-6:
                        # 更新持仓均价（含费用摊薄）
                        new_shares = self.state.shares + shares
                        self.state.cost_basis = (
                            (self.state.cost_basis * self.state.shares + total) / new_shares
                        )
                        self.state.shares = new_shares
                        self.state.cash -= total
                        self.state.last_buy_idx = self.idx
                        trade_ok = True
                        turnover = notional / max(prev_value, 1e-9)
                        self._trades.append(
                            {
                                "date": str(self.dates[self.idx]),
                                "side": "buy",
                                "price": exec_price,
                                "shares": shares,
                                "notional": notional,
                                "fee": fee,
                            }
                        )
                else:
                    blocked_reason = "现金不足以买入 1 手"
            else:
                blocked_reason = self._buy_block_reason(price, prev_close, halted)

        elif act == 2:  # 卖出
            if can_sell(self.state, price, prev_close, halted, held_days):
                exec_price = apply_slippage(price, "sell", self.cfg.slippage_rate)
                shares = int(self.state.shares)
                notional = shares * exec_price
                fee = trade_cost(
                    notional, "sell",
                    self.cfg.commission_rate, self.cfg.stamp_tax_rate,
                    self.cfg.min_commission,
                )
                proceeds = notional - fee
                self.state.cash += proceeds
                # 已实现盈亏
                cost = self.state.cost_basis * shares
                realized = proceeds - cost
                self._trades.append(
                    {
                        "date": str(self.dates[self.idx]),
                        "side": "sell",
                        "price": exec_price,
                        "shares": shares,
                        "notional": notional,
                        "fee": fee,
                        "realized_pnl": realized,
                    }
                )
                self.state.shares = 0
                self.state.cost_basis = 0.0
                self.state.last_buy_idx = -1
                trade_ok = True
                turnover = notional / max(prev_value, 1e-9)
            else:
                blocked_reason = self._sell_block_reason(price, prev_close, halted, held_days)

        # ---- 结算与推进 ----
        self.state.prev_close = price
        value = self.state.mark_to_market(price)
        reward = self._compute_reward(prev_value, value, turnover)
        self._equity.append(value)
        self._render_data.append(value)

        self.idx += 1
        terminated = self.idx >= self.n_steps - 1
        truncated = False
        if terminated:
            self.idx = self.n_steps - 1

        info = self._info(turnover=turnover, trade_ok=trade_ok,
                          blocked_reason=blocked_reason, action=int(act))
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ #
    def render(self, mode: str = "human") -> "np.ndarray | None":
        """渲染（默认返回权益曲线数组；``human`` 打印一行摘要）。"""
        curve = np.asarray(self._render_data, dtype="float64")
        if mode == "rgb_array":
            return curve.reshape(-1, 1)
        if mode == "human":
            print(
                f"[env] step={self.idx} value={self.state.portfolio_value:.2f} "
                f"cash={self.state.cash:.2f} shares={self.state.shares}"
            )
        return curve

    def close(self) -> None:
        """释放资源（无外部资源，空实现以满足接口）。"""
        return None

    # ------------------------------------------------------------------ #
    def _force_liquidate(self, price: float) -> None:
        """在标的边界强制平仓（按给定价格，含卖出成本），并重置持仓/T+1 状态。

        用于多标的样本：避免把前一标的的持仓带入下一标的造成跨标的收益失真。

        Args:
            price: 结算价（通常取上一标的的最后一日收盘价）。
        """
        shares = int(self.state.shares)
        if shares <= 0:
            self.state.shares = 0
            self.state.cost_basis = 0.0
            self.state.last_buy_idx = -1
            return
        exec_price = apply_slippage(float(price), "sell", self.cfg.slippage_rate)
        notional = shares * exec_price
        fee = trade_cost(
            notional, "sell",
            self.cfg.commission_rate, self.cfg.stamp_tax_rate, self.cfg.min_commission,
        )
        proceeds = notional - fee
        realized = proceeds - self.state.cost_basis * shares
        self.state.cash += proceeds
        self._trades.append(
            {
                "date": str(self.dates[max(0, self.idx - 1)]),
                "side": "sell",
                "price": exec_price,
                "shares": shares,
                "notional": notional,
                "fee": fee,
                "realized_pnl": realized,
                "reason": "symbol_boundary",
            }
        )
        self.state.shares = 0
        self.state.cost_basis = 0.0
        self.state.last_buy_idx = -1

    # ------------------------------------------------------------------ #
    def _compute_reward(self, prev_value: float, value: float, turnover: float) -> float:
        """委托给 :class:`RewardCalculator`。"""
        return self.reward_calc.compute(
            prev_value, value, turnover, self.state.position_ratio
        )

    def _obs(self) -> "np.ndarray":
        """返回当前观测（特征窗口）。"""
        idx = max(0, min(self.idx, self.n_steps - 1))
        return np.asarray(self.obs[idx], dtype="float32")

    def _info(
        self,
        turnover: float,
        trade_ok: bool,
        blocked_reason: str = "",
        action: int = 0,
    ) -> dict[str, Any]:
        """构造 info 字典（K10 —— 键名固定）。"""
        return {
            "portfolio_value": float(self.state.portfolio_value),
            "cash": float(self.state.cash),
            "shares": int(self.state.shares),
            "turnover": float(turnover),
            "position_ratio": float(self.state.position_ratio),
            "date": str(self.dates[max(0, min(self.idx, self.n_steps - 1))]),
            "trade_ok": bool(trade_ok),
            "blocked_reason": str(blocked_reason),
            "action": int(action),
        }

    # ------------------------------------------------------------------ #
    def _normalize_action(self, action: Any) -> int:
        """把外部动作规范为 0/1/2。

        连续模式下：``pos`` 为目标仓位，与当前仓位比较得方向。
        """
        if self.action_mode == "continuous":
            arr = np.asarray(action, dtype="float64").reshape(-1)
            target = float(np.clip(arr[0] if arr.size else 0.0, 0.0, 1.0))
            current = self.state.position_ratio
            if target > current + 0.05:
                return 1
            if target < current - 0.05:
                return 2
            return 0
        try:
            a = int(np.asarray(action).reshape(-1)[0])
        except Exception:  # pragma: no cover
            a = 0
        return a if a in (0, 1, 2) else 0

    @staticmethod
    def _buy_block_reason(price: float, prev_close: float, halted: bool) -> str:
        """给出买入被拒原因。"""
        if halted:
            return "停牌"
        if is_limit_up(prev_close, price):
            return "涨停不可买入"
        return "现金不足"

    @staticmethod
    def _sell_block_reason(
        price: float, prev_close: float, halted: bool, held_days: int
    ) -> str:
        """给出卖出被拒原因。"""
        if halted:
            return "停牌"
        if held_days < 1:
            return "T+1 冻结"
        if is_limit_down(prev_close, price):
            return "跌停不可卖出"
        return "无持仓"

    # ------------------------------------------------------------------ #
    @property
    def trades(self) -> list[dict[str, Any]]:
        """已触发成交记录。"""
        return self._trades

    @property
    def equity(self) -> "np.ndarray":
        """权益曲线。"""
        return np.asarray(self._equity, dtype="float64")
