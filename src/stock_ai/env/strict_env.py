"""严格撮合环境（unified 专用）：T 日收盘信号 -> T+1 日**真实开盘价**执行。

与旧 :class:`stock_ai.env.trading_env.TradingEnv` 的差异（**不修改旧环境**）
------------------------------------------------------------------------
1. **执行时点**：T 日以观测 ``obs[T]``（含 T 日收盘）产生信号，成交发生在
   **T+1 日的开盘价**；不是同日收盘模拟成交。
2. **价格来源**：特征用后复权（hfq）价，成交/持仓/费用全部用**未复权真实价**
   （由 :class:`ExecutionBars` 携带），因此不存在"用复权价成交"的失真。
3. **涨跌停**：按 :mod:`stock_ai.env.limits` 的板块 + 生效日期规则计算
   （主板 10% / 创业板·科创板 20% / 北交所 30% / 主板 ST 5% 近似），且
   **一字涨跌停（开盘即封板）直接拒单**。
4. **停牌**：真实 ``volume <= 0`` 视为停牌，不可买卖。
5. **T+1**：T+1 买入后当日不可卖出（沿用旧语义）。

无未来函数保证
--------------
- 观测窗口只含 ``<= T`` 的特征；
- 第 ``T`` 步的成交只读取 ``bars[T+1]`` 的开盘价与 ``bars[T]`` 的收盘价，
  不读取 ``bars[T+1].close`` 来决策；
- 估值用 ``bars[T+1].close``（成交之后才能知道），只进入 reward / equity，
  不回流到动作选择。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import Env

from stock_ai.config.schema import EnvConfig
from stock_ai.env.account import EnvState
from stock_ai.env.limits import is_blocked_at_open, limit_prices, price_limit
from stock_ai.env.rewards import RewardCalculator
from stock_ai.env.rules import trade_cost
from stock_ai.env.spaces import build_action_space, build_observation_space
from stock_ai.utils.logging import LoggerFactory

__all__ = ["EXECUTION_VERSION", "ExecutionBars", "StrictExecutionEnv", "friction_metrics"]

#: 严格撮合语义版本（写入 unified manifest，用于区分旧同日收盘语义）。
EXECUTION_VERSION: str = "strict-t1-open-v1"

logger = LoggerFactory.get(__name__)


@dataclass
class ExecutionBars:
    """一只股票一段 episode 的**真实未复权**日线（撮合唯一价格来源）。

    Attributes:
        dates: ``(T,)`` 日期字符串。
        open/high/low/close/volume: ``(T,)`` 未复权真实行情。
        symbol: 6 位代码（用于板块涨跌停规则）。
    """

    dates: "np.ndarray"
    open: "np.ndarray"
    high: "np.ndarray"
    low: "np.ndarray"
    close: "np.ndarray"
    volume: "np.ndarray"
    symbol: str = ""
    # v2 专用已核验事件账本；v1 环境明确忽略，保持原始撮合语义。
    corporate_actions: Any = None

    def __post_init__(self) -> None:
        self.dates = np.asarray(self.dates)
        for name in ("open", "high", "low", "close", "volume"):
            setattr(self, name, np.asarray(getattr(self, name), dtype="float64"))
        self.symbol = str(self.symbol or "")

    @property
    def n_steps(self) -> int:
        """交易日根数。"""
        return int(len(self.close))

    def prices_matrix(self) -> "np.ndarray":
        """返回 ``(T, 5)`` 的 ``[open, high, low, close, volume]``。"""
        return np.column_stack([self.open, self.high, self.low, self.close, self.volume])

    @classmethod
    def from_frame(cls, df: "Any", symbol: str = "") -> "ExecutionBars":
        """由日线 DataFrame 构造（要求列为真实未复权价）。"""
        return cls(
            dates=df["date"].astype(str).to_numpy(),
            open=df["open"].to_numpy(dtype="float64"),
            high=df["high"].to_numpy(dtype="float64"),
            low=df["low"].to_numpy(dtype="float64"),
            close=df["close"].to_numpy(dtype="float64"),
            volume=df["volume"].to_numpy(dtype="float64"),
            symbol=symbol,
        )


def friction_metrics(
    trades: "list[dict]",
    initial_cash: float,
    equity_curve: "np.ndarray",
    benchmark_curve: "np.ndarray | None" = None,
) -> dict:
    """计算交易磨损指标（bps 口径）。

    Args:
        trades: 成交记录（需含 ``notional`` / ``fee`` / ``side``）。
        initial_cash: 初始资金。
        equity_curve: 策略权益曲线（长度 >= 2）。
        benchmark_curve: 买入持有基准曲线（可选）。

    Returns:
        ``{"trades", "total_turnover", "turnover_ratio", "cost_drag_bps",
        "avg_wear_bps", "total_fee", "strategy_return", "benchmark_return",
        "excess_return", "avg_equity"}``。
    """
    equity = np.asarray(equity_curve, dtype="float64")
    notionals = np.asarray([float(t.get("notional", 0.0)) for t in trades], dtype="float64")
    fees = float(sum(float(t.get("fee", 0.0)) for t in trades))
    total_notional = float(notionals.sum()) if notionals.size else 0.0
    total_turnover = float(notionals.size)
    avg_equity = float(equity.mean()) if equity.size else float(initial_cash)
    base = float(initial_cash) or 1.0
    strategy_return = float(equity[-1] / base - 1.0) if equity.size else 0.0
    benchmark_return = None
    if benchmark_curve is not None and len(benchmark_curve) >= 1:
        benchmark_return = float(np.asarray(benchmark_curve, dtype="float64")[-1] / base - 1.0)
    return {
        "trades": int(total_turnover),
        "total_notional": total_notional,
        "total_turnover_ratio": total_notional / base,
        "total_fee": fees,
        # 成本拖累（bps）：总费用相对平均权益的百分比
        "cost_drag_bps": fees / avg_equity * 1e4 if avg_equity else 0.0,
        # 平均单笔磨损（bps）：单笔费用相对该笔成交金额
        "avg_wear_bps": float(
            np.mean([float(t.get("fee", 0.0)) / float(t.get("notional", 1.0)) * 1e4
                     for t in trades if float(t.get("notional", 0.0)) > 0])
        ) if trades else 0.0,
        "avg_equity": avg_equity,
        "strategy_return": strategy_return,
        "benchmark_return": benchmark_return,
        "excess_return": (strategy_return - benchmark_return
                          if benchmark_return is not None else None),
    }


class StrictExecutionEnv(Env):
    """T 日信号 / T+1 日真实开盘成交的严格撮合环境。

    Observation:
        最近 ``window`` 步归一化特征（形状 ``(window, n_features)``，来自 hfq 特征）。
    Action:
        ``Discrete(3)``：0=持有 / 1=买入 / 2=卖出。
    """

    metadata: dict[str, Any] = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        obs: "np.ndarray",
        bars: ExecutionBars,
        env_cfg: EnvConfig,
        reward_cfg: RewardCalculator | None = None,
        seed: int = 42,
        st: bool = False,
    ) -> None:
        """初始化严格撮合环境。

        Args:
            obs: ``(T, window, n_features)`` 特征张量（后复权派生）。
            bars: 同长度、同时序的**真实未复权**日线。
            env_cfg: 环境配置（费用、滑点、初始资金）。
            reward_cfg: 奖励计算器（None 则按 ``env_cfg`` 新建）。
            seed: 随机种子（供 gym ``np_random``）。
            st: 是否按 ST 近似（5%）判定涨跌停；默认 False。
        """
        super().__init__()
        if bars is None or bars.n_steps < 2:
            raise ValueError("StrictExecutionEnv 需要至少 2 根真实 K 线")
        self.obs = np.asarray(obs, dtype="float32")
        if self.obs.ndim != 3 or self.obs.shape[0] != bars.n_steps:
            raise ValueError(
                f"特征与真实 K 线长度不一致: {self.obs.shape[0]} vs {bars.n_steps}"
            )
        self.bars = bars
        self.cfg = env_cfg
        self.st = bool(st)
        self.reward_calc = reward_cfg or RewardCalculator(env_cfg)
        self.n_steps: int = int(bars.n_steps)
        self.window: int = int(self.obs.shape[1])
        self.n_features: int = int(self.obs.shape[2])
        self.observation_space = build_observation_space(self.n_features, self.window)
        self.action_space = build_action_space("discrete")

        self.state = EnvState(cash=float(env_cfg.initial_cash))
        self.idx: int = 0
        self._trades: list[dict[str, Any]] = []
        self._equity: list[float] = []
        self._np_random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, options: dict | None = None):
        """重置到第一个交易日（尚无持仓）。"""
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))
        self.idx = 0
        self.state = EnvState(cash=float(self.cfg.initial_cash))
        self._trades = []
        self._equity = [float(self.cfg.initial_cash)]
        self.reward_calc.reset(self.cfg.initial_cash)
        return self._observation(), self._info(turnover=0.0, trade_ok=False,
                                              blocked_reason="reset", action=0)

    # ------------------------------------------------------------------ #
    def step(self, action: Any):
        """执行一步：``obs[idx]`` 产生信号，在 ``bars[idx+1]`` 开盘价成交。

        Args:
            action: 离散动作 0/1/2。

        Returns:
            ``(obs, reward, terminated, truncated, info)``。
        """
        if self.idx >= self.n_steps - 1:
            raise RuntimeError("episode 已结束，请先 reset")
        signal_idx = self.idx                 # 信号日 T
        exec_idx = signal_idx + 1            # 执行日 T+1
        prev_value = self.state.mark_to_market(float(self.bars.close[signal_idx]))
        prev_close = float(self.bars.close[signal_idx])
        open_price = float(self.bars.open[exec_idx])
        volume = float(self.bars.volume[exec_idx])
        trade_date = str(self.bars.dates[exec_idx])
        rate = price_limit(self.bars.symbol, trade_date, self.st)
        up, down = limit_prices(prev_close, self.bars.symbol, trade_date, self.st)

        act = int(np.asarray(action).reshape(-1)[0]) if np.asarray(action).size else 0
        if act not in (0, 1, 2):
            act = 0

        halted = not np.isfinite(volume) or volume <= 0
        # T+1 冻结：买入发生在 exec_idx，卖出最早只能发生在 exec_idx + 1。
        held_days = 0 if self.state.last_buy_idx < 0 else exec_idx - self.state.last_buy_idx
        trade_ok = False
        blocked = ""
        turnover = 0.0

        if act == 1:
            if halted:
                blocked = "停牌"
            elif is_blocked_at_open(open_price, prev_close, self.bars.symbol,
                                    trade_date, "buy", self.st):
                blocked = f"开盘涨停({up:.2f})不可买入"
            else:
                exec_price = open_price * (1.0 + float(self.cfg.slippage_rate))
                lot = int(self.cfg.lot_size)
                budget = self.state.cash * (1.0 - float(self.cfg.commission_rate))
                shares = int(max(0.0, budget // exec_price // lot) * lot)
                if shares <= 0:
                    blocked = "现金不足以买入 1 手"
                else:
                    notional = shares * exec_price
                    fee = trade_cost(notional, "buy", self.cfg.commission_rate,
                                     self.cfg.stamp_tax_rate, self.cfg.min_commission)
                    total = notional + fee
                    if total > self.state.cash + 1e-6:
                        blocked = "现金不足以覆盖成交金额与费用"
                    else:
                        new_shares = self.state.shares + shares
                        self.state.cost_basis = (
                            (self.state.cost_basis * self.state.shares + total) / new_shares
                        )
                        self.state.shares = new_shares
                        self.state.cash -= total
                        self.state.last_buy_idx = exec_idx
                        trade_ok = True
                        turnover = notional / max(prev_value, 1e-9)
                        self._trades.append({
                            "date": trade_date, "side": "buy", "signal_date":
                            str(self.bars.dates[signal_idx]), "open_price": open_price,
                            "price": exec_price, "shares": shares, "notional": notional,
                            "fee": fee, "limit_rate": rate,
                        })
        elif act == 2:
            if self.state.shares <= 0:
                blocked = "无持仓"
            elif halted:
                blocked = "停牌"
            elif held_days < 1:
                blocked = "T+1 冻结（当日买入不可卖出）"
            elif is_blocked_at_open(open_price, prev_close, self.bars.symbol,
                                    trade_date, "sell", self.st):
                blocked = f"开盘跌停({down:.2f})不可卖出"
            else:
                exec_price = open_price * (1.0 - float(self.cfg.slippage_rate))
                shares = int(self.state.shares)
                notional = shares * exec_price
                fee = trade_cost(notional, "sell", self.cfg.commission_rate,
                                 self.cfg.stamp_tax_rate, self.cfg.min_commission)
                proceeds = notional - fee
                realized = proceeds - self.state.cost_basis * shares
                self.state.cash += proceeds
                self.state.shares = 0
                self.state.cost_basis = 0.0
                self.state.last_buy_idx = -1
                trade_ok = True
                turnover = notional / max(prev_value, 1e-9)
                self._trades.append({
                    "date": trade_date, "side": "sell", "signal_date":
                    str(self.bars.dates[signal_idx]), "open_price": open_price,
                    "price": exec_price, "shares": shares, "notional": notional,
                    "fee": fee, "realized_pnl": realized, "limit_rate": rate,
                })

        self.state.prev_close = float(self.bars.close[exec_idx])
        value = self.state.mark_to_market(self.state.prev_close)
        # 基准收益：只用信号日 T 收盘与执行日 T+1 收盘（close-to-close），
        # 绝不读取 t+2 及以后（严禁复用有前视的 buy_and_hold()）。
        bench = self._benchmark_step_return(signal_idx, exec_idx)
        ret_value = (value - prev_value) / prev_value if prev_value > 1e-9 else 0.0
        reward = self.reward_calc.compute(prev_value, value, turnover,
                                          self._position_ratio(value), benchmark_return=bench)
        self._equity.append(float(value))
        self.idx = exec_idx
        terminated = self.idx >= self.n_steps - 1
        info = self._info(turnover=turnover, trade_ok=trade_ok, blocked_reason=blocked,
                          action=act)
        info.update(portfolio_value=float(value), cash=float(self.state.cash),
                    shares=int(self.state.shares),
                    position_ratio=self._position_ratio(value),
                    execution_date=trade_date, open_price=open_price,
                    limit_up=up, limit_down=down,
                    benchmark_return=float(bench),
                    step_excess=float(ret_value - bench))
        return self._observation(), float(reward), bool(terminated), False, info

    # ------------------------------------------------------------------ #
    def buy_and_hold(self) -> "np.ndarray":
        """买入持有基准曲线（首日真实开盘买入、次日开盘买入，全程计入费用滑点）。"""
        n = self.n_steps
        cash, shares = float(self.cfg.initial_cash), 0
        curve = np.empty(n, dtype="float64")
        for i in range(n):
            if shares == 0 and i + 1 < n and float(self.bars.volume[i + 1]) > 0:
                price = float(self.bars.open[i + 1]) * (1.0 + float(self.cfg.slippage_rate))
                lots = int(max(0.0, cash * (1 - float(self.cfg.commission_rate))
                               // price // int(self.cfg.lot_size)) * int(self.cfg.lot_size))
                if lots > 0:
                    notional = lots * price
                    fee = trade_cost(notional, "buy", self.cfg.commission_rate,
                                     self.cfg.stamp_tax_rate, self.cfg.min_commission)
                    if notional + fee <= cash:
                        cash -= notional + fee
                        shares = lots
            curve[i] = cash + shares * float(self.bars.close[i])
        return curve

    def metrics(self) -> dict:
        """输出交易磨损指标（含买入持有基准对比）。"""
        return friction_metrics(self._trades, float(self.cfg.initial_cash),
                                np.asarray(self._equity, dtype="float64"),
                                self.buy_and_hold())

    def render(self, mode: str = "human"):
        """打印一行摘要。"""
        print(f"[strict-env] signal_idx={self.idx - 1} exec_idx={self.idx} "
              f"value={self.state.portfolio_value:.2f} cash={self.state.cash:.2f} "
              f"shares={self.state.shares}")
        return np.asarray(self._equity, dtype="float64")

    def close(self) -> None:
        """无外部资源。"""
        return None

    # ------------------------------------------------------------------ #
    def _benchmark_step_return(self, signal_idx: int, exec_idx: int) -> float:
        """单步买入持有基准收益（close-to-close，因果无前视）。

        ``signal_idx`` 是信号日 T 的 close 索引，``exec_idx`` 是执行日 T+1 的
        close 索引。**只用这两天**，绝不读取 ``t+2`` 及以后（因此不能复用带有
        未来信息的 :meth:`buy_and_hold`）。

        Args:
            signal_idx: 信号日 T 的收盘索引。
            exec_idx: 执行日 T+1 的收盘索引。

        Returns:
            ``close[T+1] / close[T] - 1``；分母 ≤1e-9 或非有限时返回 ``0.0``。
        """
        c0 = float(self.bars.close[signal_idx])
        c1 = float(self.bars.close[exec_idx])
        if c0 > 1e-9 and np.isfinite(c0) and np.isfinite(c1):
            return c1 / c0 - 1.0
        return 0.0

    def _position_ratio(self, value: float) -> float:
        """持仓市值 / 总市值。"""
        if value <= 0:
            return 0.0
        return float(self.state.shares * self.state.prev_close) / float(value)

    def _observation(self) -> "np.ndarray":
        idx = max(0, min(self.idx, self.n_steps - 1))
        return np.asarray(self.obs[idx], dtype="float32")

    def _info(self, turnover: float, trade_ok: bool, blocked_reason: str,
              action: int) -> dict[str, Any]:
        """构造 info 字典（键名与旧环境保持一致，便于回放复用）。"""
        return {
            "portfolio_value": float(self.state.portfolio_value),
            "cash": float(self.state.cash),
            "shares": int(self.state.shares),
            "turnover": float(turnover),
            "position_ratio": self._position_ratio(self.state.portfolio_value),
            "date": str(self.bars.dates[max(0, min(self.idx, self.n_steps - 1))]),
            "trade_ok": bool(trade_ok),
            "blocked_reason": str(blocked_reason),
            "action": int(action),
        }

    @property
    def trades(self) -> list[dict[str, Any]]:
        """成交记录。"""
        return self._trades

    @property
    def equity(self) -> "np.ndarray":
        """权益曲线。"""
        return np.asarray(self._equity, dtype="float64")


class StrictEpisodePoolEnv(gym.Env):
    """在多只股票年度 episode 间采样的严格撮合环境池。

    Args:
        episodes: 元素为 ``(episode_id, obs, bars)`` 的序列。
        cfg: 环境配置。
        seed: 随机种子。
    """

    def __init__(self, episodes, cfg: EnvConfig, seed: int = 42) -> None:
        super().__init__()
        if not episodes:
            raise ValueError("StrictEpisodePoolEnv 需要至少一个 episode")
        self.episodes = list(episodes)
        self.cfg = cfg
        self.inner = StrictExecutionEnv(*self.episodes[0][1:], env_cfg=cfg, seed=seed)
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space
        self.np_random = np.random.default_rng(int(seed))
        self.episode_id = self.episodes[0][0]

    def reset(self, seed=None, options=None):
        """随机抽一个 episode 重置。"""
        super().reset(seed=seed)
        episode = self.episodes[int(self.np_random.integers(len(self.episodes)))]
        self.inner = StrictExecutionEnv(episode[1], episode[2], self.cfg)
        self.episode_id = episode[0]
        obs, info = self.inner.reset(seed=seed)
        return obs, {**info, "episode_id": self.episode_id}

    def step(self, action):
        """委托内部严格撮合环境。"""
        obs, reward, term, trunc, info = self.inner.step(action)
        return obs, reward, term, trunc, {**info, "episode_id": self.episode_id}

    def close(self) -> None:
        """关闭内部环境。"""
        self.inner.close()
