"""回测器：加载模型，在测试样本上跑完整 episode，对比 buy&hold 基线。"""

from __future__ import annotations

from typing import Any

import numpy as np

from stock_ai.config.schema import EnvConfig
from stock_ai.eval.metrics import compute_all
from stock_ai.eval.schema import BacktestResult
from stock_ai.samples.schema import SampleSplit
from stock_ai.utils.logging import LoggerFactory

__all__ = ["Backtester"]

logger = LoggerFactory.get(__name__)


class Backtester:
    """基于 :class:`~stock_ai.env.trading_env.TradingEnv` 的回测器。

    Args:
        model: :class:`~stock_ai.train.base.Enrollment` 协议实现。
        env_cfg: 环境配置。
        samples: 测试样本切分。
        mode: 动作模式（``discrete`` / ``continuous``）。
    """

    def __init__(
        self,
        model: "Any",
        env_cfg: EnvConfig,
        samples: SampleSplit,
        mode: str = "discrete",
    ) -> None:
        """初始化。"""
        self.model = model
        self.env_cfg = env_cfg
        self.samples = samples
        self.mode = str(mode).lower()

    # ------------------------------------------------------------------ #
    def run(
        self,
        backend: str = "numpy",
        algo: str = "ppo",
        model_path: str = "",
        data_hash: str = "",
        test_range: tuple[str, str] = ("", ""),
    ) -> BacktestResult:
        """跑完整测试 episode 并计算指标。

        Args:
            backend: 模型后端标识。
            algo: 算法名。
            model_path: 模型路径。
            data_hash: 数据摘要。
            test_range: 测试区间。

        Returns:
            :class:`BacktestResult`。
        """
        from stock_ai.env.trading_env import TradingEnv

        env = TradingEnv(self.samples, self.env_cfg, seed=42, action_mode=self.mode)
        obs, info = env.reset(seed=42)
        equity = [float(info["portfolio_value"])]
        dates = [str(self.samples.dates[0])] if self.samples.n_steps else []

        steps = 0
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            equity.append(float(info["portfolio_value"]))
            dates.append(str(info["date"]))
            steps += 1
            done = bool(terminated or truncated)
            if steps > self.samples.n_steps + 5:  # 安全阀
                break

        equity_arr = np.asarray(equity, dtype="float64")
        dates_arr = np.asarray(dates)
        trades = list(env.trades)
        env.close()

        # buy&hold 基线
        baseline = self._buy_and_hold(self.samples, self.env_cfg)

        # 对齐长度（baseline / equity 取 min）
        n = min(len(equity_arr), len(baseline)) if len(baseline) else len(equity_arr)
        equity_arr = equity_arr[:n]
        baseline = baseline[:n] if len(baseline) else baseline
        dates_arr = dates_arr[:n] if len(dates_arr) >= n else dates_arr

        metrics = compute_all(equity_arr, dates_arr, trades, benchmark_equity=baseline)

        result = BacktestResult(
            metrics=metrics,
            equity_curve=equity_arr,
            dates=dates_arr,
            trades=trades,
            baseline_equity=baseline,
            backend=backend,
            algo=algo,
            model_path=model_path,
            data_hash=data_hash,
            env_snapshot=self._env_snapshot(),
            test_range=test_range,
        )
        logger.info(
            "回测完成: annual=%.2f%% total=%.2f%% mdd=%.2f%% sharpe=%.2f trades=%d",
            metrics["annual_return"] * 100,
            metrics["total_return"] * 100,
            metrics["max_drawdown"] * 100,
            metrics["sharpe"],
            metrics["num_trades"],
        )
        return result

    # ------------------------------------------------------------------ #
    def compare_with_baseline(self) -> dict:
        """返回策略 vs buy&hold 的对比指标。

        Returns:
            ``{"strategy": {...}, "baseline": {...}, "excess_return": float}``。
        """
        result = self.run()
        baseline_ret = (
            float(result.baseline_equity[-1] / result.baseline_equity[0] - 1.0)
            if len(result.baseline_equity) >= 2 and result.baseline_equity[0] != 0
            else 0.0
        )
        return {
            "strategy": result.metrics,
            "baseline": {"total_return": baseline_ret},
            "excess_return": float(result.metrics["total_return"] - baseline_ret),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _buy_and_hold(samples: SampleSplit, cfg: EnvConfig) -> "np.ndarray":
        """买入持有基线权益曲线（首日满仓买入，持有至结束，含交易成本）。"""
        from stock_ai.env.rules import apply_slippage, max_affordable_lots, trade_cost

        prices = np.asarray(samples.prices, dtype="float64")
        if len(prices) == 0:
            return np.array([], dtype="float64")

        cash = float(cfg.initial_cash)
        buy_price = apply_slippage(float(prices[0]), "buy", cfg.slippage_rate)
        qty = max_affordable_lots(cash * (1.0 - cfg.commission_rate), buy_price, cfg.lot_size)
        if qty > 0:
            notional = qty * buy_price
            fee = trade_cost(notional, "buy", cfg.commission_rate,
                             cfg.stamp_tax_rate, cfg.min_commission)
            cash -= notional + fee
        else:
            qty = 0
        equity = cash + qty * prices
        return np.asarray(equity, dtype="float64")

    def _env_snapshot(self) -> dict:
        """环境配置快照。"""
        c = self.env_cfg
        return {
            "initial_cash": float(c.initial_cash),
            "commission_rate": float(c.commission_rate),
            "stamp_tax_rate": float(c.stamp_tax_rate),
            "slippage_rate": float(c.slippage_rate),
            "reward_type": str(c.reward_type),
            "lot_size": int(c.lot_size),
        }
