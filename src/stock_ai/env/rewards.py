"""奖励计算器（K14 —— 纯函数式，无副作用）。

``r = ret − λ₁·drawdown − λ₂·turnover − λ₃·|position_ratio|``
其中 ``ret = (value - prev_value) / prev_value``，drawdown 为当前回撤（相对历史峰值）。

``excess`` 类型把 ``ret`` 换成相对买入持有基准的**超额收益**
``excess = ret − benchmark_return``，其余惩罚项不变，用于把训练目标从
「绝对收益」改为「相对基准的超额收益」。
"""

from __future__ import annotations

import numpy as np

from stock_ai.config.schema import EnvConfig
from stock_ai.utils.errors import ConfigError

__all__ = ["RewardCalculator", "SUPPORTED_REWARDS"]

#: 支持的奖励类型白名单（P2-5）。``excess`` 为相对基准超额收益（增量）。
SUPPORTED_REWARDS: frozenset[str] = frozenset(
    {"gross", "drawdown_penalty", "sharpe", "excess"}
)


class RewardCalculator:
    """多目标结构化奖励计算器。

    Args:
        cfg: 环境配置（提供 ``reward_type`` 与各 λ 系数）。

    Raises:
        ConfigError: ``cfg.reward_type`` 不在 :data:`SUPPORTED_REWARDS` 内。
    """

    def __init__(self, cfg: EnvConfig) -> None:
        """初始化。"""
        self.cfg: EnvConfig = cfg
        self._peak: float = 0.0

        # ---- 参数前置校验（P2-5）：非法 reward_type 明确报错而非静默降级 ----
        reward_type = str(cfg.reward_type or "drawdown_penalty").lower()
        if reward_type not in SUPPORTED_REWARDS:
            raise ConfigError(
                f"不支持的奖励类型: {cfg.reward_type!r}"
                f"（可选: {sorted(SUPPORTED_REWARDS)}）",
                context={"reward_type": cfg.reward_type, "valid": sorted(SUPPORTED_REWARDS)},
            )

    # ------------------------------------------------------------------ #
    def reset(self, initial_value: float | None = None) -> None:
        """重置历史峰值（每个 episode 开始时调用）。"""
        self._peak = float(initial_value) if initial_value else 0.0

    def compute(
        self,
        prev_value: float,
        value: float,
        turnover: float = 0.0,
        position_ratio: float = 0.0,
        benchmark_return: float = 0.0,
    ) -> float:
        """计算单步奖励。

        Args:
            prev_value: 上一步组合总市值。
            value: 当前组合总市值。
            turnover: 本步换手（成交额 / 总市值）。
            position_ratio: 当前仓位比例（绝对值参与杠杆惩罚）。
            benchmark_return: 本步买入持有基准收益（仅 ``excess`` 类型使用；
                其它类型忽略该参数，行为逐位不变）。

        Returns:
            标量奖励。
        """
        pv = float(prev_value)
        v = float(value)

        # 收益率
        if pv > 1e-9:
            ret = (v - pv) / pv
        else:
            ret = 0.0

        # 回撤（相对历史峰值）
        self._peak = max(self._peak, v)
        drawdown = 0.0
        if self._peak > 1e-9:
            drawdown = max(0.0, (self._peak - v) / self._peak)

        cfg = self.cfg
        reward_type = str(cfg.reward_type or "drawdown_penalty").lower()

        if reward_type == "gross":
            return float(ret)

        if reward_type == "sharpe":
            # 单步夏普近似：收益 / (|收益| + eps)，再减去成本项
            denom = abs(ret) + 1e-6
            base = ret / denom
            return float(
                base
                - cfg.lambda_turnover * float(turnover)
                - cfg.lambda_leverage * abs(float(position_ratio))
            )

        if reward_type == "excess":
            # 相对买入持有基准的超额收益，再减去三项惩罚（与 drawdown_penalty 同构）。
            # 数值防御：任何非有限分量取 0，绝不产生 NaN/Inf。
            bench = float(benchmark_return)
            if not np.isfinite(bench):
                bench = 0.0
            base_ret = ret if np.isfinite(ret) else 0.0
            dd = drawdown if np.isfinite(drawdown) else 0.0
            turn = float(turnover)
            if not np.isfinite(turn):
                turn = 0.0
            pos = abs(float(position_ratio))
            if not np.isfinite(pos):
                pos = 0.0
            excess = base_ret - bench
            value_reward = (
                excess
                - cfg.lambda_drawdown * dd
                - cfg.lambda_turnover * turn
                - cfg.lambda_leverage * pos
            )
            return float(value_reward) if np.isfinite(value_reward) else 0.0

        # 默认：drawdown_penalty
        return float(
            ret
            - cfg.lambda_drawdown * drawdown
            - cfg.lambda_turnover * float(turnover)
            - cfg.lambda_leverage * abs(float(position_ratio))
        )
