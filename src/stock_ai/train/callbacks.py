"""训练回调与曲线绘制辅助。

SB3 可用时可用其 ``BaseCallback`` 记录；NumPy 后端直接消费 trainer 的历史列表。
本模块不 import torch/sb3（用 duck-typing 兼容）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stock_ai.utils.logging import LoggerFactory

__all__ = ["RewardTracker", "dump_training_curve"]

logger = LoggerFactory.get(__name__)


class RewardTracker:
    """轻量奖励追踪器（不依赖 SB3）。

    记录每次评估的 mean reward，用于选 best model 与绘制曲线。
    """

    def __init__(self) -> None:
        """初始化空历史。"""
        self.timesteps: list[int] = []
        self.rewards: list[float] = []
        self.best_mean_reward: float = float("-inf")

    def update(self, timestep: int, mean_reward: float) -> bool:
        """记录一次评估。

        Args:
            timestep: 当前步数。
            mean_reward: 平均奖励。

        Returns:
            若刷新最佳记录返回 ``True``。
        """
        self.timesteps.append(int(timestep))
        self.rewards.append(float(mean_reward))
        improved = False
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = float(mean_reward)
            improved = True
        return improved

    def to_series(self) -> dict[str, list[float]]:
        """转为绘图友好的 ``{name: series}``。"""
        return {"mean_reward": list(self.rewards)}

    @property
    def empty(self) -> bool:
        """是否无任何记录。"""
        return len(self.rewards) == 0


def dump_training_curve(
    series: dict[str, list[float]],
    out_path: Path,
    title: str = "训练曲线",
) -> Path:
    """把训练曲线写为 PNG（内部调用 :class:`~stock_ai.eval.plot.EquityPlotter`）。

    Args:
        series: ``{曲线名: 数值列表}``。
        out_path: 输出 PNG 路径。
        title: 图标题。

    Returns:
        输出路径。
    """
    from stock_ai.eval.plot import EquityPlotter

    plotter = EquityPlotter()
    return plotter.plot_training_curve(series, Path(out_path), title=title)
