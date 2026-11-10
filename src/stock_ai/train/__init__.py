"""训练层：可插拔后端（SB3 / 纯 NumPy PPO 兜底）。"""

from __future__ import annotations

from stock_ai.train.base import Enrollment, TrainResult
from stock_ai.train.factory import TrainerFactory

__all__: list[str] = ["Enrollment", "TrainResult", "TrainerFactory"]
