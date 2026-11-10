"""训练层基础契约：``Enrollment`` Protocol + ``TrainResult`` dataclass。

``eval/`` 与 ``infer/`` 只依赖本模块的 Protocol，不依赖具体 SB3 类 —— 后端可插拔。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["TrainResult", "Enrollment"]


@dataclass
class TrainResult:
    """一次训练的产物摘要。

    Attributes:
        run_dir: 运行目录（``logs/<run>/``）。
        model_path: 模型 ``.zip`` 路径。
        algo: 算法名（``ppo`` 等）。
        backend: 后端名（``sb3`` / ``numpy``）。
        total_timesteps: 实际训练步数。
        best_mean_reward: 最佳平均奖励。
        elapsed_sec: 耗时（秒）。
        curve_path: 训练曲线 PNG 路径。
    """

    run_dir: Path
    model_path: Path
    algo: str
    backend: str
    total_timesteps: int
    best_mean_reward: float
    elapsed_sec: float
    curve_path: Path = field(default_factory=lambda: Path(""))

    def to_dict(self) -> dict[str, Any]:
        """转为普通 dict（Path -> str）。"""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


@runtime_checkable
class Enrollment(Protocol):
    """训练器端口（Ports & Adapters 的 Port）。

    所有后端（:class:`~stock_ai.train.sb3_trainer.SB3Trainer`、
    :class:`~stock_ai.train.ppo_numpy.NumpyPPOTrainer`）必须实现本协议，
    使 ``backtest`` / ``predict`` 与具体算法解耦。
    """

    #: 算法名标识
    algo_name: str

    def learn(self, total_timesteps: int, seed: int | None = None) -> None:
        """训练。

        Args:
            total_timesteps: 目标训练步数。
            seed: 覆盖随机种子。
        """
        ...

    def predict(
        self, obs: "np.ndarray", deterministic: bool = True
    ) -> tuple["np.ndarray", Any]:
        """对观测预测动作。

        Args:
            obs: 观测（``(window, n_features)`` 或带 batch 维）。
            deterministic: 是否贪心。

        Returns:
            ``(action, None)``。
        """
        ...

    def save(self, path: Path) -> None:
        """保存模型到 ``path``（``.zip``）。"""
        ...

    @classmethod
    def load(cls, path: Path, env: "Any" = None) -> "Enrollment":
        """从 ``path`` 加载模型。"""
        ...
