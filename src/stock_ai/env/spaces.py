"""Gymnasium 空间构造。"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

__all__ = ["build_observation_space", "build_action_space", "ACTION_NAMES"]

#: 离散动作语义：0=持有, 1=买, 2=卖（PRD §6 契约）
ACTION_NAMES: dict[int, str] = {0: "持有", 1: "买", 2: "卖"}


def build_observation_space(n_features: int, window: int) -> spaces.Space:
    """构造观测空间 ``Box(-inf, inf, (window, n_features), float32)``。

    Args:
        n_features: 特征维度。
        window: 历史窗口长度。

    Returns:
        :class:`gymnasium.spaces.Box`。
    """
    nf = int(max(1, n_features))
    w = int(max(1, window))
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(w, nf),
        dtype=np.float32,
    )


def build_action_space(mode: str = "discrete") -> spaces.Space:
    """构造动作空间。

    Args:
        mode: ``"discrete"`` -> ``Discrete(3)``（0=持有,1=买入,2=卖出）；
              ``"continuous"`` -> ``Box(0,1,(1,), float32)``（SAC 目标仓位）。

    Returns:
        对应的 :mod:`gymnasium.spaces` 空间。
    """
    m = str(mode).lower()
    if m == "continuous":
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    return spaces.Discrete(3)
