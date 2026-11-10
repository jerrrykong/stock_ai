"""单元归一化状态持久化（Q2 追加要求）。

- **训练**：单元窗口 ``fit`` 得到的 :class:`NormalizerState` 持久化到
  ``models/units/<unit>.norm.json``。
- **检测**：加载 assigned model 的**同一份**状态做 ``transform``，
  **绝不重新 fit**（避免训练/推理分布漂移，K9 不变量）。
"""

from __future__ import annotations

from pathlib import Path

from stock_ai.scale.atomic import atomic_write_json
from stock_ai.scale.paths import ScalePaths
from stock_ai.samples.normalizer import NormalizerState
from stock_ai.utils.io import read_json

__all__ = ["NormalizerStore"]


class NormalizerStore:
    """``NormalizerState`` 的磁盘存取（原子写）。

    Args:
        paths: 路径集合。
    """

    def __init__(self, paths: ScalePaths) -> None:
        """初始化。"""
        self.paths: ScalePaths = paths

    # ------------------------------------------------------------------ #
    def save(self, unit_id: str, state: NormalizerState) -> Path:
        """原子保存归一化状态到 ``models/units/<unit>.norm.json``。

        Args:
            unit_id: 单元主键。
            state: 冻结的归一化状态。

        Returns:
            目标路径。
        """
        return atomic_write_json(state.to_dict(), self.paths.norm(unit_id))

    def load(self, unit_id: str) -> NormalizerState:
        """加载归一化状态。

        Raises:
            ArtifactError: 文件缺失或格式非法（由 ``read_json`` /
                ``NormalizerState.from_dict`` 抛出）。
        """
        return NormalizerState.from_dict(read_json(self.paths.norm(unit_id)))

    def exists(self, unit_id: str) -> bool:
        """归一化状态文件是否存在且非空。"""
        p = self.paths.norm(unit_id)
        try:
            return p.exists() and p.stat().st_size > 0
        except OSError:  # pragma: no cover
            return False
