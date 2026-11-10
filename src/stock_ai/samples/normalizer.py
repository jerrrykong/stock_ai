"""归一化状态：训练窗口统计量的冻结与序列化（K9）。

``NormalizerState`` 在 ``sample build`` 时**仅用训练窗口 fit 并冻结**，写入
``meta.json``；``train``/``backtest``/``predict`` 一律复用，**任何环节禁止重新 fit**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stock_ai.utils.errors import ArtifactError

__all__ = ["NormalizerState"]


@dataclass
class NormalizerState:
    """冻结的标准化统计量。

    Attributes:
        mean: 每个特征列的均值，shape ``(n_features,)``。
        std: 每个特征列的标准差，shape ``(n_features,)``。
        columns: 特征列名（K8 权威顺序）。
        data_hash: 生成该状态时的原始数据摘要（K16）。
    """

    mean: "np.ndarray"
    std: "np.ndarray"
    columns: list[str] = field(default_factory=list)
    data_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的 dict。"""
        return {
            "mean": np.asarray(self.mean, dtype="float64").tolist(),
            "std": np.asarray(self.std, dtype="float64").tolist(),
            "columns": list(self.columns),
            "data_hash": self.data_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizerState":
        """从 dict 反序列化。

        Raises:
            ArtifactError: 缺少必要键。
        """
        if not isinstance(d, dict) or "mean" not in d or "std" not in d:
            raise ArtifactError("NormalizerState 缺少 mean/std 字段")
        return cls(
            mean=np.asarray(d["mean"], dtype="float64"),
            std=np.asarray(d["std"], dtype="float64"),
            columns=list(d.get("columns", [])),
            data_hash=str(d.get("data_hash", "")),
        )

    @property
    def n_features(self) -> int:
        """特征维度。"""
        return int(np.asarray(self.mean).shape[0])

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"NormalizerState(n_features={self.n_features}, data_hash={self.data_hash[:8]})"
