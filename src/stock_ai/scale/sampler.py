"""单元采样器：在「代码 × 年份」窗口上自拟合归一化状态并构建 train/val。

Q2：每个单元**自拟合**归一化状态（避免跨单元泄漏）；
Q3：单元内尾部按 ``sample.val_ratio`` 切 val，不单列 val 目录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stock_ai.samples.normalizer import NormalizerState
from stock_ai.samples.schema import SampleSplit
from stock_ai.scale.units import Unit, _slice_window
from stock_ai.utils.hashing import hash_dataframes

__all__ = ["UnitSample", "UnitSampler"]


@dataclass
class UnitSample:
    """单个单元的采样结果。

    Attributes:
        train: 单元窗口（整年）样本切分。
        val: 单元尾部验证切分（按 ``val_ratio``）。
        state: 在单元窗口上自拟合并冻结的归一化状态。
        n_steps: 训练样本步数。
        data_hash: 单元窗口数据摘要。
    """

    train: SampleSplit
    val: SampleSplit
    state: NormalizerState
    n_steps: int = 0
    data_hash: str = ""

    def __post_init__(self) -> None:
        """缺省时由 ``train`` 推断 ``n_steps``。"""
        if not self.n_steps:
            self.n_steps = int(getattr(self.train, "n_steps", 0))


class UnitSampler:
    """封装 :class:`~stock_ai.samples.builder.SampleBuilder`，在单元窗口上采样。

    Args:
        builder: 既有样本构建器（复用 ``FeatureEngineer`` / ``build_split``）。
    """

    def __init__(self, builder: Any) -> None:
        """初始化。"""
        self.builder = builder

    # ------------------------------------------------------------------ #
    def sample(self, unit: Unit, raw_df: Any, val_ratio: float) -> UnitSample:
        """在单元窗口 ``[unit.start, unit.end]`` 上采样。

        Args:
            unit: 目标单元。
            raw_df: 该 symbol 的原始日线（全景，供窗口切片）。
            val_ratio: 尾部验证集比例。

        Returns:
            :class:`UnitSample`。

        Raises:
            ArtifactError: 单元窗口数据不足（行数 < 2）。
        """
        from stock_ai.utils.errors import ArtifactError

        window_df = _slice_window(raw_df, unit.start, unit.end)
        if window_df is None or len(window_df) < 2:
            raise ArtifactError(
                f"单元数据不足: {unit.unit_id}（<2 行）",
                context={"unit_id": unit.unit_id},
            )

        # ---- 1) 仅用【本单元窗口】fit 归一化状态（Q2）----
        state: NormalizerState = self.builder.engineer.fit(window_df)
        data_hash = hash_dataframes({unit.symbol: window_df})
        state.data_hash = data_hash

        # ---- 2) 构建 train / val 切分（复用冻结状态）----
        train_split: SampleSplit = self.builder.build_split(
            {unit.symbol: raw_df}, state, unit.start, unit.end
        )
        vs, ve = self._val_range(unit.start, unit.end, val_ratio)
        val_split: SampleSplit = self.builder.build_split(
            {unit.symbol: raw_df}, state, vs, ve
        )

        return UnitSample(
            train=train_split,
            val=val_split,
            state=state,
            n_steps=int(train_split.n_steps),
            data_hash=data_hash,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _val_range(start: str, end: str, val_ratio: float) -> tuple[str, str]:
        """从单元窗口尾部切出验证区间（借鉴 ``SampleBuilder._val_range``）。"""
        import pandas as pd

        ratio = max(0.0, min(0.9, float(val_ratio or 0.0)))
        if ratio <= 0.0:
            return end, end
        ts = pd.to_datetime(start, format="%Y%m%d")
        te = pd.to_datetime(end, format="%Y%m%d")
        total_days = (te - ts).days
        if total_days <= 30:
            return end, end
        split_ts = te - pd.to_timedelta(int(total_days * ratio), unit="D")
        return split_ts.strftime("%Y%m%d"), end
