"""特征工程：指标计算（纯函数）+ 防泄漏 fit/transform（K8/K9）。

设计要点：
    - ``build_indicators`` 只用 t 及之前的数据（滚动窗口，不打乱、不居中）。
    - ``fit`` **仅使用传入的 DataFrame**（调用方保证是训练窗口）计算 mean/std。
    - ``transform`` 使用外部传入的冻结状态，绝不重新计算统计量。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stock_ai.samples.normalizer import NormalizerState
from stock_ai.utils.errors import ArtifactError
from stock_ai.utils.logging import LoggerFactory

__all__ = ["FeatureEngineer", "BASE_COLUMNS", "DERIVED_COLUMNS"]

logger = LoggerFactory.get(__name__)

#: 直接来自 OHLCV 的原始特征列
BASE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: 需要计算的技术指标列
DERIVED_COLUMNS: tuple[str, ...] = (
    "ma5", "ma10", "ma20", "momentum", "volatility", "volume_ratio",
)

#: 所有支持的列（用于校验）
ALL_SUPPORTED: tuple[str, ...] = BASE_COLUMNS + DERIVED_COLUMNS


class FeatureEngineer:
    """特征工程器。

    Args:
        feature_names: 期望产出的特征列名列表（K8 权威顺序）。
    """

    def __init__(self, feature_names: list[str]) -> None:
        """初始化。

        Raises:
            ArtifactError: 出现不支持的特征名。
        """
        unknown = [f for f in feature_names if f not in ALL_SUPPORTED]
        if unknown:
            raise ArtifactError(
                f"不支持的特征名: {unknown}（支持: {list(ALL_SUPPORTED)}）",
                context={"unknown": unknown},
            )
        self.feature_names: list[str] = list(feature_names)

    # ------------------------------------------------------------------ #
    def build_indicators(self, df: "Any", names: list[str] | None = None) -> "Any":
        """计算技术指标并返回对齐 ``names`` 顺序的特征 DataFrame。

        Args:
            df: 原始日线（含 OHLCV）。
            names: 目标列名（None 取 ``self.feature_names``）。

        Returns:
            仅含目标列的 DataFrame（索引与输入对齐，无 NaN）。

        Raises:
            ArtifactError: 输入缺少必要基础列。
        """
        import pandas as pd

        target = list(names) if names else list(self.feature_names)
        missing_base = [c for c in ("close", "volume") if c not in df.columns]
        if missing_base:
            raise ArtifactError(f"特征计算缺少基础列: {missing_base}")

        out = pd.DataFrame(index=df.index)
        close = pd.to_numeric(df["close"], errors="coerce").astype("float64")
        volume = pd.to_numeric(df["volume"], errors="coerce").astype("float64")

        # ---- 原始列 ----
        for col in BASE_COLUMNS:
            if col in target:
                if col in df.columns:
                    out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                else:
                    out[col] = np.nan

        # ---- 均线（只用 t 及之前：rolling）----
        if "ma5" in target:
            out["ma5"] = close.rolling(window=5, min_periods=1).mean()
        if "ma10" in target:
            out["ma10"] = close.rolling(window=10, min_periods=1).mean()
        if "ma20" in target:
            out["ma20"] = close.rolling(window=20, min_periods=1).mean()

        # ---- 动量：N 日收益率（t / t-N - 1）----
        if "momentum" in target:
            out["momentum"] = close.pct_change(periods=10).fillna(0.0)

        # ---- 波动率：20 日对数收益滚动标准差 ----
        if "volatility" in target:
            log_ret = np.log(close.replace(0.0, np.nan)).diff()
            out["volatility"] = log_ret.rolling(window=20, min_periods=2).std().fillna(0.0)

        # ---- 量比：当日成交量 / 过去 5 日均量 ----
        if "volume_ratio" in target:
            vol_ma5 = volume.rolling(window=5, min_periods=1).mean()
            ratio = volume / vol_ma5.replace(0.0, np.nan)
            out["volume_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # 按目标顺序排列，缺失列补 0
        for col in target:
            if col not in out.columns:
                out[col] = 0.0
        out = out[target]

        # 清洗 inf / nan（用 0 填充，保持数值稳定）
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out.astype("float64")

    # ------------------------------------------------------------------ #
    def fit(self, df: "Any") -> NormalizerState:
        """仅用传入 DataFrame 计算 mean/std 并冻结（K9）。

        Args:
            df: **训练窗口**的原始日线（调用方保证无测试数据混入）。

        Returns:
            冻结的 :class:`NormalizerState`。
        """
        feat = self.build_indicators(df, self.feature_names)
        arr = feat.to_numpy(dtype="float64")
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        std = np.where(std < 1e-9, 1.0, std)  # 常数列防除零
        return NormalizerState(mean=mean, std=std, columns=list(self.feature_names))

    def transform(self, df: "Any", state: NormalizerState) -> "np.ndarray":
        """按冻结状态标准化（不重新 fit）。

        Args:
            df: 原始日线。
            state: 冻结的归一化状态。

        Returns:
            ``(T, n_features)`` 的 float32 矩阵。

        Raises:
            ArtifactError: 特征列不匹配。
        """
        feat = self.build_indicators(df, self.feature_names)
        if state.columns and list(state.columns) != list(self.feature_names):
            raise ArtifactError(
                "归一化状态列与当前特征列不一致（K8/K9 违规）",
                context={"state": state.columns, "engineer": self.feature_names},
            )
        arr = feat.to_numpy(dtype="float64")
        if arr.shape[1] != state.n_features:
            raise ArtifactError(
                f"特征维度不匹配: data={arr.shape[1]} state={state.n_features}"
            )
        std = np.where(np.asarray(state.std) < 1e-9, 1.0, np.asarray(state.std))
        normed = (arr - np.asarray(state.mean)) / std
        normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
        return normed.astype("float32")

    # ------------------------------------------------------------------ #
    def make_obs_matrix(
        self,
        feat: "Any",
        state: NormalizerState,
        window: int,
    ) -> "np.ndarray":
        """把特征矩阵滑窗拼接为 ``(T, window, n_features)`` 观测张量。

        第 t 个样本使用 ``feat[t-window+1 : t+1]``；对前 ``window-1`` 行使用
        前端复制填充（保证输出行数与输入一致，便于与 dates/prices 对齐）。

        Args:
            feat: 原始日线 DataFrame。
            state: 冻结归一化状态。
            window: 历史窗口长度。

        Returns:
            ``(T, window, n_features)`` float32 张量。
        """
        arr = self.transform(feat, state)  # (T, F)
        t, f = arr.shape
        w = int(window)
        if w < 1:
            raise ArtifactError(f"window 必须 >= 1，当前 {w}")
        out = np.zeros((t, w, f), dtype="float32")
        for i in range(t):
            lo = max(0, i - w + 1)
            seg = arr[lo : i + 1]  # (<=w, F)
            if seg.shape[0] < w:
                pad = np.repeat(seg[0:1], w - seg.shape[0], axis=0)
                seg = np.concatenate([pad, seg], axis=0)
            out[i] = seg
        return out
