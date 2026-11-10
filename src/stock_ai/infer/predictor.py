"""预测器：加载模型 + 复用**冻结**归一化状态，逐日 stateful 推理（K9）。

关键约束：
    - 复用 ``data/samples/meta.json`` 里的 :class:`NormalizerState`，**绝不重新 fit**。
    - 逐日 stateful：持仓状态跨交易日延续，T+1 规则生效（复用 TradingEnv）。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from stock_ai.config.schema import EnvConfig
from stock_ai.samples.features import FeatureEngineer
from stock_ai.samples.normalizer import NormalizerState
from stock_ai.utils.errors import ArtifactError
from stock_ai.utils.io import write_json
from stock_ai.utils.logging import LoggerFactory

__all__ = ["Predictor"]

logger = LoggerFactory.get(__name__)


class Predictor:
    """逐日 stateful 预测器。

    Args:
        model: :class:`~stock_ai.train.base.Enrollment` 协议实现。
        env_cfg: 环境配置。
        normalizer: **冻结**的归一化状态（来自 meta.json）。
        feature_names: 特征列名（K8 权威顺序）。
        window: 滑窗长度。
        mode: 动作模式。
    """

    def __init__(
        self,
        model: "Any",
        env_cfg: EnvConfig,
        normalizer: NormalizerState,
        feature_names: list[str],
        window: int,
        mode: str = "discrete",
    ) -> None:
        """初始化。"""
        if normalizer is None or normalizer.n_features == 0:
            raise ArtifactError("Predictor 需要冻结的 NormalizerState（K9）")
        self.model = model
        self.env_cfg = env_cfg
        self.normalizer = normalizer
        self.feature_names = list(feature_names)
        self.window = int(window)
        self.mode = str(mode).lower()
        self.engineer = FeatureEngineer(self.feature_names)

    # ------------------------------------------------------------------ #
    def predict_from_frames(
        self,
        frames: dict[str, "Any"],
        symbol: str | None = None,
    ) -> list["PredictionRow"]:
        """对给定原始日线做逐日 stateful 预测。

        Args:
            frames: ``{symbol: raw_df}``。
            symbol: 仅预测该代码（None 预测全部）。

        Returns:
            :class:`PredictionRow` 列表。
        """
        from stock_ai.infer.schema import ACTION_LABELS, PredictionRow

        rows: list[PredictionRow] = []
        targets = (
            {symbol: frames[symbol]} if symbol and symbol in frames else frames
        )

        for sym, raw in targets.items():
            if raw is None or len(raw) < 2:
                continue

            # 复用冻结状态 transform（绝不重新 fit）
            normed = self.engineer.transform(raw, self.normalizer)  # (T, F)
            prices = np.asarray(raw["close"], dtype="float64")
            dates = raw["date"].dt.strftime("%Y%m%d").tolist()

            obs = self._make_obs(normed)          # (T, window, F)
            actions = self._rollout(obs)          # (T,) 动作

            # 由动作序列推导目标仓位与信号（stateful）
            position = 0.0
            for i in range(len(actions)):
                act = int(actions[i])
                if act == 1:
                    position = min(1.0, position + 1.0) if position < 0.5 else 1.0
                elif act == 2:
                    position = 0.0
                rows.append(
                    PredictionRow(
                        date=str(dates[i]),
                        symbol=str(sym),
                        action=ACTION_LABELS.get(act, "持有"),
                        target_position=float(position),
                        signal=float(1.0 if act == 1 else (-1.0 if act == 2 else 0.0)),
                        price=float(prices[i]),
                        prob=None,
                    )
                )
        logger.info("预测完成: %d 条记录", len(rows))
        return rows

    # ------------------------------------------------------------------ #
    def predict_range(
        self,
        symbols: list[str],
        start: str,
        end: str,
        loader: "Any",
    ) -> list["PredictionRow"]:
        """对指定区间预测（缓存优先，缺失才联网）。

        Args:
            symbols: 股票列表。
            start: 起始 ``YYYYMMDD``。
            end: 结束 ``YYYYMMDD``。
            loader: :class:`~stock_ai.data.loader.StockDataLoader`。

        Returns:
            :class:`PredictionRow` 列表。

        Raises:
            ConfigError: ``start > end``（P2-6）。
        """
        from stock_ai.utils.dates import validate_range

        # ---- 参数前置校验（P2-6）----
        validate_range(start, end, label="预测区间", hint="请检查 --start/--end")

        frames: dict[str, "Any"] = {}
        for sym in symbols:
            df = None
            if loader.cache.exists(sym):
                cached = loader.cache.read(sym)
                day = cached["date"].dt.strftime("%Y%m%d")
                mask = (day >= start) & (day <= end)
                sub = cached.loc[mask]
                if len(sub) >= 2:
                    df = sub.reset_index(drop=True)
            if df is None:
                # A5.4：联网拉取使用的是 config 的 adjust；若与既有缓存口径不一致，
                # 必须告警（口径混用会污染预测结果）。
                try:
                    cfg_adjust = str(getattr(loader.config, "adjust", "") or "").lower()
                    prev_adjust = str(
                        (
                            (loader.cache.read_meta().get("symbols", {}) or {})
                            .get(sym, {})
                            or {}
                        ).get("adjust", "")
                        or ""
                    ).lower()
                    if prev_adjust and cfg_adjust and prev_adjust != cfg_adjust:
                        logger.warning(
                            "预测联网拉取复权口径(%s)与既有缓存(%s)不一致，"
                            "可能造成口径混用: %s",
                            cfg_adjust, prev_adjust, sym,
                        )
                except Exception:  # noqa: BLE001 - 告警逻辑不得影响主流程
                    pass
                logger.info("缓存不足，联网拉取 %s (%s~%s)", sym, start, end)
                df = loader.fetch_symbol(sym, start, end)
            frames[sym] = df
        return self.predict_from_frames(frames)

    # ------------------------------------------------------------------ #
    def to_csv(self, rows: list["PredictionRow"], out: Path) -> Path:
        """写出预测 CSV（列 = date,symbol,action,target_position,signal,price）。

        Args:
            rows: 预测记录。
            out: 输出路径。

        Returns:
            输出路径。
        """
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["date", "symbol", "action", "target_position", "signal", "price"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_dict())
        logger.info("预测 CSV 已写出: %s", p)
        return p

    def to_json(self, rows: list["PredictionRow"], out: Path) -> Path:
        """写出预测 JSON。

        Args:
            rows: 预测记录。
            out: 输出路径。

        Returns:
            输出路径。
        """
        p = Path(out)
        payload = {
            "count": len(rows),
            "columns": ["date", "symbol", "action", "target_position", "signal", "price"],
            "rows": [r.to_dict() for r in rows],
        }
        write_json(payload, p)
        logger.info("预测 JSON 已写出: %s", p)
        return p

    # ------------------------------------------------------------------ #
    def _make_obs(self, normed: "np.ndarray") -> "np.ndarray":
        """由归一化特征构造观测张量 ``(T, window, F)``（前端复制填充）。"""
        t, f = normed.shape
        w = self.window
        out = np.zeros((t, w, f), dtype="float32")
        for i in range(t):
            lo = max(0, i - w + 1)
            seg = normed[lo : i + 1]
            if seg.shape[0] < w:
                pad = np.repeat(seg[0:1], w - seg.shape[0], axis=0)
                seg = np.concatenate([pad, seg], axis=0)
            out[i] = seg
        return out

    def _rollout(self, obs: "np.ndarray") -> "np.ndarray":
        """逐日调用模型 predict（stateful 由调用方的 position 维护）。

        Args:
            obs: ``(T, window, F)``。

        Returns:
            ``(T,)`` 动作数组。
        """
        t = obs.shape[0]
        actions = np.zeros(t, dtype="int64")
        for i in range(t):
            action, _ = self.model.predict(obs[i], deterministic=True)
            arr = np.asarray(action).reshape(-1)
            actions[i] = int(arr[0]) if arr.size else 0
        return actions
