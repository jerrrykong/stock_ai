"""回测结果 dataclass 与报告序列化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stock_ai.utils.io import timestamp, write_json

__all__ = ["BacktestResult"]


@dataclass
class BacktestResult:
    """一次回测的完整产物。

    Attributes:
        metrics: 指标 dict（PRD §6 字段）。
        equity_curve: 策略权益曲线。
        dates: 日期数组。
        trades: 成交记录。
        baseline_equity: buy&hold 基线权益曲线。
        backend: 模型后端（``sb3`` / ``numpy``）。
        algo: 算法名。
        model_path: 模型文件路径。
        data_hash: 数据摘要（复现链）。
        env_snapshot: 环境配置快照。
        test_range: 测试区间 ``(start, end)``。
    """

    metrics: dict
    equity_curve: "np.ndarray"
    dates: "np.ndarray"
    trades: list[dict] = field(default_factory=list)
    baseline_equity: "np.ndarray" = field(default_factory=lambda: np.array([]))
    backend: str = "numpy"
    algo: str = "ppo"
    model_path: str = ""
    data_hash: str = ""
    env_snapshot: dict = field(default_factory=dict)
    test_range: tuple[str, str] = ("", "")

    def to_dict(self) -> dict[str, Any]:
        """构造报告 dict（架构 §4.6 结构）。"""
        return {
            "meta": {
                "ts": timestamp(),
                "backend": self.backend,
                "algo": self.algo,
                "model_path": self.model_path,
                "data_hash": self.data_hash,
                "test_range": {"start": self.test_range[0], "end": self.test_range[1]},
                "env": self.env_snapshot,
            },
            "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in self.metrics.items()},
            "equity_curve": np.asarray(self.equity_curve, dtype="float64").tolist(),
            "baseline_curve": np.asarray(self.baseline_equity, dtype="float64").tolist(),
            "dates": [str(d) for d in np.asarray(self.dates).tolist()],
            "trades": self.trades,
        }

    def to_report_meta(self) -> dict[str, Any]:
        """仅返回元信息部分。"""
        return self.to_dict()["meta"]

    def to_json(self, path: Path) -> Path:
        """写出报告 JSON。

        Args:
            path: 目标路径。

        Returns:
            目标路径。
        """
        return write_json(self.to_dict(), Path(path))
