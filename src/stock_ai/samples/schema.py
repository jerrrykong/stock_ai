"""样本层的 dataclass：SampleSplit / SampleBundle（K7）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from stock_ai.samples.normalizer import NormalizerState
from stock_ai.utils.io import write_json

__all__ = ["SampleSplit", "SampleBundle", "SampleBundleStub"]

#: npz 内固定键名（K7，大小写敏感，跨模块不得改名）
NPZ_KEYS: tuple[str, ...] = (
    "obs",
    "dates",
    "symbols",
    "prices",
    "requested_start",
    "requested_end",
    "n_steps",
    "n_features",
)


@dataclass
class SampleSplit:
    """单个切分（train/val/test）的样本张量集合。

    Attributes:
        obs: ``(T, window, n_features)`` float32 观测张量。
        dates: ``(T,)`` 日期字符串（``YYYYMMDD``）。
        symbols: ``(T,)`` 股票代码。
        prices: ``(T,)`` 用于回测结算的 close 价。
        requested_start: 请求区间起（``YYYYMMDD``）。
        requested_end: 请求区间止。
        n_steps: ``T``。
        n_features: 特征维度。
    """

    obs: "np.ndarray"
    dates: "np.ndarray"
    symbols: "np.ndarray"
    prices: "np.ndarray"
    requested_start: str = ""
    requested_end: str = ""
    n_steps: int = 0
    n_features: int = 0

    def __post_init__(self) -> None:
        self.obs = np.asarray(self.obs, dtype="float32")
        self.dates = np.asarray(self.dates, dtype="<U8")
        self.symbols = np.asarray(self.symbols)
        self.prices = np.asarray(self.prices, dtype="float64")
        if self.obs.ndim == 3:
            self.n_steps = int(self.obs.shape[0])
            self.n_features = int(self.obs.shape[2])
        else:
            self.n_steps = int(self.obs.shape[0]) if self.obs.ndim >= 1 else 0
            self.n_features = 0

    def to_npz_dict(self) -> dict[str, "np.ndarray"]:
        """转为 npz 键值对（K7）。"""
        return {
            "obs": self.obs,
            "dates": self.dates,
            "symbols": self.symbols,
            "prices": self.prices,
            "requested_start": np.array(self.requested_start),
            "requested_end": np.array(self.requested_end),
            "n_steps": np.array(self.n_steps, dtype="int64"),
            "n_features": np.array(self.n_features, dtype="int64"),
        }

    @classmethod
    def from_npz_dict(cls, d: dict[str, "np.ndarray"]) -> "SampleSplit":
        """从 npz 键值对构造。"""
        return cls(
            obs=d["obs"],
            dates=d["dates"],
            symbols=d["symbols"],
            prices=d["prices"],
            requested_start=str(d.get("requested_start", "")),
            requested_end=str(d.get("requested_end", "")),
            n_steps=int(d.get("n_steps", 0)),
            n_features=int(d.get("n_features", 0)),
        )


@dataclass
class SampleBundle:
    """训练/验证/测试三切分 + 冻结归一化状态 + 元信息。

    Attributes:
        train/val/test: 三份 :class:`SampleSplit`。
        normalizer: 冻结的 :class:`NormalizerState`（仅训练窗口 fit）。
        feature_names: 特征列名（K8 权威顺序）。
        window: 滑窗长度。
        seed: 全局随机种子。
        data_hash: 原始数据摘要（K16）。
        symbols: 参与构建的股票列表。
        embargo_days: embargo 交易日数。
        train_range/test_range: 区间元组。
    """

    train: SampleSplit
    val: SampleSplit
    test: SampleSplit
    normalizer: NormalizerState
    feature_names: list[str]
    window: int
    seed: int
    data_hash: str
    symbols: list[str] = field(default_factory=list)
    embargo_days: int = 0
    train_range: tuple[str, str] = ("", "")
    test_range: tuple[str, str] = ("", "")

    # ------------------------------------------------------------------ #
    def meta_dict(self) -> dict[str, Any]:
        """构造 ``data/samples/meta.json`` 的内容（符合架构 §4.3 JSON Schema）。"""
        return {
            "data_hash": self.data_hash,
            "seed": int(self.seed),
            "window": int(self.window),
            "symbols": list(self.symbols),
            "feature_names": list(self.feature_names),
            "train_range": {"start": self.train_range[0], "end": self.train_range[1]},
            "test_range": {"start": self.test_range[0], "end": self.test_range[1]},
            "embargo_days": int(self.embargo_days),
            "rows": {
                "train": int(self.train.n_steps),
                "val": int(self.val.n_steps),
                "test": int(self.test.n_steps),
            },
            "normalizer": self.normalizer.to_dict(),
            "created_at": datetime.now().isoformat(),
        }

    def save_meta(self, path: Path) -> None:
        """把 meta 写入指定路径（UTF-8）。"""
        write_json(self.meta_dict(), Path(path))


class SampleBundleStub:
    """仅含 meta 的轻量对象（供 predict 复用冻结 NormalizerState，K9）。

    Args:
        meta: ``data/samples/meta.json`` 的内容。
    """

    def __init__(self, meta: dict) -> None:
        self.meta: dict = dict(meta)
        self.data_hash: str = str(meta.get("data_hash", ""))
        self.seed: int = int(meta.get("seed", 42))
        self.window: int = int(meta.get("window", 20))
        self.feature_names: list[str] = list(meta.get("feature_names", []))
        self.symbols: list[str] = list(meta.get("symbols", []))
        self.normalizer: NormalizerState = NormalizerState.from_dict(meta.get("normalizer", {}))
