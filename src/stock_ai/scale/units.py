"""单元（Unit）定义、划分计划（SplitPlan）与扫描器（UnitScanner）。

- **单元主键**：``unit_id = <symbol>_<year>``（如 ``600519_2015``）；区间 = ``[YYYY0101, YYYY1231]``。
- **扫描期状态**：``available``（样本足够）/ ``insufficient``（数据不足）/ ``missing``（缓存缺失）。
- **划分**：仅在 ``available`` 单元上进行；``split.json`` **不含任何时钟字段**，
  从而保证同种子两次运行字节一致（AC3）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stock_ai.scale.atomic import atomic_write_json
from stock_ai.scale.paths import ScalePaths

__all__ = ["Unit", "SplitPlan", "UnitScanner", "SPLIT_ALGORITHM"]

#: 划分算法文案（固定，保证可复现审计）。
SPLIT_ALGORITHM: str = "shuffle(sorted(available), seed) -> slice"

#: 扫描期非 ``available`` 状态对应的可读原因（写入 ``units.json``）。
_STATUS_REASON: dict[str, str] = {
    "missing": "no_cache",
    "insufficient": "too_few_steps",
}


def _reason_for_status(status: str) -> str:
    """把扫描期状态映射为人类可读的跳过原因。

    Args:
        status: ``available|insufficient|missing``。

    Returns:
        ``missing -> "no_cache"``、``insufficient -> "too_few_steps"``、
        ``available -> ""``（无原因）。
    """
    return _STATUS_REASON.get(str(status), "")


@dataclass
class Unit:
    """一个「（代码 × 年份）」训练/检测单元。

    Attributes:
        unit_id: 主键 ``<symbol>_<year>``。
        symbol: 股票代码（6 位数字，或含点后缀的清洗后代码）。
        year: 年份。
        start: 区间起 ``YYYY0101``。
        end: 区间止 ``YYYY1231``。
    """

    unit_id: str
    symbol: str
    year: int
    start: str
    end: str

    @staticmethod
    def make(symbol: str, year: int) -> "Unit":
        """由 ``(symbol, year)`` 构造单元。"""
        sym = str(symbol)
        yr = int(year)
        return Unit(
            unit_id=f"{sym}_{yr}",
            symbol=sym,
            year=yr,
            start=f"{yr:04d}0101",
            end=f"{yr:04d}1231",
        )

    @staticmethod
    def parse(unit_id: str) -> "Unit":
        """从 ``unit_id`` 反解析（以**最后一个** ``_`` 切分，兼容含点/下划线 symbol）。

        Raises:
            ValueError: ``unit_id`` 不含 ``_`` 或年份段非整数。
        """
        sym, sep, yr = str(unit_id).rpartition("_")
        if not sep or not sym:
            raise ValueError(f"非法 unit_id: {unit_id!r}")
        return Unit.make(sym, int(yr))

    def to_dict(self) -> dict[str, Any]:
        """转为普通 dict。"""
        return {
            "unit_id": self.unit_id,
            "symbol": self.symbol,
            "year": int(self.year),
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Unit":
        """从 dict 构造。"""
        return cls(
            unit_id=str(d["unit_id"]),
            symbol=str(d["symbol"]),
            year=int(d["year"]),
            start=str(d["start"]),
            end=str(d["end"]),
        )


@dataclass
class SplitPlan:
    """训练/检测划分计划（可复现，无时钟字段）。

    Attributes:
        schema_version: 结构版本。
        random_seed: 随机种子。
        ratio: ``{train, detect}`` 比例。
        symbols_source: 名单来源（文件路径或描述）。
        symbols_count: 名单代码数。
        year_range: ``[start_year, end_year]``。
        units_total: 单元总数（含 insufficient/missing）。
        units_available: 可用单元数。
        train_units: 训练单元 ``unit_id`` 列表（已排序）。
        detect_units: 检测单元 ``unit_id`` 列表（已排序）。
        algorithm: 划分算法固定文案。
        config_hash: 配置指纹。
    """

    schema_version: int = 1
    random_seed: int = 42
    ratio: dict = field(default_factory=lambda: {"train": 0.8, "detect": 0.2})
    symbols_source: str = ""
    symbols_count: int = 0
    year_range: list[int] = field(default_factory=lambda: [0, 0])
    units_total: int = 0
    units_available: int = 0
    train_units: list[str] = field(default_factory=list)
    detect_units: list[str] = field(default_factory=list)
    algorithm: str = SPLIT_ALGORITHM
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转为 dict（**仅确定性字段**，不含 created_at，保障 AC3）。"""
        return {
            "schema_version": int(self.schema_version),
            "random_seed": int(self.random_seed),
            "ratio": {k: float(v) for k, v in self.ratio.items()},
            "symbols_source": str(self.symbols_source),
            "symbols_count": int(self.symbols_count),
            "year_range": [int(x) for x in self.year_range],
            "units_total": int(self.units_total),
            "units_available": int(self.units_available),
            "train_units": list(self.train_units),
            "detect_units": list(self.detect_units),
            "algorithm": str(self.algorithm),
            "config_hash": str(self.config_hash),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SplitPlan":
        """从 dict 构造。"""
        return cls(
            schema_version=int(d.get("schema_version", 1)),
            random_seed=int(d.get("random_seed", 42)),
            ratio=dict(d.get("ratio", {"train": 0.8, "detect": 0.2})),
            symbols_source=str(d.get("symbols_source", "")),
            symbols_count=int(d.get("symbols_count", 0)),
            year_range=[int(x) for x in d.get("year_range", [0, 0])],
            units_total=int(d.get("units_total", 0)),
            units_available=int(d.get("units_available", 0)),
            train_units=[str(x) for x in d.get("train_units", [])],
            detect_units=[str(x) for x in d.get("detect_units", [])],
            algorithm=str(d.get("algorithm", SPLIT_ALGORITHM)),
            config_hash=str(d.get("config_hash", "")),
        )

    def stage_units(self, phase: str) -> list[str]:
        """返回指定阶段的单元 ``unit_id`` 列表。"""
        if phase == "train":
            return list(self.train_units)
        if phase == "detect":
            return list(self.detect_units)
        raise ValueError(f"未知阶段: {phase!r}（应为 train/detect）")


class UnitScanner:
    """扫描「代码 × 年份」单元并生成 ``units.json`` + ``split.json``。

    Args:
        paths: 路径集合。
        builder: :class:`~stock_ai.samples.builder.SampleBuilder`（复用既有样本栈）。
        loader: :class:`~stock_ai.data.loader.StockDataLoader`（读取缓存）。
    """

    def __init__(self, paths: ScalePaths, builder: Any, loader: Any) -> None:
        """初始化。"""
        self.paths: ScalePaths = paths
        self.builder = builder
        self.loader = loader
        #: 扫描期每个单元的状态（``available|insufficient|missing``）。
        self.last_status: dict[str, str] = {}
        #: 扫描期每个单元的样本步数。
        self.last_steps: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    @property
    def min_steps(self) -> int:
        """判定为 ``available`` 所需的最小样本步数（``window + 2``）。"""
        window = int(getattr(self.builder.config, "window", 20))
        return int(window) + 2

    # ------------------------------------------------------------------ #
    def scan_units(
        self, symbols: list[str], year_range: tuple[int, int]
    ) -> list[Unit]:
        """扫描全部 ``(symbol, year)`` 组合并判定生成期状态。

        Args:
            symbols: 股票代码列表。
            year_range: ``(start_year, end_year)`` 闭区间。

        Returns:
            :class:`Unit` 列表（顺序 = 外层 symbol × 内层 year）。
            副作用：填充 :attr:`last_status` / :attr:`last_steps`。
        """
        y0, y1 = int(year_range[0]), int(year_range[1])
        years = list(range(y0, y1 + 1))
        units: list[Unit] = []
        self.last_status = {}
        self.last_steps = {}

        for sym in symbols:
            raw = None
            if self.loader is not None and self.loader.cache.exists(sym):
                try:
                    raw = self.loader.cache.read(sym)
                except Exception:  # noqa: BLE001 - 单只坏缓存不阻断整批
                    raw = None

            for y in years:
                unit = Unit.make(sym, y)
                units.append(unit)
                if raw is None:
                    self.last_status[unit.unit_id] = "missing"
                    self.last_steps[unit.unit_id] = 0
                    continue
                status, steps = self._probe_unit(raw, unit)
                self.last_status[unit.unit_id] = status
                self.last_steps[unit.unit_id] = steps
        return units

    def _probe_unit(self, raw: Any, unit: Unit) -> tuple[str, int]:
        """对单个单元窗口计数并判定 ``available|insufficient``。"""
        try:
            sub = _slice_window(raw, unit.start, unit.end)
            if sub is None or len(sub) < 2:
                return "insufficient", 0
            state = self.builder.engineer.fit(sub)
            split = self.builder.build_split(
                {unit.symbol: raw}, state, unit.start, unit.end
            )
            steps = int(split.n_steps)
        except Exception:  # noqa: BLE001 - 单单元异常降级为 insufficient
            return "insufficient", 0
        status = "available" if steps >= self.min_steps else "insufficient"
        return status, steps

    # ------------------------------------------------------------------ #
    def write_units(
        self,
        units: list[Unit],
        status: dict[str, str] | None = None,
        steps: dict[str, int] | None = None,
    ) -> Path:
        """原子写出 ``data/scale/units.json``。

        Args:
            units: 单元列表。
            status: ``unit_id -> 状态``（默认取 :attr:`last_status`）。
            steps: ``unit_id -> 步数``（默认取 :attr:`last_steps`）。

        Returns:
            目标路径。
        """
        st = status if status is not None else self.last_status
        sp = steps if steps is not None else self.last_steps
        counts = {"available": 0, "insufficient": 0, "missing": 0}
        records: list[dict[str, Any]] = []
        for u in units:
            s = st.get(u.unit_id, "missing")
            counts[s] = counts.get(s, 0) + 1
            rec = u.to_dict()
            rec["status"] = s
            rec["steps"] = int(sp.get(u.unit_id, 0))
            # 非 available 单元显式给出原因（no_cache / too_few_steps），便于 AC2 排查
            rec["reason"] = _reason_for_status(s)
            records.append(rec)

        payload = {
            "schema_version": 1,
            "units_total": len(units),
            "counts": counts,
            "units": records,
        }
        return atomic_write_json(payload, self.paths.units_json)

    # ------------------------------------------------------------------ #
    def build_split(
        self,
        available: list[Unit],
        seed: int,
        ratio: dict,
        chash: str,
        symbols_source: str,
        year_range: tuple[int, int],
        total: int,
        symbols_count: int = 0,
    ) -> SplitPlan:
        """在可用单元上做确定性随机划分（``shuffle(sorted) -> slice``）。

        Args:
            available: ``status=available`` 的单元列表。
            seed: 随机种子。
            ratio: ``{train, detect}`` 比例。
            chash: 配置指纹。
            symbols_source: 名单来源描述。
            year_range: ``(start_year, end_year)``。
            total: 单元总数（含不可用）。
            symbols_count: 名单代码数。

        Returns:
            :class:`SplitPlan`（``train_units`` / ``detect_units`` 已排序）。
        """
        ids = sorted(u.unit_id for u in available)
        rng = random.Random(int(seed))
        shuffled = list(ids)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_ratio = float(ratio.get("train", 0.8))
        n_train = int(round(n * train_ratio))
        n_train = max(0, min(n, n_train))

        train_units = sorted(shuffled[:n_train])
        detect_units = sorted(shuffled[n_train:])

        return SplitPlan(
            schema_version=1,
            random_seed=int(seed),
            ratio={"train": float(ratio.get("train", 0.8)),
                   "detect": float(ratio.get("detect", 0.2))},
            symbols_source=str(symbols_source),
            symbols_count=int(symbols_count),
            year_range=[int(year_range[0]), int(year_range[1])],
            units_total=int(total),
            units_available=int(n),
            train_units=train_units,
            detect_units=detect_units,
            algorithm=SPLIT_ALGORITHM,
            config_hash=str(chash),
        )

    # ------------------------------------------------------------------ #
    def write_split(self, plan: SplitPlan) -> Path:
        """原子写出 ``data/scale/split.json``（无时钟字段 → 可复现）。"""
        return atomic_write_json(plan.to_dict(), self.paths.split_json)


def _slice_window(raw: Any, start: str, end: str) -> Any:
    """取 ``raw`` 中 ``date ∈ [start, end]`` 且 ``date`` 升序的行（重置索引）。

    Args:
        raw: 原始日线 DataFrame。
        start: ``YYYYMMDD``。
        end: ``YYYYMMDD``。

    Returns:
        子集 DataFrame（按日期升序，索引重置）。
    """
    if raw is None or len(raw) == 0:
        return None
    day = np.asarray([int(str(x).replace("-", "")[:8]) for x in raw["date"].dt.strftime("%Y%m%d")])
    mask = (day >= int(start)) & (day <= int(end))
    sub = raw.loc[mask]
    if len(sub) == 0:
        return None
    return sub.sort_values("date", kind="stable").reset_index(drop=True)
