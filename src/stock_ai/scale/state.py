"""单元状态与阶段状态（``state/{train,detect}.json``）的原子持久化与崩溃恢复。

- **双重校验**：单元「完成」= 状态文件 ``done`` **且** 产物文件存在且非空。
- **崩溃恢复**：启动时扫描 ``units``：``running`` 但产物无效 → ``pending``；
  ``done`` 但产物缺失 → ``pending``。
- **原子写盘**：状态文件统一走 :func:`~stock_ai.scale.atomic.atomic_write_json`。
- **计数现算**：``counts`` 每次由 :meth:`ScaleState.recompute_counts` 现算，避免漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from stock_ai.scale.atomic import atomic_write_json
from stock_ai.scale.paths import ScalePaths
from stock_ai.utils.io import read_json

__all__ = ["UnitState", "ScaleState", "StateStore"]

#: 单元状态枚举。
UNIT_STATUSES: frozenset[str] = frozenset(
    {"pending", "running", "done", "failed", "skipped"}
)

#: 进程状态枚举。
PROCESS_STATES: frozenset[str] = frozenset(
    {"idle", "running", "pause_requested", "paused",
     "stop_requested", "stopped", "finished"}
)


def _now() -> str:
    """当前时间戳 ``YYYYMMDD_HHMMSS``。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class UnitState:
    """单个单元的状态明细。

    Attributes:
        status: ``pending|running|done|failed|skipped``。
        started_at: 开始时间。
        finished_at: 结束时间。
        duration_sec: 耗时（秒）。
        model_path: 阶段产物路径（train：模型 zip）。
        assigned_model: detect 阶段实际使用的模型 ``unit_id``。
        report_path: 阶段产物路径（detect：报告 json）。
        metrics: detect 阶段指标 dict。
        seed: 该单元使用的随机种子。
        data_hash: 该单元数据摘要。
        error: 失败原因（``failed``）。
        reason: 跳过原因（``skipped``）。
    """

    status: str = "pending"
    started_at: str = ""
    finished_at: str = ""
    duration_sec: float = 0.0
    model_path: str = ""
    assigned_model: str = ""
    report_path: str = ""
    metrics: dict | None = None
    seed: int = 0
    data_hash: str = ""
    error: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转为普通 dict。"""
        return {
            "status": str(self.status),
            "started_at": str(self.started_at),
            "finished_at": str(self.finished_at),
            "duration_sec": float(self.duration_sec),
            "model_path": str(self.model_path),
            "assigned_model": self.assigned_model or None,
            "report_path": str(self.report_path),
            "metrics": self.metrics,
            "seed": int(self.seed),
            "data_hash": str(self.data_hash),
            "error": self.error or None,
            "reason": self.reason or None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UnitState":
        """从 dict 构造。"""
        return cls(
            status=str(d.get("status", "pending")),
            started_at=str(d.get("started_at", "") or ""),
            finished_at=str(d.get("finished_at", "") or ""),
            duration_sec=float(d.get("duration_sec", 0.0) or 0.0),
            model_path=str(d.get("model_path", "") or ""),
            assigned_model=str(d.get("assigned_model", "") or ""),
            report_path=str(d.get("report_path", "") or ""),
            metrics=d.get("metrics"),
            seed=int(d.get("seed", 0) or 0),
            data_hash=str(d.get("data_hash", "") or ""),
            error=str(d.get("error", "") or ""),
            reason=str(d.get("reason", "") or ""),
        )


@dataclass
class ScaleState:
    """一个阶段（train / detect）的完整状态。

    Attributes:
        schema_version: 结构版本。
        phase: ``train|detect``。
        process_state: 进程态枚举。
        pid: 进程号。
        created_at: 状态创建时间。
        updated_at: 最近更新时间。
        seed: 全局随机种子。
        backend: 训练后端偏好。
        config_hash: 配置指纹。
        total: 目标单元总数。
        counts: 各状态计数（现算）。
        current_unit: 当前处理单元。
        started_at: 本轮开始时间。
        elapsed_sec: 已用时（秒）。
        avg_unit_sec: 单单元平均耗时。
        eta_sec: 预计剩余时间（秒）。
        units: ``unit_id -> UnitState``。
    """

    schema_version: int = 1
    phase: str = "train"
    process_state: str = "idle"
    pid: int = 0
    created_at: str = ""
    updated_at: str = ""
    seed: int = 42
    backend: str = "auto"
    config_hash: str = ""
    total: int = 0
    counts: dict = field(default_factory=dict)
    current_unit: str = ""
    started_at: str = ""
    elapsed_sec: float = 0.0
    avg_unit_sec: float = 0.0
    eta_sec: float = 0.0
    units: dict[str, UnitState] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def recompute_counts(self) -> dict:
        """由 ``units`` 现算各状态计数（避免计数漂移）。"""
        counts = {s: 0 for s in sorted(UNIT_STATUSES)}
        for us in self.units.values():
            st = us.status if us.status in UNIT_STATUSES else "pending"
            counts[st] = counts.get(st, 0) + 1
        self.counts = counts
        return counts

    def refresh_progress(self) -> dict:
        """重算计数并更新 ``avg_unit_sec`` / ``eta_sec``。"""
        counts = self.recompute_counts()
        processed = counts.get("done", 0) + counts.get("failed", 0)
        remaining = max(0, int(self.total) - processed - counts.get("skipped", 0))
        if processed > 0 and self.elapsed_sec > 0:
            self.avg_unit_sec = float(self.elapsed_sec) / float(processed)
        else:
            self.avg_unit_sec = 0.0
        self.eta_sec = float(self.avg_unit_sec) * float(remaining)
        return counts

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """转为普通 dict（符合 PRD §5.6 字段契约）。"""
        return {
            "schema_version": int(self.schema_version),
            "phase": str(self.phase),
            "process_state": str(self.process_state),
            "pid": int(self.pid),
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            "seed": int(self.seed),
            "backend": str(self.backend),
            "config_hash": str(self.config_hash),
            "total": int(self.total),
            "counts": dict(self.counts),
            "current_unit": str(self.current_unit),
            "started_at": str(self.started_at),
            "elapsed_sec": float(self.elapsed_sec),
            "avg_unit_sec": float(self.avg_unit_sec),
            "eta_sec": float(self.eta_sec),
            "units": {uid: us.to_dict() for uid, us in self.units.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScaleState":
        """从 dict 构造。"""
        units_raw = d.get("units", {}) or {}
        units: dict[str, UnitState] = {}
        if isinstance(units_raw, dict):
            for uid, us in units_raw.items():
                units[str(uid)] = UnitState.from_dict(us if isinstance(us, dict) else {})
        return cls(
            schema_version=int(d.get("schema_version", 1)),
            phase=str(d.get("phase", "train")),
            process_state=str(d.get("process_state", "idle")),
            pid=int(d.get("pid", 0) or 0),
            created_at=str(d.get("created_at", "") or ""),
            updated_at=str(d.get("updated_at", "") or ""),
            seed=int(d.get("seed", 42) or 42),
            backend=str(d.get("backend", "auto") or "auto"),
            config_hash=str(d.get("config_hash", "") or ""),
            total=int(d.get("total", 0) or 0),
            counts=dict(d.get("counts", {}) or {}),
            current_unit=str(d.get("current_unit", "") or ""),
            started_at=str(d.get("started_at", "") or ""),
            elapsed_sec=float(d.get("elapsed_sec", 0.0) or 0.0),
            avg_unit_sec=float(d.get("avg_unit_sec", 0.0) or 0.0),
            eta_sec=float(d.get("eta_sec", 0.0) or 0.0),
            units=units,
        )


class StateStore:
    """状态文件读写与崩溃恢复（原子写）。

    Args:
        paths: 路径集合。
    """

    def __init__(self, paths: ScalePaths) -> None:
        """初始化。"""
        self.paths: ScalePaths = paths

    # ------------------------------------------------------------------ #
    def load(self, phase: str) -> ScaleState:
        """载入阶段状态；文件不存在时返回全新的空状态。"""
        p = self.paths.state_of(phase)
        if not p.exists():
            state = ScaleState(phase=phase, created_at=_now(), updated_at=_now())
            return state
        try:
            data = read_json(p)
        except Exception:  # noqa: BLE001 - 损坏状态回退为空（由 recover 纠正）
            state = ScaleState(phase=phase, created_at=_now(), updated_at=_now())
            return state
        state = ScaleState.from_dict(data)
        state.phase = phase
        return state

    def save(self, state: ScaleState) -> None:
        """原子写出状态文件。"""
        state.updated_at = _now()
        if not state.created_at:
            state.created_at = _now()
        state.recompute_counts()
        atomic_write_json(state.to_dict(), self.paths.state_of(state.phase))

    # ------------------------------------------------------------------ #
    def recover(
        self, state: ScaleState, verify: Callable[[str], bool]
    ) -> dict:
        """崩溃恢复扫描：按产物有效性校正 ``running`` / ``done``。

        Args:
            state: 待校正状态（原地修改）。
            verify: ``unit_id -> bool``，返回产物是否有效（双重校验）。

        Returns:
            ``{"reset_to_pending": [...], "confirmed_done": [...], "reset_done": [...]}``。
        """
        reset_to_pending: list[str] = []
        confirmed_done: list[str] = []
        reset_done: list[str] = []

        for uid, us in state.units.items():
            if us.status == "running":
                # 崩溃悬挂：产物有效则视为完成，否则回滚 pending
                if verify(uid):
                    us.status = "done"
                    confirmed_done.append(uid)
                else:
                    us.status = "pending"
                    us.started_at = ""
                    us.finished_at = ""
                    us.duration_sec = 0.0
                    reset_to_pending.append(uid)
            elif us.status == "done":
                # 状态标记完成但产物缺失：回滚 pending
                if not verify(uid):
                    us.status = "pending"
                    reset_done.append(uid)
        state.recompute_counts()
        return {
            "reset_to_pending": reset_to_pending,
            "confirmed_done": confirmed_done,
            "reset_done": reset_done,
        }
