"""共享编排骨架（``ScaleOrchestrator``）＋ ``UnitRunner`` 端口。

``ScaleOrchestrator`` 是**唯一**的「遍历单元 → 控制轮询 → 崩溃恢复 → 落状态 → 算 ETA」
骨架；``train`` / ``detect`` 通过注入 ``UnitRunner``（``UnitTrainer`` / ``UnitDetector``）
区分，避免两套状态机分叉（架构 §1.2）。

关键行为：
- **严格串行**：逐单元处理，``workers>1`` 仅预留（默认 1）。
- **单元边界控制轮询**：``control.json`` 仅在单元之间轮询。
- **中断即回滚**：``KeyboardInterrupt`` → 当前单元回滚 ``pending`` → 优雅退出（130）。
- **跳过且不覆盖**：``done`` 单元在重启后直接跳过。
- **单元级异常隔离**：记 ``failed``，整批继续。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from stock_ai.config.schema import AppConfig
from stock_ai.data.cache import DataCache
from stock_ai.scale.artifacts import UnitArtifacts
from stock_ai.scale.control import ControlStore
from stock_ai.scale.errors import ScaleError
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.state import ScaleState, StateStore, UnitState, _now
from stock_ai.scale.units import SplitPlan, Unit
from stock_ai.utils.io import read_json

__all__ = ["UnitRunner", "ScaleOrchestrator"]


@runtime_checkable
class UnitRunner(Protocol):
    """单元执行器端口（Ports & Adapters 的 Port）。

    实现者：:class:`~stock_ai.scale.unit_trainer.UnitTrainer`、
    :class:`~stock_ai.scale.unit_detector.UnitDetector`。

    契约：
        - 成功 → ``UnitState(status="done", ...)``；
        - 缺前置 → ``UnitState(status="skipped", reason=...)``；
        - 异常 → **不抛出**，返回 ``UnitState(status="failed", error=...)``。
    """

    def run_unit(
        self, unit: Unit, raw_df: Any, index: int, total: int
    ) -> UnitState:
        """执行单个单元。"""
        ...


class ScaleOrchestrator:
    """大规模训练/检测的共享编排骨架。

    Args:
        cfg: 全局配置。
        phase: ``train|detect``。
        runner: 单元执行器。
        paths: 路径集合。
        logger: 日志器。
        seed: 全局随机种子。
        only_failed: 仅重跑 ``failed`` 单元（``scale train --only-failed`）。
    """

    def __init__(
        self,
        cfg: AppConfig,
        phase: str,
        runner: UnitRunner,
        paths: ScalePaths,
        logger: Any,
        seed: int = 42,
        only_failed: bool = False,
    ) -> None:
        """初始化。"""
        self.cfg: AppConfig = cfg
        self.phase: str = str(phase)
        self.runner: UnitRunner = runner
        self.paths: ScalePaths = paths
        self.logger = logger
        self.seed: int = int(seed)
        self.only_failed: bool = bool(only_failed)

        self.store: StateStore = StateStore(paths)
        self.control: ControlStore = ControlStore(paths)
        self.artifacts: UnitArtifacts = UnitArtifacts(paths)
        self.poll_sec: float = float(getattr(cfg.scale, "poll_sec", 2.0) or 2.0)

    # ------------------------------------------------------------------ #
    def run(self) -> int:
        """执行整个阶段，返回退出码（``0`` 正常，``130`` 被中断）。"""
        if int(getattr(self.cfg.scale, "workers", 1) or 1) > 1:
            # U6：workers>1 仅预留，明确报「暂不支持」而非静默
            raise ScaleError(
                "scale.workers>1 暂不支持（首版严格串行，Windows 稳定优先）",
                context={"workers": self.cfg.scale.workers},
            )

        plan = self._load_plan()
        units = plan.stage_units(self.phase)
        cache = DataCache(Path(self.cfg.data.cache_dir))

        state = self.store.load(self.phase)
        self._sync_state(state, units, plan)

        info = self.store.recover(
            state, lambda uid: self.artifacts.is_done(self.phase, uid)
        )
        if any(info.values()):
            self.logger.info(
                "崩溃恢复扫描: running->pending=%d, running->done=%d, done->pending=%d",
                len(info["reset_to_pending"]), len(info["confirmed_done"]),
                len(info["reset_done"]),
            )

        state.process_state = "running"
        state.pid = os.getpid()
        if not state.started_at:
            state.started_at = _now()
        self.store.save(state)

        started = time.time()
        interrupted = False
        try:
            self._loop(state, units, cache, started)
        except KeyboardInterrupt:
            interrupted = True
            self.logger.warning("收到中断（Ctrl+C），当前单元已回滚 pending，优雅退出。")
        finally:
            state.elapsed_sec = time.time() - started
            state.current_unit = ""
            state.refresh_progress()
            if interrupted:
                state.process_state = "stopped"
            elif state.process_state not in ("stopped",):
                state.process_state = "finished"
            self.store.save(state)
            c = state.counts
            self.logger.info(
                "[%s] 结束 process_state=%s total=%d done=%d failed=%d skipped=%d pending=%d",
                self.phase, state.process_state, state.total,
                c.get("done", 0), c.get("failed", 0), c.get("skipped", 0), c.get("pending", 0),
            )
        return 130 if interrupted else 0

    # ------------------------------------------------------------------ #
    def _load_plan(self) -> SplitPlan:
        """载入 ``split.json``。

        Raises:
            ScaleError: 计划文件缺失。
        """
        p = self.paths.split_json
        if not p.exists():
            raise ScaleError(
                "缺少划分计划 split.json：请先执行 `python -m stock_ai scale units`",
                context={"path": str(p)},
            )
        return SplitPlan.from_dict(read_json(p))

    def _sync_state(
        self, state: ScaleState, units: list[str], plan: SplitPlan
    ) -> None:
        """对齐状态与之计划：保留既有单元状态，剔除已移除单元，补入新单元。"""
        existing = dict(state.units)
        new_units: dict[str, UnitState] = {}
        for uid in units:
            new_units[uid] = existing.get(uid, UnitState(status="pending", seed=self.seed))
        state.units = new_units
        state.phase = self.phase
        state.total = len(units)
        state.seed = self.seed
        state.backend = str(self.cfg.scale.train.backend)
        state.config_hash = str(plan.config_hash)

    # ------------------------------------------------------------------ #
    def _loop(
        self, state: ScaleState, units: list[str], cache: DataCache, started: float
    ) -> None:
        """逐单元主循环（严格串行 + 单元边界控制轮询）。"""
        total = len(units)
        raw_memo: dict[str, Any] = {}
        for index, uid in enumerate(units):
            action = self._handle_control(state)
            if action == "stop":
                state.process_state = "stopped"
                self.control.clear()
                self.logger.info("收到 stop，停止于单元边界。")
                break

            us = state.units.get(uid)
            if us is None:
                continue

            if self.only_failed:
                if us.status != "failed":
                    continue
            elif us.status != "pending":
                # done / failed / skipped 一律跳过（done 绝不覆盖）
                continue

            unit = Unit.parse(uid)
            raw = self._get_raw(cache, unit.symbol, raw_memo)
            if raw is None:
                us.status = "skipped"
                us.reason = "data_missing"
                us.finished_at = _now()
                self.store.save(state)
                self.logger.warning("[%d/%d] %s 跳过：数据缺失", index + 1, total, uid)
                continue

            us.status = "running"
            us.started_at = _now()
            us.error = ""
            us.reason = ""
            state.current_unit = uid
            self.store.save(state)

            t0 = time.time()
            try:
                res = self.runner.run_unit(unit, raw, index, total)
            except KeyboardInterrupt:
                us.status = "pending"
                us.started_at = ""
                self.store.save(state)
                raise
            except Exception as exc:  # noqa: BLE001 - 兜底隔离（runner 约定不上抛）
                res = UnitState(
                    status="failed",
                    finished_at=_now(),
                    duration_sec=time.time() - t0,
                    seed=self.seed,
                    error=f"{type(exc).__name__}: {exc}",
                )

            self._apply(state, uid, res, t0)
            state.elapsed_sec = time.time() - started
            state.refresh_progress()
            self.store.save(state)

            c = state.counts
            eta = state.eta_sec
            self.logger.info(
                "[%d/%d] %s -> %s (%.2fs) done=%d failed=%d skipped=%d pending=%d eta=%.0fs",
                index + 1, total, uid, us.status, us.duration_sec,
                c.get("done", 0), c.get("failed", 0), c.get("skipped", 0),
                c.get("pending", 0), eta,
            )

    # ------------------------------------------------------------------ #
    def _handle_control(self, state: ScaleState) -> str:
        """在单元边界轮询控制信号；返回 ``""|pause-handled|resume|stop``。"""
        cmd = self.control.poll(self.phase)
        if cmd == "pause":
            state.process_state = "paused"
            self.store.save(state)
            self.logger.info("已暂停（单元边界），等待 resume/stop ...")
            while True:
                time.sleep(max(0.05, self.poll_sec))
                c = self.control.poll(self.phase)
                if c == "resume":
                    self.control.clear()
                    state.process_state = "running"
                    self.store.save(state)
                    self.logger.info("收到 resume，继续。")
                    return "resume"
                if c == "stop":
                    return "stop"
        if cmd == "stop":
            return "stop"
        if cmd == "resume":
            self.control.clear()
            return "resume"
        return ""

    def _get_raw(
        self, cache: DataCache, symbol: str, memo: dict[str, Any]
    ) -> Any:
        """读取并记忆化某 symbol 的原始日线（缺失返回 None）。"""
        if symbol in memo:
            return memo[symbol]
        raw = None
        try:
            if cache.exists(symbol):
                raw = cache.read(symbol)
        except Exception as exc:  # noqa: BLE001 - 单只坏缓存不阻断
            self.logger.warning("读取缓存失败 %s: %s", symbol, exc)
            raw = None
        memo[symbol] = raw
        return raw

    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply(
        state: ScaleState, uid: str, res: UnitState, t0: float
    ) -> None:
        """把执行器返回的 :class:`UnitState` 合并进状态（保留已记录字段）。"""
        us = state.units[uid]
        us.status = res.status
        us.finished_at = res.finished_at or _now()
        us.duration_sec = float(res.duration_sec) if res.duration_sec else (time.time() - t0)
        if res.model_path:
            us.model_path = res.model_path
        if res.assigned_model:
            us.assigned_model = res.assigned_model
        if res.report_path:
            us.report_path = res.report_path
        if res.metrics is not None:
            us.metrics = res.metrics
        if res.seed:
            us.seed = res.seed
        if res.data_hash:
            us.data_hash = res.data_hash
        us.error = res.error
        us.reason = res.reason
