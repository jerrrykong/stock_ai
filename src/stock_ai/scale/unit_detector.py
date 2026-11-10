"""单元检测器（:class:`UnitRunner` 实现）。

流程：``选模型 -> 加载 assigned norm（绝不重新 fit）-> build_split -> Backtester
-> 写 report``。

关键约束：
- assigned model 的 :class:`NormalizerState` **缺失** → ``skipped`` +
  ``reason="normalizer_missing"``（**不得**静默 fallback 到重新 fit，Q2/§8.9）。
- 无匹配模型 → ``skipped`` + ``reason="no_model:<policy>"``。
- 首版仅支持 ``mode=backtest``；``predict/both`` 明确报「暂不支持」（U4）。
- 单元级异常不上抛：记 ``failed``，整批继续。
"""

from __future__ import annotations

import time
from typing import Any

from stock_ai.config.schema import AppConfig
from stock_ai.scale.model_policy import ModelSelector
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.state import UnitState, _now
from stock_ai.scale.units import SplitPlan, Unit

__all__ = ["UnitDetector"]


class UnitDetector:
    """对检测单元加载 assigned 模型并回测，产出报告。

    Args:
        cfg: 全局配置。
        paths: 路径集合。
        sampler: 单元采样器（复用 ``build_split``）。
        normalizers: 归一化状态存储（读取训练时状态）。
        artifacts: 产物校验器（枚举可用模型）。
        split: 划分计划（提供同代码训练组）。
        seed: 全局随机种子。
    """

    def __init__(
        self,
        cfg: AppConfig,
        paths: ScalePaths,
        sampler: Any,
        normalizers: Any,
        artifacts: Any,
        split: SplitPlan,
        seed: int = 42,
    ) -> None:
        """初始化。"""
        self.cfg: AppConfig = cfg
        self.paths: ScalePaths = paths
        self.sampler = sampler
        self.normalizers = normalizers
        self.artifacts = artifacts
        self.split: SplitPlan = split
        self.seed: int = int(seed)

    # ------------------------------------------------------------------ #
    def run_unit(
        self, unit: Unit, raw_df: Any, index: int, total: int
    ) -> UnitState:
        """检测单个单元并返回其状态（异常不上抛）。"""
        t0 = time.time()
        try:
            mode = str(self.cfg.scale.detect.mode or "backtest").lower()
            if mode != "backtest":
                # U4：predict/both 仅为预留分支，未实现时明确报「暂不支持」而非静默
                return UnitState(
                    status="skipped",
                    reason=f"mode_not_supported:{mode}（首版仅支持 backtest）",
                    seed=self.seed,
                    finished_at=_now(),
                )

            policy = str(self.cfg.scale.detect.model_policy or "nearest_past")
            available = self.artifacts.available_train_models()
            assigned = ModelSelector.select(
                policy, unit.symbol, unit.year, self.split.train_units, available
            )
            if not assigned:
                return UnitState(
                    status="skipped",
                    reason=f"no_model:{policy}",
                    seed=self.seed,
                    finished_at=_now(),
                )

            if not self.normalizers.exists(assigned):
                # Q2/§8.9：不得静默重新 fit
                return UnitState(
                    status="skipped",
                    reason="normalizer_missing",
                    assigned_model=assigned,
                    seed=self.seed,
                    finished_at=_now(),
                )

            # ---- 1) 加载【训练时】归一化状态（绝不重新 fit）----
            state = self.normalizers.load(assigned)

            # ---- 2) 用 assigned norm 构建检测样本 ----
            detect_split = self.sampler.builder.build_split(
                {unit.symbol: raw_df}, state, unit.start, unit.end
            )
            if int(detect_split.n_steps) < 2:
                return UnitState(
                    status="skipped",
                    reason=f"insufficient_data(n_steps={detect_split.n_steps})",
                    assigned_model=assigned,
                    seed=self.seed,
                    finished_at=_now(),
                )

            # ---- 3) 加载模型（复用既有按后端加载逻辑）----
            model = self._load_model(self.artifacts.model_zip(assigned))
            backend = self._backend_of(model)

            # ---- 4) 回测 ----
            from stock_ai.eval.backtester import Backtester

            bt = Backtester(model, self.cfg.env, detect_split, mode="discrete")
            result = bt.run(
                backend=backend,
                algo=str(self.cfg.scale.train.algo),
                model_path=str(self.artifacts.model_zip(assigned)),
                data_hash=str(getattr(state, "data_hash", "")),
                test_range=(unit.start, unit.end),
            )

            # ---- 5) 落盘报告 ----
            report_path = self.artifacts.report(unit.unit_id)
            result.to_json(report_path)
            self._save_equity_png(result, unit)

            metrics = {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in dict(result.metrics).items()
            }
            return UnitState(
                status="done",
                finished_at=_now(),
                duration_sec=time.time() - t0,
                report_path=str(report_path),
                assigned_model=assigned,
                metrics=metrics,
                seed=self.seed,
                data_hash=str(getattr(state, "data_hash", "")),
            )
        except Exception as exc:  # noqa: BLE001 - 单元级隔离
            return UnitState(
                status="failed",
                finished_at=_now(),
                duration_sec=time.time() - t0,
                seed=self.seed,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------ #
    def _save_equity_png(self, result: Any, unit: Unit) -> None:
        """绘制回测资金曲线（失败不阻塞）。"""
        try:
            from stock_ai.eval.plot import EquityPlotter

            EquityPlotter().plot(
                result, self.artifacts.report_png(unit.unit_id),
                title=f"detect {unit.unit_id}",
            )
        except Exception:  # noqa: BLE001 - 曲线非关键产物
            pass

    @staticmethod
    def _load_model(model_path: Any) -> Any:
        """按后端加载模型（复用既有 ``_load_model``）。"""
        from stock_ai.eval.backtest_cli import _load_model

        return _load_model(model_path)

    @staticmethod
    def _backend_of(model: Any) -> str:
        """推断模型后端名。"""
        name = getattr(model, "backend_name", "")
        if name:
            return str(name)
        return "numpy" if model.__class__.__name__.startswith("Numpy") else "sb3"
