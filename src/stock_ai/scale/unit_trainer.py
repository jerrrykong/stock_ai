"""单元训练器（:class:`UnitRunner` 实现）。

流程：``fit(单元窗口) -> 持久化 norm -> TradingEnv -> TrainerFactory -> learn
-> 存 zip / sidecar / curve``。

设计要点：
- **单元级异常不上抛**：捕获后返回 ``failed`` 状态，整批继续（单元级隔离）。
- **归一化持久化**：训练时把自拟合的 :class:`NormalizerState` 落盘，检测侧复用（Q2）。
- 复用既有栈：``TradingEnv`` / ``TrainerFactory`` / ``EquityPlotter`` / ``atomic_write_json``。
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from stock_ai.config.schema import AppConfig, TrainConfig
from stock_ai.scale.atomic import atomic_write_json
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.state import UnitState, _now
from stock_ai.scale.units import Unit
from stock_ai.utils.io import timestamp

__all__ = ["UnitTrainer"]


class UnitTrainer:
    """把单个「（代码 × 年份）」单元训练成一个独立模型。

    Args:
        cfg: 全局配置。
        paths: 路径集合。
        sampler: 单元采样器。
        normalizers: 归一化状态存储。
        artifacts: 产物校验器。
        seed: 全局随机种子。
    """

    def __init__(
        self,
        cfg: AppConfig,
        paths: ScalePaths,
        sampler: Any,
        normalizers: Any,
        artifacts: Any,
        seed: int = 42,
    ) -> None:
        """初始化。"""
        self.cfg: AppConfig = cfg
        self.paths: ScalePaths = paths
        self.sampler = sampler
        self.normalizers = normalizers
        self.artifacts = artifacts
        self.seed: int = int(seed)

    # ------------------------------------------------------------------ #
    def run_unit(
        self, unit: Unit, raw_df: Any, index: int, total: int
    ) -> UnitState:
        """训练单个单元并返回其状态（异常不上抛）。"""
        t0 = time.time()
        try:
            min_steps = int(self.cfg.sample.window) + 2
            sample = self.sampler.sample(unit, raw_df, self.cfg.sample.val_ratio)
            if int(sample.n_steps) < min_steps:
                return UnitState(
                    status="skipped",
                    reason=f"insufficient_data(n_steps={sample.n_steps})",
                    seed=self.seed,
                    duration_sec=time.time() - t0,
                    finished_at=_now(),
                )

            # ---- 1) 持久化单元归一化状态（检测侧复用，Q2）----
            self.normalizers.save(unit.unit_id, sample.state)

            # ---- 2) 构建环境 + 训练器（复用既有栈）----
            train_cfg = self._train_config()
            env = self._make_env(sample.train)
            trainer = self._make_trainer(train_cfg, env)

            # ---- 3) 训练 ----
            trainer.learn(int(train_cfg.timesteps), seed=self.seed)

            # ---- 4) 保存模型 ----
            model_path = self.artifacts.model_zip(unit.unit_id)
            trainer.save(model_path)

            # ---- 5) 训练曲线（非阻塞）----
            self._save_curve(trainer, unit, total)

            # ---- 6) sidecar ----
            duration = time.time() - t0
            backend_used = self._backend_used(trainer)
            self._write_sidecar(
                unit, train_cfg, sample, backend_used, duration, model_path
            )

            return UnitState(
                status="done",
                started_at="",
                finished_at=_now(),
                duration_sec=duration,
                model_path=str(model_path),
                seed=self.seed,
                data_hash=sample.data_hash,
            )
        except Exception as exc:  # noqa: BLE001 - 单元级隔离：记 failed，整批继续
            return UnitState(
                status="failed",
                finished_at=_now(),
                duration_sec=time.time() - t0,
                seed=self.seed,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------ #
    def _train_config(self) -> TrainConfig:
        """由全局 ``train`` 段 + ``scale.train`` 覆盖派生单元训练配置。"""
        sc = self.cfg.scale.train
        return dataclasses.replace(
            self.cfg.train,
            algo=str(sc.algo),
            timesteps=int(sc.timesteps),
            backend=str(sc.backend),
        )

    def _make_env(self, split: Any) -> Any:
        """构造 :class:`TradingEnv`（复用既有环境）。"""
        from stock_ai.env.trading_env import TradingEnv

        return TradingEnv(split, self.cfg.env, seed=self.seed, action_mode="discrete")

    def _make_trainer(self, train_cfg: TrainConfig, env: Any) -> Any:
        """构造训练器（后端 ``auto`` 优先 SB3，不可用自动回退 NumPy）。"""
        from stock_ai.train.factory import TrainerFactory

        return TrainerFactory.create(train_cfg, env, preference=str(train_cfg.backend))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _backend_used(trainer: Any) -> str:
        """推断实际使用的后端名。"""
        name = getattr(trainer, "backend_name", "")
        if name:
            return str(name)
        cls = trainer.__class__.__name__.lower()
        if "sb3" in cls:
            return "sb3"
        return "numpy"

    def _save_curve(self, trainer: Any, unit: Unit, total: int) -> None:
        """绘制并保存训练曲线（失败不阻塞主流程）。"""
        try:
            from stock_ai.eval.plot import EquityPlotter

            series: dict[str, list[float]] = {}
            rewards = list(getattr(trainer, "reward_history", []) or [])
            losses = list(getattr(trainer, "loss_history", []) or [])
            if rewards:
                series["mean_reward"] = [float(x) for x in rewards]
            if losses:
                series["loss"] = [float(x) for x in losses]
            EquityPlotter().plot_training_curve(
                series, self.artifacts.curve(unit.unit_id), title=f"train {unit.unit_id}"
            )
        except Exception:  # noqa: BLE001 - 曲线非关键产物
            pass

    def _write_sidecar(
        self,
        unit: Unit,
        train_cfg: TrainConfig,
        sample: Any,
        backend: str,
        duration: float,
        model_path: Any,
    ) -> None:
        """原子写出单元 sidecar（PRD §5.2 字段）。"""
        payload = {
            "unit_id": unit.unit_id,
            "symbol": unit.symbol,
            "year": int(unit.year),
            "algo": str(train_cfg.algo),
            "backend": str(backend),
            "seed": int(self.seed),
            "timesteps": int(train_cfg.timesteps),
            "data_hash": str(sample.data_hash),
            "sample_steps": int(sample.n_steps),
            "duration_sec": float(duration),
            "created_at": timestamp(),
            "model_path": str(model_path),
        }
        atomic_write_json(payload, self.artifacts.sidecar(unit.unit_id))
