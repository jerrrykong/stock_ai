"""后端工厂：探测 SB3/torch 可用性并选择训练器（K —— 架构 §4.5）。

``auto`` 优先 SB3；不可用时自动回退 :class:`NumpyPPOTrainer`。
"""

from __future__ import annotations

import importlib.util
from typing import Any

from stock_ai.config.schema import TrainConfig
from stock_ai.train.base import Enrollment
from stock_ai.utils.errors import ArtifactError, ConfigError
from stock_ai.utils.logging import LoggerFactory

__all__ = ["TrainerFactory", "SUPPORTED_ALGOS", "SUPPORTED_BACKENDS"]

logger = LoggerFactory.get(__name__)

#: 支持的算法白名单（与 SB3Trainer.SUPPORTED 保持一致）
SUPPORTED_ALGOS: frozenset[str] = frozenset({"ppo", "sac", "dqn"})

#: 支持的后端白名单（CLI ``--backend`` 与 ``train.backend`` 通用）
SUPPORTED_BACKENDS: frozenset[str] = frozenset({"auto", "sb3", "numpy"})


class TrainerFactory:
    """训练器工厂（依赖倒置：返回 :class:`Enrollment` 协议的实现）。"""

    @staticmethod
    def backend_available(name: str) -> bool:
        """探测后端是否可用（不抛异常）。

        Args:
            name: ``"sb3"`` / ``"stable_baselines3"`` / ``"torch"`` / ``"numpy"``。

        Returns:
            可用返回 ``True``。
        """
        key = str(name).lower()
        if key == "numpy":
            return True  # numpy 是硬依赖
        spec_name = {
            "sb3": "stable_baselines3",
            "stable_baselines3": "stable_baselines3",
            "torch": "torch",
            "gymnasium": "gymnasium",
        }.get(key, key)

        if key in ("sb3", "stable_baselines3"):
            # SB3 依赖 torch，必须两者都在
            if importlib.util.find_spec("torch") is None:
                return False
            if importlib.util.find_spec("stable_baselines3") is None:
                return False
            return True
        try:
            return importlib.util.find_spec(spec_name) is not None
        except Exception:  # pragma: no cover
            return False

    # ------------------------------------------------------------------ #
    @staticmethod
    def create(
        cfg: TrainConfig,
        env: "Any",
        preference: str = "auto",
    ) -> Enrollment:
        """创建训练器实例。

        Args:
            cfg: 训练配置。
            env: Gymnasium 环境。
            preference: ``auto`` / ``sb3`` / ``numpy``。

        Returns:
            :class:`Enrollment` 实现（:class:`SB3Trainer` 或 :class:`NumpyPPOTrainer`）。

        Raises:
            ArtifactError: ``preference='sb3'`` 但 SB3 不可用。
            ConfigError: ``cfg.algo`` 不在支持白名单内，或 SAC 用于离散动作空间。
        """
        pref_raw = preference or cfg.backend or "auto"
        pref = str(pref_raw).lower()
        algo = str(cfg.algo or "ppo").lower()

        # ---- 参数前置校验（早于任何后端选择/回退）----
        # 非法 algo / backend 必须在这里明确报错，绝不能被 auto 回退链路静默吞掉。
        if algo not in SUPPORTED_ALGOS:
            raise ConfigError(
                f"不支持的算法: {cfg.algo!r}（可选: {sorted(SUPPORTED_ALGOS)}）",
                context={"algo": cfg.algo, "valid": sorted(SUPPORTED_ALGOS)},
            )
        if pref not in SUPPORTED_BACKENDS:
            raise ConfigError(
                f"不支持的后端: {pref_raw!r}（可选: {sorted(SUPPORTED_BACKENDS)}）",
                context={"backend": pref_raw, "valid": sorted(SUPPORTED_BACKENDS)},
            )

        # 强制 numpy
        if pref == "numpy":
            logger.info("后端: 纯 NumPy PPO（用户强制）")
            return TrainerFactory._make_numpy(env, cfg)

        # 强制 sb3
        if pref == "sb3":
            if not TrainerFactory.backend_available("sb3"):
                raise ArtifactError(
                    "指定 backend=sb3 但 stable_baselines3/torch 不可用；"
                    "请改用 --backend numpy 或安装依赖"
                )
            logger.info("后端: SB3 (%s)", algo)
            return TrainerFactory._make_sb3(env, cfg, algo)

        # auto
        if TrainerFactory.backend_available("sb3"):
            try:
                logger.info("后端: SB3 (%s) [auto]", algo)
                return TrainerFactory._make_sb3(env, cfg, algo)
            except ConfigError:
                # 参数/配置错误必须原样上抛，绝不能当作"后端不可用"而静默回退。
                raise
            except (ImportError, ModuleNotFoundError, RuntimeError, OSError,
                    ArtifactError) as exc:
                # 仅后端不可用类异常触发回退（缺依赖 / 动态库 / 运行期环境问题）。
                logger.warning("SB3 不可用，回退 NumPy PPO: %s", exc)
                return TrainerFactory._make_numpy(env, cfg)

        logger.info("后端: 纯 NumPy PPO [auto]（SB3 不可用）")
        return TrainerFactory._make_numpy(env, cfg)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_numpy(env: "Any", cfg: TrainConfig) -> "Any":
        """构造 NumPy PPO 训练器（本地 import，避免循环依赖）。"""
        from stock_ai.train.ppo_numpy import NumpyPPOTrainer

        return NumpyPPOTrainer(env, cfg, seed=42)

    @staticmethod
    def _make_sb3(env: "Any", cfg: TrainConfig, algo: str) -> "Any":
        """构造 SB3 训练器（延迟 import）。"""
        from stock_ai.train.sb3_trainer import SB3Trainer

        return SB3Trainer(env, cfg, algo=algo, seed=42)

    # ------------------------------------------------------------------ #
    @staticmethod
    def available_backends() -> dict[str, bool]:
        """返回各后端可用性快照（供 ``doctor`` 使用）。"""
        return {
            "numpy": True,
            "gymnasium": TrainerFactory.backend_available("gymnasium"),
            "torch": TrainerFactory.backend_available("torch"),
            "sb3": TrainerFactory.backend_available("sb3"),
        }
