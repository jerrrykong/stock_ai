"""SB3 训练器封装（Primary 后端）。

**本文件与 ``train/policy.py`` 是全项目唯一允许 import torch / stable_baselines3 的地方**
（架构 §3 HARD RULES）。所有 import 均为延迟导入，便于在无 SB3 环境下降级。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stock_ai.config.schema import TrainConfig
from stock_ai.train.base import TrainResult
from stock_ai.utils.errors import ArtifactError, ConfigError
from stock_ai.utils.logging import LoggerFactory

__all__ = ["SB3Trainer", "ContinuousActionWrapper", "make_unified_trainer", "learn_unified"]


def make_unified_trainer(episodes, cfg, n_envs, total, checkpoint=None, strict=False,
                         strict_episodes=None, execution_version="strict-t1-open-v1"):
    """统一模式显式要求 SB3；一个 PPO 在多个股票年度 episode 间共享参数。

    Args:
        episodes: 旧语义（``Episode``）episode 列表。
        cfg: 应用配置。
        n_envs: 并行环境数。
        total: 目标总步数。
        checkpoint: 续训 checkpoint 路径。
        strict: 是否使用严格撮合（T+1 次日开盘成交）。
        strict_episodes: 严格撮合用的 ``(id, obs, bars)`` 三元组列表。
    """
    from dataclasses import replace
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stock_ai.scale.unified import make_pool_env

    env = DummyVecEnv([lambda i=i: make_pool_env(episodes, cfg.env, cfg.random_seed + i,
                                                 strict=strict, strict_episodes=strict_episodes,
                                                 execution_version=execution_version)
                       for i in range(n_envs)])
    try:
        if checkpoint is not None:
            trainer = SB3Trainer.load(checkpoint, env=env)
            trainer.seed = cfg.random_seed
            return trainer
        steps = max(2, min(int(cfg.train.n_steps), int(total)))
        batch = max(2, min(int(cfg.train.batch_size), steps * n_envs))
        train_cfg = replace(cfg.train, algo="ppo", backend="sb3", n_steps=steps, batch_size=batch)
        return SB3Trainer(env, train_cfg, algo="ppo", seed=cfg.random_seed)
    except Exception:
        env.close()
        raise


def learn_unified(trainer, remaining, tick):
    from stable_baselines3.common.callbacks import BaseCallback

    class UnifiedCallback(BaseCallback):
        def _on_step(self):
            return tick(self.model)

    trainer.model.learn(total_timesteps=remaining, reset_num_timesteps=False,
                        callback=UnifiedCallback(), progress_bar=False)

logger = LoggerFactory.get(__name__)


class ContinuousActionWrapper:
    """把 TC 连续目标仓位 ``[0,1]`` 映射为 A 股整手买卖动作。

    仅在 ``--algo sac`` 且被 SB3 使用时构造；纯逻辑无 torch 依赖。
    """

    def __init__(self, lot_size: int = 100, band: float = 0.05) -> None:
        """初始化。

        Args:
            lot_size: 每手股数。
            band: 触发交易的最小仓位变化阈值。
        """
        self.lot_size: int = int(lot_size)
        self.band: float = float(band)

    def to_discrete(self, target_position: float, current_position: float) -> int:
        """把目标仓位差映射为离散动作 0/1/2。

        Args:
            target_position: 目标仓位 ``[0,1]``。
            current_position: 当前仓位 ``[0,1]``。

        Returns:
            0=持有, 1=买入, 2=卖出。
        """
        diff = float(target_position) - float(current_position)
        if diff > self.band:
            return 1
        if diff < -self.band:
            return 2
        return 0


class SB3Trainer:
    """Stable-Baselines3 封装（PPO / SAC / DQN）。

    Args:
        env: Gymnasium 环境。
        cfg: 训练配置。
        algo: ``ppo`` / ``sac`` / ``dqn``。
        seed: 随机种子。
    """

    SUPPORTED: dict[str, str] = {"ppo": "PPO", "sac": "SAC", "dqn": "DQN"}
    backend_name: str = "sb3"

    def __init__(self, env: "Any", cfg: TrainConfig, algo: str = "ppo", seed: int = 42) -> None:
        """初始化并构造 SB3 模型。"""
        self.env = env
        self.cfg = cfg
        self.algo = str(algo).lower()
        self.seed = int(seed)
        self.api_switch = ContinuousActionWrapper(
            lot_size=getattr(getattr(env, "cfg", None), "lot_size", 100)
        )
        self.model = self._build_model()
        self.algo_name: str = f"sb3_{self.algo}"

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def _is_continuous_action_space(self) -> bool:
        """判断当前环境的动作空间是否为连续 Box（``Box`` 且为浮点）。"""
        space = getattr(self.env, "action_space", None)
        if space is None:
            return False
        # gymnasium.spaces.Box -> 连续；Discrete -> 离散
        return bool(getattr(space, "shape", None) is not None
                    and str(type(space).__name__) == "Box")

    def _build_model(self) -> "Any":
        """按算法构造 SB3 模型（延迟导入）。"""
        if self.algo not in self.SUPPORTED:
            raise ConfigError(
                f"SB3 不支持算法: {self.algo!r}（支持 {sorted(self.SUPPORTED)}）",
                context={"algo": self.algo, "valid": sorted(self.SUPPORTED)},
            )

        # SAC 需要连续动作空间：提前给出领域化提示，避免 SB3 内部断言冒到顶层。
        if self.algo == "sac" and not self._is_continuous_action_space():
            raise ConfigError(
                "SAC 需要连续动作空间，请使用 ppo 或启用 continuous 动作模式",
                context={"algo": "sac", "action_space": str(self.env.action_space)},
            )

        # 限制 torch 线程（Windows 防线程爆炸）
        try:
            import torch

            torch.set_num_threads(int(self.cfg.torch_threads))
        except Exception:  # pragma: no cover
            pass

        from stable_baselines3 import DQN, PPO, SAC

        cfg = self.cfg
        common = {
            "learning_rate": float(cfg.learning_rate),
            "gamma": float(cfg.gamma),
            "seed": int(self.seed),
            "verbose": 0,
            "device": "cpu",
        }
        cls = {"ppo": PPO, "sac": SAC, "dqn": DQN}[self.algo]
        if self.algo == "ppo":
            return cls(
                "MlpPolicy",
                self.env,
                n_steps=int(cfg.n_steps),
                batch_size=int(cfg.batch_size),
                n_epochs=int(cfg.n_epochs),
                ent_coef=float(cfg.ent_coef),
                clip_range=float(cfg.clip_range),
                gae_lambda=float(cfg.gae_lambda),
                **common,
            )
        if self.algo == "sac":
            return cls("MlpPolicy", self.env, batch_size=int(cfg.batch_size), **common)
        # dqn
        return cls(
            "MlpPolicy",
            self.env,
            batch_size=int(cfg.batch_size),
            learning_starts=min(1000, int(cfg.n_steps)),
            **common,
        )

    # ------------------------------------------------------------------ #
    def learn(self, total_timesteps: int, seed: int | None = None) -> None:
        """训练。

        Args:
            total_timesteps: 目标步数。
            seed: 覆盖随机种子。
        """
        if seed is not None:
            self.seed = int(seed)
        self.model.learn(total_timesteps=int(total_timesteps), progress_bar=False)

    def predict(
        self, obs: "np.ndarray", deterministic: bool = True
    ) -> tuple["np.ndarray", Any]:
        """预测动作。

        Args:
            obs: 观测。
            deterministic: 是否贪心。

        Returns:
            ``(action, None)``。
        """
        import numpy as np

        arr = np.asarray(obs)
        action, _state = self.model.predict(arr, deterministic=deterministic)
        return np.asarray(action), None

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        """保存 SB3 原生 zip。

        Args:
            path: 目标 ``.zip`` 路径。
        """
        p = Path(path)
        if p.suffix.lower() != ".zip":
            p = p.with_suffix(".zip")
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p))
        logger.info("SB3 模型已保存: %s", p)

    @classmethod
    def load(cls, path: Path, env: "Any" = None) -> "SB3Trainer":
        """加载 SB3 模型。

        Args:
            path: 模型 zip 路径。
            env: 环境（可选）。

        Returns:
            恢复后的 :class:`SB3Trainer`。

        Raises:
            ArtifactError: 文件缺失或算法不支持。
        """
        p = Path(path)
        if p.suffix.lower() != ".zip":
            p = p.with_suffix(".zip")
        if not p.exists():
            raise ArtifactError(
                f"模型文件不存在: {p}（请先执行 `stock_ai train`）", context={"path": str(p)}
            )

        from stable_baselines3 import DQN, PPO, SAC

        # 从文件名推断算法（<algo>_<ts>.zip）
        stem = p.stem.lower()
        algo = "ppo"
        for key in ("sac", "dqn", "ppo"):
            if stem.startswith(key):
                algo = key
                break
        cls_model = {"ppo": PPO, "sac": SAC, "dqn": DQN}[algo]

        obj = cls.__new__(cls)
        obj.env = env
        obj.cfg = TrainConfig(algo=algo)
        obj.algo = algo
        obj.seed = 42
        obj.algo_name = f"sb3_{algo}"
        obj.api_switch = ContinuousActionWrapper(
            lot_size=getattr(getattr(env, "cfg", None), "lot_size", 100)
        )
        obj.model = cls_model.load(str(p), env=env, device="cpu")
        return obj

    # ------------------------------------------------------------------ #
    def build(self, timesteps: int) -> "SB3Trainer":
        """链式兼容：暂存训练步数。"""
        self._pending_timesteps = int(timesteps)
        return self

    def train(self, timesteps: int | None = None) -> TrainResult:
        """执行训练并返回结果摘要。"""
        import time

        steps = int(timesteps or getattr(self, "_pending_timesteps", self.cfg.timesteps))
        t0 = time.time()
        self.learn(steps)
        elapsed = time.time() - t0
        return TrainResult(
            run_dir=Path("."),
            model_path=Path("."),
            algo=self.algo,
            backend="sb3",
            total_timesteps=steps,
            best_mean_reward=0.0,
            elapsed_sec=float(elapsed),
        )

    @property
    def loss_history(self) -> list[float]:
        """（SB3 不暴露逐轮损失，返回空列表）。"""
        return []

    @property
    def reward_history(self) -> list[float]:
        """（SB3 不暴露逐轮奖励，返回空列表）。"""
        return []
