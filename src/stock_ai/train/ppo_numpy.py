"""纯 NumPy PPO 兜底训练器（仅依赖 numpy）。

这是保证全链路可跑通的**关键组件**：当 SB3 / torch 无法安装时自动接管。
实现要点：
    - rollout(n_steps) 收集轨迹 -> GAE(γ, λ) 计算优势与回报
    - 多 epoch 的 clipped surrogate objective + value loss + entropy bonus
    - 手写 Adam 优化器（见 :class:`~stock_ai.train.policy.AdamOptimizer`）
    - 离散 softmax 策略头，隐藏层 ``[64, 64]``，``tanh`` 激活
    - 与 SB3 共享同一 ``.zip`` 模型契约（内部为 ``weights.npz`` + ``hparams.json``）
    - 固定 seed 全流程可复现

模型 zip 内部结构::

    ppo_numpy.zip
    ├── weights.npz       # 策略权重
    ├── hparams.json      # 超参 + 元信息
    └── backend.json      # {"backend": "numpy", ...}
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from stock_ai.config.schema import TrainConfig
from stock_ai.train.base import TrainResult
from stock_ai.train.policy import AdamOptimizer, NumpyActorCritic
from stock_ai.utils.errors import ArtifactError
from stock_ai.utils.logging import LoggerFactory

__all__ = ["NumpyPPOTrainer", "RolloutBuffer"]

logger = LoggerFactory.get(__name__)


class RolloutBuffer:
    """定长轨迹缓冲区（on-policy）。

    Args:
        n_steps: 缓冲区长度。
        obs_dim: 展平观测维度。
    """

    def __init__(self, n_steps: int, obs_dim: int) -> None:
        """分配缓冲区。"""
        n = int(max(1, n_steps))
        self.n_steps: int = n
        self.obs_dim: int = int(obs_dim)
        self.obs = np.zeros((n, obs_dim), dtype="float64")
        self.actions = np.zeros(n, dtype="int64")
        self.logps = np.zeros(n, dtype="float64")
        self.rewards = np.zeros(n, dtype="float64")
        self.values = np.zeros(n, dtype="float64")
        self.dones = np.zeros(n, dtype="float64")
        self.ptr: int = 0

    def add(
        self,
        obs: "np.ndarray",
        action: int,
        logp: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """追加一步。"""
        i = self.ptr
        self.obs[i] = np.asarray(obs, dtype="float64").reshape(-1)
        self.actions[i] = int(action)
        self.logps[i] = float(logp)
        self.rewards[i] = float(reward)
        self.values[i] = float(value)
        self.dones[i] = 1.0 if done else 0.0
        self.ptr += 1

    @property
    def full(self) -> bool:
        """缓冲区是否写满。"""
        return self.ptr >= self.n_steps

    def reset(self) -> None:
        """清空指针与数据。"""
        self.ptr = 0
        self.rewards[:] = 0.0
        self.dones[:] = 0.0


def compute_gae(
    rewards: "np.ndarray",
    values: "np.ndarray",
    dones: "np.ndarray",
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    """广义优势估计（GAE）。

    Args:
        rewards: ``(T,)`` 奖励。
        values: ``(T,)`` 状态价值。
        dones: ``(T,)`` episode 终止标记（1.0 表示终止）。
        last_value: ``V(s_{T})`` 引导值。
        gamma: 折扣因子。
        gae_lambda: GAE λ。

    Returns:
        ``(advantages (T,), returns (T,))``。
    """
    t = len(rewards)
    advantages = np.zeros(t, dtype="float64")
    last_gae = 0.0
    next_value = float(last_value)
    for i in reversed(range(t)):
        nonterminal = 1.0 - float(dones[i])
        delta = float(rewards[i]) + gamma * next_value * nonterminal - float(values[i])
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[i] = last_gae
        next_value = float(values[i])
    returns = advantages + np.asarray(values, dtype="float64")
    return advantages, returns


class NumpyPPOTrainer:
    """纯 NumPy PPO 训练器（兜底后端）。

    Args:
        env: :class:`~stock_ai.env.trading_env.TradingEnv` 实例。
        cfg: 训练配置。
        seed: 随机种子。
    """

    algo_name: str = "ppo_numpy"
    backend_name: str = "numpy"

    def __init__(self, env: "Any", cfg: TrainConfig, seed: int = 42) -> None:
        """初始化训练器与策略网络。"""
        self.env = env
        self.cfg: TrainConfig = cfg
        self.seed: int = int(seed)

        obs_space = env.observation_space
        self.obs_shape: tuple[int, ...] = tuple(obs_space.shape)
        self.obs_dim: int = int(np.prod(self.obs_shape))
        self.n_actions: int = int(getattr(env.action_space, "n", 2))

        self.policy: NumpyActorCritic = NumpyActorCritic(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            hidden=(64, 64),
            seed=self.seed,
        )
        self.optimizer: AdamOptimizer = AdamOptimizer(
            self.policy.get_params(), lr=float(cfg.learning_rate)
        )
        self._loss_history: list[float] = []
        self._reward_history: list[float] = []

    # ------------------------------------------------------------------ #
    def _flatten(self, obs: "np.ndarray") -> "np.ndarray":
        """展平观测到 ``(obs_dim,)``。"""
        return np.asarray(obs, dtype="float64").reshape(-1)

    def learn(self, total_timesteps: int, seed: int | None = None) -> None:
        """执行 PPO 训练主循环。

        Args:
            total_timesteps: 目标累计环境步数。
            seed: 覆盖随机种子（会重建策略以复现）。
        """
        if seed is not None and int(seed) != self.seed:
            self.seed = int(seed)
            self.policy = NumpyActorCritic(
                self.obs_dim, self.n_actions, (64, 64), self.seed
            )
            self.optimizer = AdamOptimizer(
                self.policy.get_params(), lr=float(self.cfg.learning_rate)
            )

        rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

        cfg = self.cfg
        n_steps = int(max(8, min(cfg.n_steps, max(8, total_timesteps))))
        batch_size = int(max(1, min(cfg.batch_size, n_steps)))
        n_epochs = int(max(1, cfg.n_epochs))
        gamma = float(cfg.gamma)
        gae_lambda = float(cfg.gae_lambda)
        clip_range = float(cfg.clip_range)
        ent_coef = float(cfg.ent_coef)

        buffer = RolloutBuffer(n_steps, self.obs_dim)
        obs, _ = self.env.reset(seed=self.seed)
        ep_reward = 0.0
        ep_rewards: list[float] = []
        total_steps = 0

        while total_steps < total_timesteps:
            buffer.reset()
            # ---------------- 1) Rollout ----------------
            for _ in range(n_steps):
                flat = self._flatten(obs)
                action, logp, value = self.policy.act(flat, deterministic=False)
                next_obs, reward, terminated, truncated, _info = self.env.step(action)
                done = bool(terminated or truncated)
                buffer.add(flat, action, logp, reward, value, done)
                ep_reward += float(reward)
                obs = next_obs
                total_steps += 1

                if done:
                    ep_rewards.append(ep_reward)
                    ep_reward = 0.0
                    obs, _ = self.env.reset(seed=int(rng.integers(0, 2**31 - 1)))

            # ---------------- 2) GAE ----------------
            last_flat = self._flatten(obs)
            _, last_value_arr = self.policy.forward(last_flat[None, :])
            last_value = float(last_value_arr[0])
            advantages, returns = compute_gae(
                buffer.rewards[: buffer.ptr],
                buffer.values[: buffer.ptr],
                buffer.dones[: buffer.ptr],
                last_value,
                gamma,
                gae_lambda,
            )
            if advantages.size > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_batch = buffer.obs[: buffer.ptr]
            act_batch = buffer.actions[: buffer.ptr]
            old_logp_batch = buffer.logps[: buffer.ptr]
            n = buffer.ptr

            # ---------------- 3) 多 epoch 更新 ----------------
            idxs = np.arange(n)
            epoch_losses: list[float] = []
            for _epoch in range(n_epochs):
                rng.shuffle(idxs)
                for start in range(0, n, batch_size):
                    mb = idxs[start : start + batch_size]
                    if mb.size == 0:
                        continue
                    loss, grads = self.policy.compute_loss_and_grads(
                        obs_batch[mb],
                        act_batch[mb],
                        old_logp_batch[mb],
                        advantages[mb],
                        returns[mb],
                        clip_range=clip_range,
                        ent_coef=ent_coef,
                        vf_coef=0.5,
                    )
                    self.optimizer.step(grads)
                    epoch_losses.append(loss)

            self._loss_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
            mean_ep = float(np.mean(ep_rewards[-10:])) if ep_rewards else float(
                np.mean(buffer.rewards[: buffer.ptr])
            )
            self._reward_history.append(mean_ep)
            logger.info(
                "PPO(numpy) steps=%d/%d loss=%.4f mean_ep_reward=%.4f",
                total_steps, total_timesteps,
                self._loss_history[-1], mean_ep,
            )

        self.env.close()

    # ------------------------------------------------------------------ #
    def predict(
        self, obs: "np.ndarray", deterministic: bool = True
    ) -> tuple["np.ndarray", Any]:
        """预测动作。

        Args:
            obs: 单条或批量观测。
            deterministic: True 时取 argmax。

        Returns:
            ``(action, None)``；单条观测返回标量 ndarray，批量返回 1D 数组。
        """
        arr = np.asarray(obs, dtype="float64")
        single = arr.ndim == len(self.obs_shape)
        if single:
            action, _logp, _v = self.policy.act(arr, deterministic=deterministic)
            return np.asarray([action], dtype="int64"), None
        flat = arr.reshape(arr.shape[0], -1)
        logits, _value = self.policy.forward(flat)
        actions = np.argmax(logits, axis=-1) if deterministic else np.array(
            [int(np.random.choice(self.n_actions, p=self.policy._softmax(l)[0])) for l in logits]
        )
        return actions.astype("int64"), None

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        """保存模型为 ``.zip``（内部为 weights.npz + hparams.json + backend.json）。

        Args:
            path: 目标 ``.zip`` 路径（``.zip`` 后缀可省略，会自动补全）。
        """
        p = Path(path)
        if p.suffix.lower() != ".zip":
            p = p.with_suffix(".zip")
        p.parent.mkdir(parents=True, exist_ok=True)

        # 权重写入内存缓冲
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            **{f"param::{k}": v for k, v in self.policy.params.items()},
            np_obs_dim=np.array(self.obs_dim, dtype="int64"),
            np_n_actions=np.array(self.n_actions, dtype="int64"),
            np_hidden=np.array(self.policy.hidden, dtype="int64"),
        )
        hparams = {
            "algo": "ppo",
            "backend": "numpy",
            "seed": int(self.seed),
            "learning_rate": float(self.cfg.learning_rate),
            "gamma": float(self.cfg.gamma),
            "gae_lambda": float(self.cfg.gae_lambda),
            "clip_range": float(self.cfg.clip_range),
            "n_steps": int(self.cfg.n_steps),
            "batch_size": int(self.cfg.batch_size),
            "n_epochs": int(self.cfg.n_epochs),
            "ent_coef": float(self.cfg.ent_coef),
            "hidden": list(self.policy.hidden),
            "obs_shape": list(self.obs_shape),
        }
        with zipfile.ZipFile(str(p), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("weights.npz", buf.getvalue())
            zf.writestr("hparams.json", json.dumps(hparams, ensure_ascii=False, indent=2))
            zf.writestr(
                "backend.json",
                json.dumps({"backend": "numpy", "algo": "ppo_numpy"}, indent=2),
            )
        logger.info("NumPy PPO 模型已保存: %s", p)

    @classmethod
    def load(cls, path: Path, env: "Any" = None) -> "NumpyPPOTrainer":
        """从 ``.zip`` 加载模型。

        Args:
            path: 模型 zip 路径。
            env: 可选环境（用于推断 obs shape）。

        Returns:
            恢复后的 :class:`NumpyPPOTrainer`。

        Raises:
            ArtifactError: 文件缺失或格式损坏。
        """
        p = Path(path)
        if p.suffix.lower() != ".zip":
            p = p.with_suffix(".zip")
        if not p.exists():
            raise ArtifactError(
                f"模型文件不存在: {p}（请先执行 `stock_ai train`）", context={"path": str(p)}
            )
        try:
            with zipfile.ZipFile(str(p), "r") as zf:
                names = set(zf.namelist())
                if "weights.npz" not in names or "hparams.json" not in names:
                    raise ArtifactError(f"不是有效的 NumPy PPO 模型 zip: {p}")
                weights_bytes = zf.read("weights.npz")
                hparams = json.loads(zf.read("hparams.json").decode("utf-8"))
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactError(f"模型 zip 解析失败: {p} ({exc})") from exc

        buf = io.BytesIO(weights_bytes)
        with np.load(buf, allow_pickle=True) as data:
            obs_dim = int(data["np_obs_dim"])
            n_actions = int(data["np_n_actions"])
            hidden = tuple(int(x) for x in data["np_hidden"])
            policy = NumpyActorCritic(obs_dim, n_actions, hidden)
            for key in data.files:
                if key.startswith("param::"):
                    policy.params[key[len("param::"):]] = np.asarray(data[key], dtype="float64")

        cfg = TrainConfig(
            algo="ppo",
            learning_rate=float(hparams.get("learning_rate", 3e-4)),
            gamma=float(hparams.get("gamma", 0.99)),
            gae_lambda=float(hparams.get("gae_lambda", 0.95)),
            clip_range=float(hparams.get("clip_range", 0.2)),
            n_steps=int(hparams.get("n_steps", 2048)),
            batch_size=int(hparams.get("batch_size", 64)),
            n_epochs=int(hparams.get("n_epochs", 10)),
            ent_coef=float(hparams.get("ent_coef", 0.0)),
        )

        obj = cls.__new__(cls)
        obj.env = env
        obj.cfg = cfg
        obj.seed = int(hparams.get("seed", 42))
        obj.obs_shape = tuple(hparams.get("obs_shape", (1, obs_dim)))
        obj.obs_dim = obs_dim
        obj.n_actions = n_actions
        obj.policy = policy
        obj.optimizer = AdamOptimizer(policy.get_params(), lr=float(cfg.learning_rate))
        obj._loss_history = []
        obj._reward_history = []
        return obj

    # ------------------------------------------------------------------ #
    def build(self, timesteps: int) -> "NumpyPPOTrainer":
        """兼容类图中 ``build(timesteps)`` 的链式调用。

        Args:
            timesteps: 训练步数（暂存，供 :meth:`train` 使用）。

        Returns:
            ``self``。
        """
        self._pending_timesteps = int(timesteps)
        return self

    def train(self, timesteps: int | None = None) -> TrainResult:
        """执行训练并返回结果摘要。

        Args:
            timesteps: 训练步数（None 时用 :meth:`build` 暂存值或 ``cfg.timesteps``）。

        Returns:
            :class:`TrainResult`。
        """
        steps = int(timesteps or getattr(self, "_pending_timesteps", self.cfg.timesteps))
        t0 = time.time()
        self.learn(steps)
        elapsed = time.time() - t0
        best = max(self._reward_history) if self._reward_history else 0.0
        return TrainResult(
            run_dir=Path("."),
            model_path=Path("."),
            algo="ppo",
            backend="numpy",
            total_timesteps=steps,
            best_mean_reward=float(best),
            elapsed_sec=float(elapsed),
        )

    @property
    def loss_history(self) -> list[float]:
        """每轮平均损失历史。"""
        return list(self._loss_history)

    @property
    def reward_history(self) -> list[float]:
        """每轮平均奖励历史。"""
        return list(self._reward_history)
