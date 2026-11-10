"""纯 NumPy Actor-Critic 策略网络（仅依赖 numpy）。

结构：输入展平 -> Dense(hidden) tanh -> Dense(hidden) tanh -> {policy head, value head}。
Policy head 输出离散 logits（softmax），Value head 输出标量状态价值。
Adam 优化器手写实现（带一阶/二阶矩估计）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stock_ai.utils.errors import ArtifactError
from stock_ai.utils.logging import LoggerFactory

__all__ = ["NumpyActorCritic", "AdamOptimizer", "orthogonal_init"]

logger = LoggerFactory.get(__name__)


def orthogonal_init(rng: "np.random.Generator", shape: tuple[int, ...], gain: float = 1.0) -> "np.ndarray":
    """正交初始化（PPO 常用），保证训练稳定性。

    Args:
        rng: numpy 随机生成器。
        shape: 目标矩阵形状 ``(out_dim, in_dim)``。
        gain: 增益因子。

    Returns:
        初始化后的权重矩阵。
    """
    flat = np.prod(shape)
    if flat == 0:
        return np.zeros(shape, dtype="float64")
    a = rng.normal(0.0, 1.0, size=(flat, flat))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    out_dim = shape[0]
    q = q[:out_dim, :] if q.shape[1] >= out_dim else q.T[:out_dim, :]
    # 需要裁剪到 shape
    w = q[: shape[0], : shape[1]] if q.shape[0] >= shape[0] and q.shape[1] >= shape[1] else None
    if w is None:
        # 退化场景：直接用小尺度高斯
        w = rng.normal(0.0, 1.0 / np.sqrt(max(1, shape[1])), size=shape)
    return (w * gain).astype("float64")


class AdamOptimizer:
    """手写 Adam 优化器（无 torch 依赖）。

    Args:
        params: 参数列表（会被原地更新）。
        lr: 学习率。
        beta1: 一阶矩衰减。
        beta2: 二阶矩衰减。
        eps: 数值稳定项。
    """

    def __init__(
        self,
        params: list["np.ndarray"],
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """初始化，为每个参数分配矩估计缓冲。"""
        self.params: list["np.ndarray"] = params
        self.lr: float = float(lr)
        self.beta1: float = float(beta1)
        self.beta2: float = float(beta2)
        self.eps: float = float(eps)
        self.m: list["np.ndarray"] = [np.zeros_like(p) for p in params]
        self.v: list["np.ndarray"] = [np.zeros_like(p) for p in params]
        self.t: int = 0

    def step(self, grads: list["np.ndarray"]) -> None:
        """执行一次参数更新。

        Args:
            grads: 与 ``params`` 一一对应的梯度列表。
        """
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if p.shape != g.shape:
                raise ArtifactError(
                    f"Adam 梯度形状不匹配: param={p.shape} grad={g.shape}"
                )
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

    def zero_grad(self) -> None:
        """（语义占位）梯度由调用方新建，无需清零。"""
        return None

    def state_dict(self) -> dict[str, Any]:
        """序列化优化器状态。"""
        return {
            "m": [m.tolist() for m in self.m],
            "v": [v.tolist() for v in self.v],
            "t": int(self.t),
            "lr": self.lr,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """恢复优化器状态。"""
        if not d:
            return
        self.m = [np.asarray(x, dtype="float64") for x in d.get("m", [])] or self.m
        self.v = [np.asarray(x, dtype="float64") for x in d.get("v", [])] or self.v
        self.t = int(d.get("t", 0))


class NumpyActorCritic:
    """纯 NumPy MLP Actor-Critic。

    Args:
        obs_dim: 展平后的观测维度。
        n_actions: 离散动作数。
        hidden: 隐藏层宽度元组（默认 ``(64, 64)``）。
        seed: 初始化随机种子。
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (64, 64),
        seed: int = 42,
    ) -> None:
        """初始化网络与参数。"""
        self.obs_dim: int = int(obs_dim)
        self.n_actions: int = int(n_actions)
        self.hidden: tuple[int, ...] = tuple(int(h) for h in hidden) or (64, 64)
        self.seed: int = int(seed)

        rng = np.random.default_rng(self.seed)
        dims = [self.obs_dim, *self.hidden]
        self.params: dict[str, "np.ndarray"] = {}
        self._param_order: list[str] = []

        # 隐藏层
        for i in range(len(dims) - 1):
            w, b = self._init_layer(rng, dims[i], dims[i + 1], gain=np.sqrt(2.0))
            self._register(f"W{i}", w)
            self._register(f"b{i}", b)

        last = dims[-1]
        # 策略头（小增益，让初始 softmax 接近均匀）
        pw, pb = self._init_layer(rng, last, self.n_actions, gain=0.01)
        self._register("Wpi", pw)
        self._register("bpi", pb)
        # 价值头
        vw, vb = self._init_layer(rng, last, 1, gain=1.0)
        self._register("Wv", vw)
        self._register("bv", vb)

    # ------------------------------------------------------------------ #
    def _register(self, name: str, value: "np.ndarray") -> None:
        """登记一个参数。"""
        self.params[name] = value
        self._param_order.append(name)

    @staticmethod
    def _init_layer(
        rng: "np.random.Generator", in_dim: int, out_dim: int, gain: float
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """初始化一层（权重 + 偏置）。"""
        std = gain / np.sqrt(max(1, in_dim))
        w = rng.normal(0.0, std, size=(in_dim, out_dim)).astype("float64")
        b = np.zeros(out_dim, dtype="float64")
        return w, b

    # ------------------------------------------------------------------ #
    def _forward_all(self, obs: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", list["np.ndarray"]]:
        """前向传播，返回 logits、value 与隐藏激活缓存。

        Args:
            obs: ``(B, obs_dim)`` 展平观测。

        Returns:
            ``(logits (B,A), value (B,), activations)``。
        """
        x = np.asarray(obs, dtype="float64")
        if x.ndim == 1:
            x = x[None, :]
        activations: list["np.ndarray"] = [x]
        h = x
        n_hidden = len(self.hidden)
        for i in range(n_hidden):
            h = np.tanh(h @ self.params[f"W{i}"] + self.params[f"b{i}"])
            activations.append(h)
        logits = h @ self.params["Wpi"] + self.params["bpi"]
        value = (h @ self.params["Wv"] + self.params["bv"]).reshape(-1)
        return logits, value, activations

    def forward(self, obs: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        """前向传播，返回 ``(logits, value)``。

        Args:
            obs: 观测（会自动展平 batch 维）。

        Returns:
            ``(logits (B,A), value (B,))``。
        """
        logits, value, _ = self._forward_all(obs)
        return logits, value

    @staticmethod
    def _softmax(logits: "np.ndarray") -> "np.ndarray":
        """数值稳定的 softmax。"""
        z = logits - np.max(logits, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)

    def act(
        self, obs: "np.ndarray", deterministic: bool = False
    ) -> tuple[int, float, float]:
        """对单条观测采样动作。

        Args:
            obs: 单条观测（``(window, n_features)`` 或展平）。
            deterministic: True 时取 argmax。

        Returns:
            ``(action, log_prob, value)``。
        """
        flat = np.asarray(obs, dtype="float64").reshape(1, -1)
        logits, value, _ = self._forward_all(flat)
        probs = self._softmax(logits)[0]
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.n_actions, p=probs))
        logp = float(np.log(probs[action] + 1e-12))
        return action, logp, float(value[0])

    # ------------------------------------------------------------------ #
    def log_prob_value(
        self, obs: "np.ndarray", actions: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """批量计算 ``log_prob`` / ``value`` / ``entropy``（训练用）。

        Args:
            obs: ``(B, obs_dim)``。
            actions: ``(B,)`` 离散动作。

        Returns:
            ``(logp (B,), value (B,), entropy (B,))``。
        """
        logits, value, _ = self._forward_all(obs)
        probs = self._softmax(logits)
        a = np.asarray(actions, dtype="int64").reshape(-1)
        logp = np.log(probs[np.arange(len(a)), a] + 1e-12)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
        return logp, value, entropy

    # ------------------------------------------------------------------ #
    def compute_loss_and_grads(
        self,
        obs: "np.ndarray",
        actions: "np.ndarray",
        old_logp: "np.ndarray",
        advantages: "np.ndarray",
        returns: "np.ndarray",
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
    ) -> tuple[float, list["np.ndarray"]]:
        """计算 PPO clipped surrogate + value + entropy 损失及其梯度。

        Args:
            obs: ``(B, obs_dim)``。
            actions: ``(B,)``。
            old_logp: ``(B,)`` 旧策略的 log 概率。
            advantages: ``(B,)`` 优势（已标准化）。
            returns: ``(B,)`` 目标回报。
            clip_range: PPO 裁剪范围。
            ent_coef: 熵系数。
            vf_coef: 价值损失系数。

        Returns:
            ``(total_loss, grads)``，``grads`` 顺序与 :attr:`_param_order` 一致。
        """
        obs = np.asarray(obs, dtype="float64")
        actions = np.asarray(actions, dtype="int64").reshape(-1)
        old_logp = np.asarray(old_logp, dtype="float64").reshape(-1)
        advantages = np.asarray(advantages, dtype="float64").reshape(-1)
        returns = np.asarray(returns, dtype="float64").reshape(-1)
        batch = obs.shape[0]

        logits, value, acts = self._forward_all(obs)
        probs = self._softmax(logits)                       # (B, A)
        logp_all = np.log(probs + 1e-12)                    # (B, A)
        chosen_logp = logp_all[np.arange(batch), actions]   # (B,)

        ratio = np.exp(chosen_logp - old_logp)              # (B,)
        clipped = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
        unclipped_obj = ratio * advantages
        clipped_obj = clipped * advantages

        # 哪些样本用 unclipped（否则梯度为 0）
        use_unclipped = (unclipped_obj <= clipped_obj).astype("float64")  # (B,)
        policy_loss = -np.mean(np.minimum(unclipped_obj, clipped_obj))

        value_loss = np.mean((value - returns) ** 2)
        entropy = -np.sum(probs * logp_all, axis=-1)
        entropy_mean = np.mean(entropy)
        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

        # ---------------- 反向传播 ----------------
        # d(policy_loss)/d(chosen_logp)
        # 由 ratio 对 chosen_logp 的导数：ratio * adv * (-use_unclipped) （因为 loss 取了负号）
        d_logp = (ratio * advantages * (-use_unclipped)) / batch      # (B,)
        # d(value_loss)/d(value)
        d_value = 2.0 * (value - returns) / batch * vf_coef           # (B,)

        # entropy 梯度：d(-ent_coef*H)/d(logits)
        # H = -sum p*logp ; d(-ent)/d logits = ent_coef * p * (logp + H)
        d_logits_ent = ent_coef * probs * (logp_all + entropy[:, None]) / batch  # (B, A)

        # 组合 logits 梯度
        d_logits = np.zeros_like(logits)
        d_logits[np.arange(batch), actions] += d_logp
        d_logits += d_logits_ent

        h_last = acts[-1]                                     # (B, hidden_last)

        grads_map: dict[str, "np.ndarray"] = {}
        grads_map["Wpi"] = h_last.T @ d_logits                # (H, A)
        grads_map["bpi"] = d_logits.sum(axis=0)               # (A,)

        grads_map["Wv"] = h_last.T @ d_value[:, None]         # (H, 1)
        grads_map["bv"] = np.array([d_value.sum()], dtype="float64")

        # 反传到隐藏层
        d_h = d_logits @ self.params["Wpi"].T + d_value[:, None] @ self.params["Wv"].T
        n_hidden = len(self.hidden)
        for i in range(n_hidden - 1, -1, -1):
            h_prev = acts[i]
            h_cur = acts[i + 1]
            d_pre = d_h * (1.0 - h_cur * h_cur)               # tanh 导数
            grads_map[f"W{i}"] = h_prev.T @ d_pre
            grads_map[f"b{i}"] = d_pre.sum(axis=0)
            if i > 0:
                d_h = d_pre @ self.params[f"W{i}"].T

        grads = [grads_map[name] for name in self._param_order]
        return float(total_loss), grads

    # ------------------------------------------------------------------ #
    def get_params(self) -> list["np.ndarray"]:
        """按固定顺序返回参数列表。"""
        return [self.params[name] for name in self._param_order]

    def save(self, path: Path) -> None:
        """保存权重到 ``.npz``。

        Args:
            path: 目标路径（建议 ``.npz``）。
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(p),
            **{f"param::{k}": v for k, v in self.params.items()},
            np_obs_dim=np.array(self.obs_dim, dtype="int64"),
            np_n_actions=np.array(self.n_actions, dtype="int64"),
            np_hidden=np.array(self.hidden, dtype="int64"),
        )

    @classmethod
    def load(cls, path: Path) -> "NumpyActorCritic":
        """从 ``.npz`` 恢复网络。

        Args:
            path: ``.npz`` 路径。

        Returns:
            恢复后的 :class:`NumpyActorCritic`。
        """
        p = Path(path)
        if not p.exists():
            raise ArtifactError(f"策略权重不存在: {p}", context={"path": str(p)})
        with np.load(str(p), allow_pickle=True) as data:
            obs_dim = int(data["np_obs_dim"])
            n_actions = int(data["np_n_actions"])
            hidden = tuple(int(x) for x in data["np_hidden"])
            obj = cls(obs_dim, n_actions, hidden)
            for key in data.files:
                if key.startswith("param::"):
                    obj.params[key[len("param::"):]] = np.asarray(data[key], dtype="float64")
        return obj

    def to_state_dict(self) -> dict[str, Any]:
        """返回参数字典（用于 zip 内嵌入）。"""
        return {k: v for k, v in self.params.items()}
