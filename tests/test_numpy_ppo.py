"""NumPy PPO 兜底训练器单测。"""

from __future__ import annotations

import numpy as np
import pytest

from stock_ai.config.schema import TrainConfig
from stock_ai.env.trading_env import TradingEnv
from stock_ai.train.policy import AdamOptimizer, NumpyActorCritic
from stock_ai.train.ppo_numpy import NumpyPPOTrainer, compute_gae


def test_policy_forward_shapes():
    """网络前向形状正确。"""
    net = NumpyActorCritic(obs_dim=12, n_actions=3, hidden=(64, 64), seed=0)
    logits, value = net.forward(np.zeros((4, 12)))
    assert logits.shape == (4, 3)
    assert value.shape == (4,)


def test_policy_act_returns_valid_action():
    """act 返回合法动作与 logp/value。"""
    net = NumpyActorCritic(obs_dim=8, n_actions=3, seed=1)
    action, logp, value = net.act(np.zeros(8), deterministic=True)
    assert action in (0, 1, 2)
    assert logp <= 1e-6
    assert np.isfinite(value)


def test_policy_save_load_roundtrip(tmp_path):
    """save/load 往返后 predict 一致。"""
    net = NumpyActorCritic(obs_dim=10, n_actions=3, seed=3)
    obs = np.random.default_rng(0).normal(size=10)
    a1, _, _ = net.act(obs, deterministic=True)
    p = tmp_path / "policy.npz"
    net.save(p)
    net2 = NumpyActorCritic.load(p)
    a2, _, _ = net2.act(obs, deterministic=True)
    assert a1 == a2


def test_adam_optimizer_reduces_loss():
    """Adam 能降低简单二次损失。"""
    param = [np.array([5.0, -3.0])]
    opt = AdamOptimizer(param, lr=0.1)
    for _ in range(200):
        grad = [2.0 * param[0]]  # d/dx x^2
        opt.step([grad[0]])
    assert np.allclose(param[0], 0.0, atol=0.1)


def test_compute_gae_known_values():
    """GAE 对照手算（单步、无终止）。"""
    rewards = np.array([1.0])
    values = np.array([0.5])
    dones = np.array([0.0])
    adv, ret = compute_gae(rewards, values, dones, last_value=0.0,
                           gamma=0.99, gae_lambda=0.95)
    # delta = 1 + 0 - 0.5 = 0.5 ; adv = 0.5
    assert adv[0] == pytest.approx(0.5)
    assert ret[0] == pytest.approx(1.0)   # adv + value


def test_numpy_ppo_learn_and_predict(small_split, env_cfg):
    """兜底训练器能学习并预测。"""
    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="ppo", timesteps=300, n_steps=64, batch_size=32, n_epochs=2)
    trainer = NumpyPPOTrainer(env, cfg, seed=42)
    trainer.learn(300, seed=42)
    obs, _ = env.reset(seed=42)
    action, _ = trainer.predict(obs, deterministic=True)
    assert int(np.asarray(action).reshape(-1)[0]) in (0, 1, 2)


def test_numpy_ppo_save_load_roundtrip(small_split, env_cfg, tmp_path):
    """兜底训练器 save/load 往返 predict 一致。"""
    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="ppo", timesteps=200, n_steps=32, batch_size=16, n_epochs=1)
    trainer = NumpyPPOTrainer(env, cfg, seed=7)
    trainer.learn(200, seed=7)

    model_path = tmp_path / "ppo_test.zip"
    trainer.save(model_path)
    assert model_path.exists()

    loaded = NumpyPPOTrainer.load(model_path)
    obs, _ = env.reset(seed=42)
    a1, _ = trainer.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert int(np.asarray(a1).reshape(-1)[0]) == int(np.asarray(a2).reshape(-1)[0])


def test_numpy_ppo_deterministic_reproducibility(small_split, env_cfg):
    """相同 seed 两次训练结果一致（容差 1e-6）。"""
    cfg = TrainConfig(algo="ppo", timesteps=200, n_steps=64, batch_size=32, n_epochs=2)
    e1 = TradingEnv(small_split, env_cfg, seed=1)
    t1 = NumpyPPOTrainer(e1, cfg, seed=1)
    t1.learn(200, seed=1)

    e2 = TradingEnv(small_split, env_cfg, seed=1)
    t2 = NumpyPPOTrainer(e2, cfg, seed=1)
    t2.learn(200, seed=1)

    obs = np.zeros((1, small_split.obs.shape[1] * small_split.obs.shape[2]))
    l1, _ = t1.policy.forward(obs)
    l2, _ = t2.policy.forward(obs)
    np.testing.assert_allclose(l1, l2, atol=1e-6)
