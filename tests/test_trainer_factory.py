"""TrainerFactory 后端探测单测。"""

from __future__ import annotations

from stock_ai.config.schema import TrainConfig
from stock_ai.train.factory import TrainerFactory


def test_backend_available_no_exception():
    """backend_available 探测不抛异常。"""
    assert TrainerFactory.backend_available("numpy") is True
    # 无论是否安装 torch/sb3，都不应抛异常
    result = TrainerFactory.backend_available("sb3")
    assert isinstance(result, bool)
    result2 = TrainerFactory.backend_available("torch")
    assert isinstance(result2, bool)
    result3 = TrainerFactory.backend_available("nonexistent_package_xyz")
    assert result3 is False


def test_available_backends_snapshot():
    """可用后端快照包含全部键。"""
    snap = TrainerFactory.available_backends()
    for key in ("numpy", "gymnasium", "torch", "sb3"):
        assert key in snap
        assert isinstance(snap[key], bool)


def test_create_numpy_backend(small_split, env_cfg):
    """强制 numpy 后端创建成功。"""
    from stock_ai.env.trading_env import TradingEnv

    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="ppo", backend="numpy", n_steps=32, batch_size=16, n_epochs=1)
    trainer = TrainerFactory.create(cfg, env, preference="numpy")
    assert trainer.algo_name == "ppo_numpy"
    assert hasattr(trainer, "learn")
    assert hasattr(trainer, "predict")
    assert hasattr(trainer, "save")


def test_create_auto_falls_back(small_split, env_cfg):
    """auto 模式总能返回可用训练器。"""
    from stock_ai.env.trading_env import TradingEnv

    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="ppo", backend="auto", n_steps=32, batch_size=16, n_epochs=1)
    trainer = TrainerFactory.create(cfg, env, preference="auto")
    assert hasattr(trainer, "learn")
    assert getattr(trainer, "backend_name", None) in ("numpy", "sb3")


def test_create_sb3_forced_raises_when_unavailable(small_split, env_cfg):
    """强制 sb3 但不可用时抛 ArtifactError。"""
    import pytest

    from stock_ai.utils.errors import ArtifactError

    if TrainerFactory.backend_available("sb3"):
        pytest.skip("SB3 可用，跳过该断言")

    from stock_ai.env.trading_env import TradingEnv

    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="ppo", backend="sb3")
    with pytest.raises(ArtifactError):
        TrainerFactory.create(cfg, env, preference="sb3")
