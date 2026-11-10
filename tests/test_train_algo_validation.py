"""训练参数校验单测（QA 缺陷修复回归）。

覆盖两个 QA 报告的源码缺陷：
    1. 非法 ``--algo`` 必须抛 :class:`ConfigError`（不能被 auto 后端回退吞掉）；
    2. ``--algo sac`` 在离散动作环境下必须给出领域化提示，不得泄漏 SB3 内部断言。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from stock_ai.config.schema import TrainConfig
from stock_ai.train.factory import TrainerFactory
from stock_ai.utils.errors import ConfigError

#: 仓库内默认配置的绝对路径（测试会 chdir，必须用绝对路径避免找不到配置）
_CFG = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


def _args(**overrides) -> argparse.Namespace:
    """构造 train_cli.run 所需的最小 argparse.Namespace。"""
    base = dict(
        config=_CFG,
        algo=None,
        samples=None,
        timesteps=8,
        seed=42,
        backend=None,
        reward=None,
        out=None,
        n_steps=None,
        batch_size=None,
        n_epochs=None,
        learning_rate=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------- #
# 缺陷 1：非法 algo 必须明确报错
# ---------------------------------------------------------------------- #
def test_factory_rejects_unknown_algo_numpy_backend(small_split, env_cfg):
    """强制 numpy 后端下非法 algo 也必须被拒绝（早于后端选择）。"""
    from stock_ai.env.trading_env import TradingEnv

    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="bogus", backend="numpy")
    with pytest.raises(ConfigError):
        TrainerFactory.create(cfg, env, preference="numpy")


def test_factory_rejects_unknown_algo_sb3_backend(small_split, env_cfg):
    """强制 sb3 后端下非法 algo 抛 ConfigError。"""
    from stock_ai.env.trading_env import TradingEnv

    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="bogus", backend="sb3")
    with pytest.raises(ConfigError):
        TrainerFactory.create(cfg, env, preference="sb3")


def test_factory_auto_does_not_swallow_unknown_algo(small_split, env_cfg):
    """auto 分支不得把非法 algo 当成"后端不可用"而静默回退到 numpy。"""
    from stock_ai.env.trading_env import TradingEnv

    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="bogus", backend="auto")
    # 无论 SB3 是否可用，都必须是 ConfigError，而不是"成功返回一个 numpy 训练器"
    with pytest.raises(ConfigError):
        TrainerFactory.create(cfg, env, preference="auto")


def test_train_cli_rejects_unknown_algo_nonzero_exit(tmp_workspace):
    """train CLI：非法 --algo 必须非零退出且提示明确。"""
    from stock_ai.train import train_cli

    with pytest.raises(ConfigError):
        train_cli.run(_args(algo="BOGUS"))


def test_train_cli_main_exits_nonzero_on_bad_algo(tmp_workspace, monkeypatch):
    """train CLI 主入口：非法 algo 经 main() 返回非零退出码。"""
    import sys

    from stock_ai.train import train_cli

    monkeypatch.setattr(
        sys, "argv", ["train_cli", "--config", _CFG, "--algo", "BOGUS"]
    )
    with pytest.raises(SystemExit) as ei:
        train_cli.main()
    assert ei.value.code != 0


# ---------------------------------------------------------------------- #
# 缺陷 2：SAC + 离散动作 -> 领域化 ConfigError（不泄漏 SB3 断言）
# ---------------------------------------------------------------------- #
def test_sb3_sac_discrete_raises_config_error(small_split, env_cfg):
    """SB3Trainer 用 SAC + 离散动作空间时应抛 ConfigError（而非 SB3 内部 AssertionError）。"""
    pytest.importorskip("stable_baselines3")

    from stock_ai.env.trading_env import TradingEnv

    env = TradingEnv(small_split, env_cfg, seed=42, action_mode="discrete")
    cfg = TrainConfig(algo="sac", backend="sb3", n_steps=16, batch_size=8)
    with pytest.raises(ConfigError) as ei:
        TrainerFactory.create(cfg, env, preference="sb3")
    msg = str(ei.value)
    assert "SAC" in msg
    assert "连续" in msg


def test_train_cli_rejects_sac_discrete(tmp_workspace):
    """train CLI：--algo sac（默认离散）应抛 ConfigError 且提示领域化。"""
    from stock_ai.train import train_cli

    with pytest.raises(ConfigError) as ei:
        train_cli.run(_args(algo="sac", backend="sb3"))
    assert "连续" in str(ei.value)


def test_valid_algo_whitelist_constant():
    """白名单常量与 SB3 支持集合一致。"""
    from stock_ai.train.factory import SUPPORTED_ALGOS
    from stock_ai.train.sb3_trainer import SB3Trainer

    assert frozenset({"ppo", "sac", "dqn"}) == SUPPORTED_ALGOS
    assert SUPPORTED_ALGOS == frozenset(SB3Trainer.SUPPORTED.keys())
