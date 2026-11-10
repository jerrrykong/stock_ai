"""环境合规性测试：gymnasium check_env 必须通过。"""

from __future__ import annotations

import numpy as np
import pytest

from stock_ai.env.rewards import RewardCalculator
from stock_ai.env.trading_env import TradingEnv


def test_check_env_gymnasium(small_split, env_cfg):
    """gymnasium.utils.env_checker.check_env 通过。"""
    from gymnasium.utils.env_checker import check_env

    env = TradingEnv(small_split, env_cfg, seed=42)
    check_env(env, skip_render_check=True)


def test_check_env_sb3_if_available(small_split, env_cfg):
    """SB3 可用时用 SB3 checker（更严格）。"""
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    env = TradingEnv(small_split, env_cfg, seed=42)
    sb3_check_env(env)


def test_reset_returns_obs_and_info(small_split, env_cfg):
    """reset 返回 (obs, info) 且 info 含固定键（K10）。"""
    env = TradingEnv(small_split, env_cfg, seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (small_split.obs.shape[1], small_split.obs.shape[2])
    for key in ("portfolio_value", "cash", "shares", "turnover", "position_ratio", "date", "trade_ok"):
        assert key in info


def test_step_info_keys_and_types(small_split, env_cfg):
    """step 返回 5 元组且 info 键类型正确（K10）。"""
    env = TradingEnv(small_split, env_cfg, seed=42)
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (small_split.obs.shape[1], small_split.obs.shape[2])
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info["portfolio_value"], float)
    assert isinstance(info["shares"], int)
    assert "blocked_reason" in info


def test_t1_blocks_same_day_sell(small_split, env_cfg):
    """T+1：买入当日不可卖出（held_days=0 被拒）；次日可卖。"""
    from stock_ai.env.rules import can_sell

    env = TradingEnv(small_split, env_cfg, seed=42)
    env.reset(seed=42)
    _, _, _, _, info_buy = env.step(1)   # 买入，记录 last_buy_idx = 该步 idx
    if not info_buy["trade_ok"]:
        pytest.skip("该样本首步无法买入，跳过 T+1 断言")

    # 同一交易日索引（held_days=0）必须被 T+1 拒绝
    st = env.state
    assert can_sell(st, price=float(env.prices[env.idx]), prev_close=float(env.prices[env.idx]),
                    halted=False, held_days=0) is False

    # 下一交易日索引（held_days=1）允许卖出（T+1 语义）
    assert can_sell(st, price=float(env.prices[env.idx]), prev_close=float(env.prices[env.idx]),
                    halted=False, held_days=1) is True


def test_t1_field_recorded_after_buy(small_split, env_cfg):
    """买入后 last_buy_idx 被记录，使 held_days 随时间增长。"""
    env = TradingEnv(small_split, env_cfg, seed=42)
    env.reset(seed=42)
    _, _, _, _, info_buy = env.step(1)
    if not info_buy["trade_ok"]:
        pytest.skip("该样本首步无法买入，跳过断言")
    assert env.state.last_buy_idx >= 0
    # 推进一天后 held_days 应 >= 1
    env.step(0)
    assert env.idx - env.state.last_buy_idx >= 1


def test_equity_curve_recorded(small_split, env_cfg):
    """跑完整 episode，权益曲线长度正确。"""
    env = TradingEnv(small_split, env_cfg, seed=42)
    env.reset(seed=42)
    done = False
    steps = 0
    while not done and steps < small_split.n_steps + 5:
        _, _, term, trunc, _ = env.step(0)
        done = term or trunc
        steps += 1
    assert len(env.equity) == steps + 1


def test_continuous_action_mode(small_split, env_cfg):
    """连续动作模式可运行。"""
    env = TradingEnv(small_split, env_cfg, seed=42, action_mode="continuous")
    env.reset(seed=42)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert "portfolio_value" in info
