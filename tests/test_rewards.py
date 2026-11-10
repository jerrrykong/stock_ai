"""奖励函数数值正确性单测。"""

from __future__ import annotations

import pytest

from stock_ai.config.schema import EnvConfig
from stock_ai.env.rewards import RewardCalculator


def test_reward_gross_is_plain_return():
    """gross 模式 = 纯收益率。"""
    cfg = EnvConfig(reward_type="gross")
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    r = calc.compute(prev_value=1_000_000.0, value=1_010_000.0, turnover=0.0, position_ratio=1.0)
    assert r == pytest.approx(0.01)


def test_reward_drawdown_penalty_formula():
    """drawdown_penalty = ret − λ₁dd − λ₂turnover − λ₃|pos|。"""
    cfg = EnvConfig(
        reward_type="drawdown_penalty",
        lambda_drawdown=0.1,
        lambda_turnover=0.001,
        lambda_leverage=0.01,
    )
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    # 先涨到峰值 1.05M
    calc.compute(1_000_000.0, 1_050_000.0, turnover=0.0, position_ratio=1.0)
    # 回落到 1.0M
    ret = (1_000_000.0 - 1_050_000.0) / 1_050_000.0
    dd = (1_050_000.0 - 1_000_000.0) / 1_050_000.0
    turnover = 0.02
    pos = 0.8
    expected = ret - cfg.lambda_drawdown * dd - cfg.lambda_turnover * turnover - cfg.lambda_leverage * pos
    got = calc.compute(1_050_000.0, 1_000_000.0, turnover=turnover, position_ratio=pos)
    assert got == pytest.approx(expected)


def test_reward_no_drawdown_when_rising():
    """持续上涨时无回撤惩罚。"""
    cfg = EnvConfig(reward_type="drawdown_penalty", lambda_turnover=0.0, lambda_leverage=0.0)
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    r1 = calc.compute(1_000_000.0, 1_020_000.0, turnover=0.0, position_ratio=0.0)
    r2 = calc.compute(1_020_000.0, 1_040_000.0, turnover=0.0, position_ratio=0.0)
    assert r1 == pytest.approx(0.02)
    assert r2 == pytest.approx(1_040_000.0 / 1_020_000.0 - 1.0)


def test_reward_turnover_and_leverage_penalty():
    """换手与杠杆惩罚为负贡献。"""
    cfg = EnvConfig(reward_type="drawdown_penalty", lambda_drawdown=0.0,
                    lambda_turnover=0.01, lambda_leverage=0.1)
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    r = calc.compute(1_000_000.0, 1_000_000.0, turnover=1.0, position_ratio=1.0)
    assert r == pytest.approx(-(0.01 * 1.0 + 0.1 * 1.0))
    assert r < 0


def test_reward_reset_clears_peak():
    """reset 后峰值清零。"""
    cfg = EnvConfig(reward_type="drawdown_penalty")
    calc = RewardCalculator(cfg)
    calc.reset(2_000_000.0)
    # 新峰值 2.0M，随后跌到 1.0M 应有回撤
    r = calc.compute(2_000_000.0, 1_000_000.0, turnover=0.0, position_ratio=1.0)
    assert r < 0


def test_reward_excess_is_excess_return():
    """excess 模式（无惩罚）= ret − benchmark_return。"""
    cfg = EnvConfig(reward_type="excess", lambda_drawdown=0.0,
                    lambda_turnover=0.0, lambda_leverage=0.0)
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    ret = 1_010_000.0 / 1_000_000.0 - 1.0
    r = calc.compute(1_000_000.0, 1_010_000.0, turnover=0.0, position_ratio=0.0,
                     benchmark_return=0.005)
    assert r == pytest.approx(ret - 0.005)


def test_reward_excess_penalizes_turnover():
    """excess 模式下换手惩罚仍生效（turnover 越大 reward 越小）。"""
    cfg = EnvConfig(reward_type="excess", lambda_drawdown=0.0,
                    lambda_turnover=0.01, lambda_leverage=0.0)
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    r_low = calc.compute(1_000_000.0, 1_000_000.0, turnover=0.0, position_ratio=0.0,
                         benchmark_return=0.0)
    r_high = calc.compute(1_000_000.0, 1_000_000.0, turnover=1.0, position_ratio=0.0,
                          benchmark_return=0.0)
    assert r_low > r_high


def test_reward_legacy_types_ignore_benchmark_kwarg():
    """旧三种类型对 benchmark_return 形参无感（零回归）。"""
    for reward_type in ("gross", "drawdown_penalty", "sharpe"):
        cfg = EnvConfig(reward_type=reward_type, lambda_turnover=0.002,
                        lambda_leverage=0.01)
        calc = RewardCalculator(cfg)
        calc.reset(1_000_000.0)
        base = calc.compute(1_000_000.0, 1_020_000.0, turnover=0.3, position_ratio=0.7)
        calc.reset(1_000_000.0)
        with_bench = calc.compute(1_000_000.0, 1_020_000.0, turnover=0.3,
                                  position_ratio=0.7, benchmark_return=0.09)
        assert base == with_bench
