"""跨标的边界处理回归测试（U6：v1 单标的独立训练）。

验证多标的样本按日期拼接后，``TradingEnv`` 不会把前一标的的持仓 / 昨收
带到下一标的，从而避免"跨标的低买高卖"导致的组合市值爆炸。
"""

from __future__ import annotations

import numpy as np

from stock_ai.env.trading_env import TradingEnv
from stock_ai.samples.schema import SampleSplit


def _two_symbol_split(window: int = 3) -> SampleSplit:
    """构造两只股票拼接的样本（价格量级差异巨大，放大潜在失真）。

    第一只 60 天价格区间 ~10 元，第二只 60 天 ~300 元，拼接处价格从
    10 元跳到 300 元（30 倍），若不做边界处理，账户会瞬间虚增 30 倍。
    """
    n = 60
    p_a = np.linspace(10.0, 11.0, n)
    p_b = np.linspace(300.0, 330.0, n)
    prices = np.concatenate([p_a, p_b])
    dates = np.array(
        [f"2023{(i // 20) + 1:02d}{(i % 20) + 1:02d}" for i in range(n * 2)],
        dtype="<U8",
    )
    symbols = np.array(["AAA"] * n + ["BBB"] * n, dtype=object)
    obs = np.zeros((len(prices), window, 4), dtype="float32")
    return SampleSplit(
        obs=obs,
        dates=dates,
        symbols=symbols,
        prices=prices,
        requested_start="20230101",
        requested_end="20231231",
    )


def test_symbol_boundary_flagged(env_cfg):
    """符号切换处必须被标记，且 prev_close 取当日本身（无跨标的昨收）。"""
    split = _two_symbol_split()
    env = TradingEnv(split, env_cfg, seed=42)
    # 第 60 步（索引）是新标的 BBB 的第一天
    assert bool(env.symbol_boundary[60]) is True
    assert bool(env.symbol_boundary[0]) is False
    # 前 60 步为同一标的，仅第 60 步是边界
    assert int(env.symbol_boundary.sum()) == 1
    # 边界处 prev_close 必须等于当日本身，避免误用前一标的昨收
    assert env.prev_closes[60] == split.prices[60]


def test_no_cross_symbol_position_carry(env_cfg):
    """在 AAA 段满仓买入后，进入 BBB 段必须被强平（不留跨标的持仓）。"""
    split = _two_symbol_split()
    env = TradingEnv(split, env_cfg, seed=42)
    env.reset(seed=42)
    # 前 59 步一直持有 AAA（买入后持续 hold）——idx 0..59 属于 AAA
    for _ in range(59):
        _obs, _r, term, _tr, _info = env.step(1 if env.state.shares == 0 else 0)
        if term:
            break
    assert env.state.shares > 0, "前置条件：应在 AAA 段持有仓位"

    # 第 59 步仍是 AAA（内部推进 idx -> 60），不触发边界
    env.step(0)
    assert env.idx == 60
    # 第 60 步（idx==60）才是 BBB 第一天：动作前应先被强平
    _obs, _r, _term, _tr, info = env.step(0)
    assert env.state.shares == 0, "跨标的必须强平，不得带走上一标的持仓"
    # 有 symbol_boundary 强平记录
    assert any(t.get("reason") == "symbol_boundary" for t in env.trades)
    # info 中不应出现异常大的组合市值（无 30 倍虚增）
    assert info["portfolio_value"] < env_cfg.initial_cash * 2.0


def test_equity_stays_bounded(env_cfg):
    """整段 episode 组合市值必须保持在合理量级（回归 1e198 爆炸 bug）。"""
    split = _two_symbol_split()
    env = TradingEnv(split, env_cfg, seed=42)
    env.reset(seed=42)
    done = False
    while not done:
        _obs, _r, term, trunc, _info = env.step(1)  # 始终尝试满仓
        done = term or trunc
    eq = env.equity
    # 无杠杆、无跨标的套利时组合市值不可能超过初始资金的 3 倍
    assert eq.max() < env_cfg.initial_cash * 3.0
    assert np.all(np.isfinite(eq))
