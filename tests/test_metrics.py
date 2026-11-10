"""指标计算单测（对照手算）。"""

from __future__ import annotations

import numpy as np
import pytest

from stock_ai.eval.metrics import (
    annual_return,
    compute_all,
    max_drawdown,
    sharpe,
    total_return,
    turnover,
    win_rate,
)


def test_total_return():
    """累计收益率。"""
    assert total_return(np.array([100.0, 110.0])) == pytest.approx(0.10)
    assert total_return(np.array([100.0, 90.0])) == pytest.approx(-0.10)
    assert total_return(np.array([100.0])) == 0.0


def test_max_drawdown_known_answer():
    """已知序列的最大回撤。"""
    # 峰值 12，谷底 6 -> 回撤 50%
    equity = np.array([10.0, 12.0, 8.0, 6.0, 9.0, 11.0])
    assert max_drawdown(equity) == pytest.approx((12.0 - 6.0) / 12.0)


def test_max_drawdown_monotonic_zero():
    """单调上涨 -> 无回撤。"""
    assert max_drawdown(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.0)


def test_sharpe_zero_std():
    """零波动 -> 夏普 0。"""
    assert sharpe(np.array([0.01, 0.01, 0.01])) == 0.0


def test_sharpe_positive():
    """正收益序列 -> 正夏普。"""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 252)
    assert sharpe(rets) != 0.0


def test_annual_return_geometric():
    """年化收益率（几何）对照手算。"""
    # 两年翻倍 -> 年化 sqrt(2)-1 ≈ 41.4%
    dates = np.array(["20200101", "20220101"])
    equity = np.array([100.0, 200.0])
    ar = annual_return(equity, dates)
    assert ar == pytest.approx(2 ** (365.25 / 731) - 1, abs=0.01)


def test_win_rate_and_turnover():
    """胜率与换手率。"""
    trades = [
        {"notional": 100.0, "realized_pnl": 10.0},   # win
        {"notional": 100.0, "realized_pnl": -5.0},   # loss
        {"notional": 100.0, "realized_pnl": 3.0},    # win
        {"notional": 100.0, "side": "buy"},          # open, not counted
    ]
    assert win_rate(trades) == pytest.approx(2 / 3)
    equity = np.array([1000.0, 1000.0, 1000.0])
    # 总成交额 400 / 平均权益 1000 = 0.4
    assert turnover(trades, equity) == pytest.approx(0.4)


def test_win_rate_empty():
    """无平仓 -> 胜率 0。"""
    assert win_rate([]) == 0.0
    assert win_rate([{"side": "buy", "notional": 100.0}]) == 0.0


def test_compute_all_fields():
    """compute_all 返回 PRD §6 全部字段。"""
    equity = np.array([1000000.0, 1010000.0, 990000.0, 1050000.0])
    dates = np.array(["20230103", "20230104", "20230105", "20230106"])
    trades = [{"notional": 100000.0, "realized_pnl": 5000.0}]
    baseline = np.array([1000000.0, 1005000.0, 1000000.0, 1010000.0])
    m = compute_all(equity, dates, trades, benchmark_equity=baseline)
    for key in (
        "annual_return", "total_return", "max_drawdown", "sharpe",
        "win_rate", "turnover", "num_trades", "benchmark_return",
    ):
        assert key in m
    assert m["num_trades"] == 1
    assert m["benchmark_return"] == pytest.approx(0.01)
