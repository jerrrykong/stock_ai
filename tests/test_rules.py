"""A 股交易规则纯函数单测（重点）。"""

from __future__ import annotations

import math

import pytest

from stock_ai.env.account import EnvState
from stock_ai.env.rules import (
    apply_slippage,
    can_buy,
    can_sell,
    is_halted,
    is_limit_down,
    is_limit_up,
    limit_price,
    max_affordable_lots,
    round_lots,
    trade_cost,
)


# ------------------------- 涨跌停 -------------------------
def test_limit_price_rounding():
    """涨跌停价四舍五入到分。"""
    assert limit_price(10.0, "up") == 11.0
    assert limit_price(10.0, "down") == 9.0
    assert limit_price(3.33, "up") == pytest.approx(3.66, abs=0.01)


def test_is_limit_up():
    """最高价触及涨停价即判定涨停。"""
    assert is_limit_up(10.0, 11.0)
    assert is_limit_up(10.0, 11.5)
    assert not is_limit_up(10.0, 10.99)


def test_is_limit_down():
    """最低价触及跌停价即判定跌停。"""
    assert is_limit_down(10.0, 9.0)
    assert is_limit_down(10.0, 8.5)
    assert not is_limit_down(10.0, 9.01)


# ------------------------- 停牌 -------------------------
def test_is_halted():
    """成交量极低或为 0 视为停牌。"""
    assert is_halted(0.0, 1e6)
    assert is_halted(1000.0, 1e6)          # < 5%
    assert not is_halted(1e6, 1e6)
    assert is_halted(-1.0, 1e6)            # 异常值


# ------------------------- T+1 -------------------------
def test_can_buy_and_sell_t1():
    """当日买入后，held_days<1 时不可卖（T+1）。"""
    st = EnvState(cash=100000.0, shares=100, cost_basis=10.0)
    # 未停牌、未涨停 -> 可买
    assert can_buy(st, price=10.0, prev_close=10.0, halted=False)
    # 当日买入（held_days=0）不可卖
    assert not can_sell(st, price=11.0, prev_close=10.0, halted=False, held_days=0)
    # 次日（held_days=1）可卖
    assert can_sell(st, price=11.0, prev_close=10.0, halted=False, held_days=1)


def test_can_buy_blocked_by_limit_up_and_halt():
    """涨停 / 停牌时不可买入。"""
    st = EnvState(cash=100000.0)
    assert not can_buy(st, price=11.0, prev_close=10.0, halted=False)   # 涨停
    assert not can_buy(st, price=10.0, prev_close=10.0, halted=True)    # 停牌
    assert not can_buy(EnvState(cash=0.0), price=10.0, prev_close=10.0, halted=False)


def test_can_sell_blocked_by_limit_down():
    """跌停时不可卖出。"""
    st = EnvState(cash=0.0, shares=100, cost_basis=10.0)
    assert not can_sell(st, price=9.0, prev_close=10.0, halted=False, held_days=1)
    assert not can_sell(st, price=10.0, prev_close=10.0, halted=True, held_days=1)
    assert not can_sell(EnvState(cash=0.0, shares=0), price=10.0, prev_close=10.0,
                        halted=False, held_days=1)


# ------------------------- 滑点 -------------------------
def test_apply_slippage_direction():
    """买入价上浮、卖出价下浮。"""
    assert apply_slippage(100.0, "buy", 0.001) == pytest.approx(100.1)
    assert apply_slippage(100.0, "sell", 0.001) == pytest.approx(99.9)


# ------------------------- 成本 -------------------------
def test_trade_cost_buy_only_commission():
    """买入：仅佣金，且受最低佣金约束。"""
    # 大额：佣金 > 最低佣金
    cost = trade_cost(1_000_000.0, "buy", 0.00025, 0.0005, 5.0)
    assert cost == pytest.approx(250.0)
    # 小额：触发最低佣金 5 元
    cost_small = trade_cost(1000.0, "buy", 0.00025, 0.0005, 5.0)
    assert cost_small == pytest.approx(5.0)


def test_trade_cost_sell_commission_plus_stamp_tax():
    """卖出：佣金 + 印花税（精确匹配手算）。"""
    notional = 1_000_000.0
    expected = max(notional * 0.00025, 5.0) + notional * 0.0005
    assert trade_cost(notional, "sell", 0.00025, 0.0005, 5.0) == pytest.approx(expected)
    assert expected == pytest.approx(750.0)   # 250 佣金 + 500 印花税


def test_trade_cost_zero_notional():
    """零成交额无成本。"""
    assert trade_cost(0.0, "buy", 0.00025, 0.0005, 5.0) == 0.0
    assert trade_cost(-100.0, "sell", 0.00025, 0.0005, 5.0) == 0.0


# ------------------------- 整手 -------------------------
def test_round_lots_and_affordable():
    """整手向下取整。"""
    assert round_lots(1234, 100) == 1200
    assert round_lots(99, 100) == 0
    assert max_affordable_lots(100000.0, 10.0, 100) == 10000
    assert max_affordable_lots(0.0, 10.0, 100) == 0
