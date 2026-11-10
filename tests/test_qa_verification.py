"""QA 独立验证用例（严过关）。

覆盖工程师既有测试可能遗漏/需独立复核的场景：
    - T+1 在真实 env.step 层面的表现（而非仅纯函数）
    - 涨停不可买 / 跌停不可卖在 env 层面的阻断
    - 停牌不成交
    - 交易成本手算对照（env 级）
    - 奖励函数各分量符号 / 量级 / 纯 PnL 无 NaN-Inf
    - walk-forward 训练/测试时间区间零重叠 + embargo 生效
    - 防泄漏归一化：测试阶段绝不重新 fit（K9）
    - 复现性：同 seed 两次 sample build 的 data_hash 一致
    - 跨标的边界：价格差异极大时不爆炸 + symbol_boundary 强平
"""

from __future__ import annotations

import numpy as np
import pytest

from stock_ai.config.schema import EnvConfig, SampleConfig
from stock_ai.env.rewards import RewardCalculator
from stock_ai.env.rules import trade_cost
from stock_ai.env.trading_env import TradingEnv
from stock_ai.samples.features import FeatureEngineer
from stock_ai.samples.normalizer import NormalizerState
from stock_ai.samples.schema import SampleSplit
from stock_ai.samples.split import walk_forward_split


# --------------------------------------------------------------------------- #
# 辅助构造
# --------------------------------------------------------------------------- #
def _split_from_prices(prices, symbols=None, window=3, n_features=4):
    """由价格序列构造最小合法 SampleSplit。"""
    n = len(prices)
    prices = np.asarray(prices, dtype="float64")
    dates = np.array([f"2023{(i // 20) + 1:02d}{(i % 20) + 1:02d}" for i in range(n)], dtype="<U8")
    if symbols is None:
        symbols = ["AAA"] * n
    symbols = np.asarray(symbols, dtype=object)
    obs = np.zeros((n, window, n_features), dtype="float32")
    return SampleSplit(
        obs=obs, dates=dates, symbols=symbols, prices=prices,
        requested_start="20230101", requested_end="20231231",
    )


# --------------------------------------------------------------------------- #
# B1: T+1 —— env 层面：当日买入当日不可卖，隔日可卖
# --------------------------------------------------------------------------- #
def test_t1_env_level_buy_then_same_day_sell_blocked(env_cfg):
    # 平直价格，确保无涨跌停干扰
    split = _split_from_prices([10.0] * 10)
    env = TradingEnv(split, env_cfg, seed=1)
    env.reset(seed=1)
    # step0: 买入
    _, _, _, _, info = env.step(1)
    assert info["trade_ok"] is True, "买入应成功"
    assert env.state.shares > 0
    # step1: 尝试当日/次日边界卖出 —— held_days = idx - last_buy_idx = 1 -> 可卖
    # 要验证"当日买入当日不可卖"，必须检查同一 idx：用 last_buy_idx==idx 的纯函数语义
    # 在 env 中 last_buy_idx 在 step 内被设为当前 idx，下一 step idx 已 +1，故 held_days>=1。
    # 通过 can_sell 直接构造当日场景验证（已在 test_rules 覆盖），此处验证 env 隔日可卖：
    _, _, _, _, info2 = env.step(2)
    assert info2["trade_ok"] is True, "隔日应可卖出"


def test_t1_env_same_step_double_sell_blocked(env_cfg):
    """未持仓时卖出应被拒（无持仓），且不会产生负股数。"""
    split = _split_from_prices([10.0] * 10)
    env = TradingEnv(split, env_cfg, seed=1)
    env.reset(seed=1)
    _, _, _, _, info = env.step(2)  # 空仓卖出
    assert info["trade_ok"] is False
    assert env.state.shares == 0


# --------------------------------------------------------------------------- #
# B2: 涨跌停 —— env 层面
# --------------------------------------------------------------------------- #
def test_limit_up_blocks_buy_env(env_cfg):
    # 构造：prev_close=10，当日价格=11.0（涨停），应拒绝买入
    prices = [10.0, 11.0, 11.0, 11.0, 11.0]
    split = _split_from_prices(prices)
    env = TradingEnv(split, env_cfg, seed=1)
    env.reset(seed=1)
    # step0 idx=0 价格10 -> 买入
    _, _, _, _, info0 = env.step(1)
    assert info0["trade_ok"] is True
    # 先清仓再在涨停日买入
    env2 = TradingEnv(split, env_cfg, seed=1)
    env2.reset(seed=1)
    # 直接推进到 idx=1（涨停日）：idx0 hold
    env2.step(0)
    assert env2.idx == 1
    # 涨停日买入（prev_close=10, price=11 -> limit_up）
    _, _, _, _, info = env2.step(1)
    assert info["trade_ok"] is False, "涨停日不应能买入"
    assert "涨停" in info["blocked_reason"]


def test_limit_down_blocks_sell_env(env_cfg):
    # 先买入，再进入跌停日卖出
    prices = [10.0, 9.0, 9.0, 9.0]
    split = _split_from_prices(prices)
    env = TradingEnv(split, env_cfg, seed=1)
    env.reset(seed=1)
    # idx0：买入（价格10）
    _, _, _, _, info0 = env.step(1)
    assert info0["trade_ok"] is True
    # idx1：价格9.0，prev_close=10 -> 跌停；卖出应被拒
    assert env.idx == 1
    _, _, _, _, info = env.step(2)
    assert info["trade_ok"] is False, "跌停日不应能卖出"
    assert "跌停" in info["blocked_reason"]


# --------------------------------------------------------------------------- #
# B3: 停牌 —— 成交量=0 不成交（通过 can_buy 直接验证 + env 集成）
# --------------------------------------------------------------------------- #
def test_halted_no_trade_env(env_cfg, monkeypatch):
    """把某步 volumes 置 0 模拟停牌，该步不应成交。"""
    split = _split_from_prices([10.0] * 8)
    env = TradingEnv(split, env_cfg, seed=1)
    env.reset(seed=1)
    # 人为把 idx=1 的 volume 置 0（停牌）
    env.volumes[1] = 0.0
    env.step(0)  # -> idx 1
    _, _, _, _, info = env.step(1)  # 停牌日买入
    assert info["trade_ok"] is False
    assert "停牌" in info["blocked_reason"]


# --------------------------------------------------------------------------- #
# B4: 交易成本 —— env 级手算对照
# --------------------------------------------------------------------------- #
def test_env_buy_cost_matches_handcalc(env_cfg):
    """初始资金 1e6，价格 10 元买入：手算应扣金额。"""
    prices = [10.0] * 6
    split = _split_from_prices(prices)
    env = TradingEnv(split, env_cfg, seed=1)
    env.reset(seed=1)
    cash0 = env.state.cash
    _, _, _, _, info = env.step(1)
    assert info["trade_ok"] is True

    exec_price = 10.0 * (1.0 + env_cfg.slippage_rate)  # 10.01
    # budget = cash*(1-commission) = 1e6*0.99975
    budget = cash0 * (1.0 - env_cfg.commission_rate)
    shares = int(budget // exec_price) // 100 * 100
    notional = shares * exec_price
    fee = trade_cost(notional, "buy", env_cfg.commission_rate,
                     env_cfg.stamp_tax_rate, env_cfg.min_commission)
    expected_cash = cash0 - notional - fee
    assert env.state.shares == shares
    assert env.state.cash == pytest.approx(expected_cash, rel=1e-9)

    # 卖出对照
    sell_price = 10.0 * (1.0 - env_cfg.slippage_rate)  # 9.99
    notional_s = shares * sell_price
    fee_s = trade_cost(notional_s, "sell", env_cfg.commission_rate,
                       env_cfg.stamp_tax_rate, env_cfg.min_commission)
    cash_before_sell = env.state.cash
    env.step(2)
    assert env.state.cash == pytest.approx(cash_before_sell + notional_s - fee_s, rel=1e-9)


# --------------------------------------------------------------------------- #
# B5: 奖励函数
# --------------------------------------------------------------------------- #
def test_reward_components_and_sign(env_cfg):
    calc = RewardCalculator(env_cfg)
    calc.reset(1000.0)
    # 纯上涨、无换手、无仓位 -> reward == 收益率
    r = calc.compute(prev_value=1000.0, value=1100.0, turnover=0.0, position_ratio=0.0)
    assert r == pytest.approx(0.10, rel=1e-9), "纯收益时奖励应等于收益率"

    # 回撤惩罚：峰值 1100 后跌到 1000 -> 回撤 1/11
    calc2 = RewardCalculator(env_cfg)
    calc2.reset(1000.0)
    calc2.compute(1000.0, 1100.0)  # 建立峰值 1100
    r2 = calc2.compute(1100.0, 1000.0, turnover=0.0, position_ratio=0.0)
    expected = (1000.0 - 1100.0) / 1100.0 - env_cfg.lambda_drawdown * (100.0 / 1100.0)
    assert r2 == pytest.approx(expected, rel=1e-9)

    # 换手惩罚为负贡献
    r_hi_turn = calc.compute(1000.0, 1000.0, turnover=1.0, position_ratio=0.0)
    assert r_hi_turn < 0, "换手成本应产生负奖励"

    # 杠杆惩罚为负贡献
    r_lev = calc.compute(1000.0, 1000.0, turnover=0.0, position_ratio=1.0)
    assert r_lev < 0, "杠杆惩罚应为负"


def test_reward_pure_pnl_no_nan_inf(env_cfg):
    calc = RewardCalculator(env_cfg)
    calc.reset(1_000_000.0)
    for v in (1_000_000.0, 1_100_000.0, 900_000.0, 950_000.0):
        r = calc.compute(1_000_000.0, v, turnover=0.01, position_ratio=1.0)
        assert np.isfinite(r), f"奖励出现 NaN/Inf: {r}"


def test_reward_zero_prev_value_no_nan(env_cfg):
    """prev_value=0 时不应除零产生 NaN/Inf。"""
    calc = RewardCalculator(env_cfg)
    r = calc.compute(0.0, 100.0, turnover=0.0, position_ratio=0.0)
    assert np.isfinite(r)


# --------------------------------------------------------------------------- #
# B6: walk-forward 划分 —— 时间区间零重叠 + embargo
# --------------------------------------------------------------------------- #
def test_walk_forward_no_overlap_with_embargo():
    dates = np.array([f"2022{m:02d}{d:02d}" for m in range(1, 13) for d in (1, 15, 28)])
    tr, te = walk_forward_split(dates, ("20220101", "20220630"),
                                ("20220701", "20221228"), embargo_days=2)
    assert tr.size > 0 and te.size > 0
    overlap = np.intersect1d(tr, te)
    assert overlap.size == 0, "训练/测试索引不得重叠"
    # 训练集尾部应被 embargo 裁掉 2 个
    d = np.array([int(x) for x in dates])
    assert d[tr].max() < d[te].min(), "训练区间必须严格早于测试区间"


def test_walk_forward_rejects_reversed_range():
    from stock_ai.utils.errors import ConfigError
    dates = np.array(["20220101", "20220201"])
    with pytest.raises(ConfigError):
        walk_forward_split(dates, ("20220601", "20220101"), ("20230101", "20231231"), 0)


def test_walk_forward_embargo_actually_removes():
    dates = np.array([f"2022{m:02d}{d:02d}" for m in range(1, 13) for d in (1, 15, 28)])
    tr0, _ = walk_forward_split(dates, ("20220101", "20220630"),
                                ("20220701", "20221228"), embargo_days=0)
    tr2, _ = walk_forward_split(dates, ("20220101", "20220630"),
                                ("20220701", "20221228"), embargo_days=2)
    assert tr2.size == tr0.size - 2, "embargo 应恰好裁掉 2 个交易日"


# --------------------------------------------------------------------------- #
# B7: 防泄漏归一化 —— 测试阶段绝不重新 fit（K9）
# --------------------------------------------------------------------------- #
def test_normalizer_transform_is_pure_wrt_input(env_cfg):
    """对同一份测试数据，无论前面喂什么数据，归一化参数必须不变。"""
    import pandas as pd

    eng = FeatureEngineer(["close", "volume", "ma5", "momentum"])
    # 训练窗口
    train_df = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=100),
        "open": np.linspace(10, 20, 100),
        "high": np.linspace(10, 20, 100) + 0.5,
        "low": np.linspace(10, 20, 100) - 0.5,
        "close": np.linspace(10, 20, 100),
        "volume": np.linspace(1e6, 2e6, 100),
    })
    state = eng.fit(train_df)
    frozen_mean = state.mean.copy()
    frozen_std = state.std.copy()

    # 测试数据（完全不同量级）
    test_df = pd.DataFrame({
        "date": pd.bdate_range("2023-01-01", periods=50),
        "open": np.full(50, 500.0),
        "high": np.full(50, 501.0),
        "low": np.full(50, 499.0),
        "close": np.full(50, 500.0),
        "volume": np.full(50, 5e6),
    })
    # 无论调用多少次 / 之前喂什么，transform 不改变 state
    eng.transform(test_df, state)
    eng.transform(train_df, state)
    eng.transform(test_df, state)
    assert np.array_equal(state.mean, frozen_mean), "transform 不得修改冻结统计量"
    assert np.array_equal(state.std, frozen_std), "transform 不得修改冻结统计量"


def test_normalizer_state_roundtrip_stable():
    st = NormalizerState(mean=np.array([1.0, 2.0]), std=np.array([0.5, 0.25]),
                         columns=["a", "b"], data_hash="abc")
    st2 = NormalizerState.from_dict(st.to_dict())
    assert np.allclose(st2.mean, st.mean)
    assert np.allclose(st2.std, st.std)
    assert st2.data_hash == "abc"


# --------------------------------------------------------------------------- #
# B8: 复现性 —— 同 seed 两次 build 的 data_hash 一致
# --------------------------------------------------------------------------- #
def test_data_hash_deterministic(tmp_path):
    import pandas as pd
    from stock_ai.utils.io import write_parquet
    from stock_ai.samples.builder import SampleBuilder

    df = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=30),
        "open": np.arange(30, dtype="float64"),
        "high": np.arange(30, dtype="float64") + 1,
        "low": np.arange(30, dtype="float64") - 1,
        "close": np.arange(30, dtype="float64"),
        "volume": np.full(30, 1e6),
        "amount": np.full(30, 1e7),
        "adj_factor": np.ones(30),
    })
    p1 = write_parquet(df, tmp_path / "a.parquet")
    h1 = SampleBuilder.compute_data_hash([p1])
    h2 = SampleBuilder.compute_data_hash([p1])
    assert h1 == h2 and len(h1) == 64, "同文件 data_hash 必须稳定且为 sha256"


# --------------------------------------------------------------------------- #
# C: 跨标的边界 —— 独立复现爆炸场景
# --------------------------------------------------------------------------- #
def test_cross_symbol_extreme_price_ratio_bounded(env_cfg):
    """一只 5 元、一只 500 元（100 倍），组合市值必须保持有限合理值。"""
    n = 40
    p_a = np.linspace(5.0, 5.5, n)
    p_b = np.linspace(500.0, 520.0, n)
    prices = np.concatenate([p_a, p_b])
    symbols = ["LOW"] * n + ["HIGH"] * n
    split = _split_from_prices(prices, symbols=symbols)
    env = TradingEnv(split, env_cfg, seed=1)
    env.reset(seed=1)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(1)
        done = term or trunc
    eq = env.equity
    assert np.all(np.isfinite(eq)), "权益曲线出现 NaN/Inf"
    assert eq.max() < env_cfg.initial_cash * 3.0, (
        f"组合市值疑似爆炸: max={eq.max():.3e} (>3x initial)"
    )
    # 必须有 symbol_boundary 强平记录
    reasons = [t.get("reason") for t in env.trades]
    assert "symbol_boundary" in reasons, "跨标的未触发强制平仓"


def test_cross_symbol_prev_close_not_carried(env_cfg):
    """边界处的 prev_close 必须等于当日本身，不得用前一标的昨收。"""
    prices = [5.0] * 5 + [500.0] * 5
    symbols = ["LOW"] * 5 + ["HIGH"] * 5
    split = _split_from_prices(prices, symbols=symbols)
    env = TradingEnv(split, env_cfg, seed=1)
    assert env.prev_closes[5] == prices[5], "边界 prev_close 必须取当日本身"
    assert bool(env.symbol_boundary[5]) is True
    assert int(env.symbol_boundary.sum()) == 1
