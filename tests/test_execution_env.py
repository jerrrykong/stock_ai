"""需求 4：严格撮合环境（T 日信号 -> T+1 日真实开盘成交）测试。

覆盖：
1. **T+1 次日执行**：T 日信号在 T+1 日开盘价成交，不是同日收盘；
2. **真实开盘价**：成交价 = 当日真实开盘价 ± 滑点，且用未复权真实价（非 hfq 价）；
3. **涨跌停拒单**：一字板（开盘即涨停/跌停）买入/卖出被拒，含板块分段规则；
4. **T+1 冻结**：当日买入当日不可卖；
5. **磨损计算**：总换手、交易次数、成本拖累 bps、平均单笔磨损、与买入持有基准对比；
6. **无未来函数**：改写 T+1 及之后的行情不会改变 T 时刻的决策与成交；
7. 旧 :class:`TradingEnv` 行为未被改动；unified manifest 记录 ``execution_version``。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from stock_ai.config.schema import AppConfig, EnvConfig
from stock_ai.env.limits import (
    CHINEXT_FULL_FROM,
    board_of,
    is_blocked_at_open,
    limit_prices,
    price_limit,
)
from stock_ai.env.strict_env import (
    EXECUTION_VERSION,
    ExecutionBars,
    StrictEpisodePoolEnv,
    StrictExecutionEnv,
    friction_metrics,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def make_bars(n=30, start="2024-01-02", base=20.0, symbol="600519", open_offsets=None,
              closes=None, volumes=None):
    """构造未复权真实 K 线。"""
    dates = pd.bdate_range(start, periods=n).strftime("%Y%m%d").to_numpy()
    close = np.asarray(closes if closes is not None
                       else base * (1 + np.arange(n) * 0.001), dtype="float64")
    if open_offsets is None:
        opens = close * 1.001
    else:
        opens = np.asarray(open_offsets, dtype="float64")
    return ExecutionBars(
        dates=dates, open=opens, high=np.maximum(opens, close) * 1.01,
        low=np.minimum(opens, close) * 0.99, close=close,
        volume=np.asarray(volumes if volumes is not None else [1e6] * n, dtype="float64"),
        symbol=symbol,
    )


def make_obs(n, window=3, n_features=9):
    return np.zeros((n, window, n_features), dtype="float32")


@pytest.fixture
def env_cfg():
    return EnvConfig(initial_cash=1_000_000.0, commission_rate=0.00025,
                     stamp_tax_rate=0.0005, slippage_rate=0.001,
                     min_commission=5.0, lot_size=100)


# --------------------------------------------------------------------------- #
# 1. 涨跌停规则（板块 + 生效日期）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("symbol,board", [
    ("600519", "main"), ("601398", "main"), ("000001", "main"), ("002415", "main"),
    ("300750", "growth"), ("301029", "growth"), ("688981", "star"), ("689009", "star"),
    ("830799", "bse"), ("430047", "bse"),
])
def test_board_detection(symbol, board):
    assert board_of(symbol) == board


def test_price_limit_by_board_and_date():
    """主板 10%、创业板注册制前后分段、科创板 20%、北交所 30%。"""
    assert price_limit("600519", "20240102") == pytest.approx(0.10)
    assert price_limit("300750", "20200821") == pytest.approx(0.10)   # 注册制之前
    assert price_limit("300750", CHINEXT_FULL_FROM) == pytest.approx(0.20)
    assert price_limit("301029", "20210101") == pytest.approx(0.20)
    assert price_limit("688981", "20240102") == pytest.approx(0.20)
    assert price_limit("830799", "20240102") == pytest.approx(0.30)


def test_st_limit_only_affects_main_board():
    """ST 5% 仅作用于主板（创业板/科创板仍 20%、北交所 30%）。"""
    assert price_limit("600519", "20240102", st=True) == pytest.approx(0.05)
    assert price_limit("300750", "20240102", st=True) == pytest.approx(0.20)
    assert price_limit("688981", "20240102", st=True) == pytest.approx(0.20)
    assert price_limit("830799", "20240102", st=True) == pytest.approx(0.30)


def test_limit_prices_round_to_cent():
    """涨跌停价四舍五入到 0.01 元。"""
    up, down = limit_prices(10.03, "600519", "20240102")
    assert up == pytest.approx(11.03) and down == pytest.approx(9.03)
    assert limit_prices(0, "600519", "20240102") == (0.0, 0.0)


def test_is_blocked_at_open_detects_one_word_limit():
    """开盘即涨停则买不进；开盘即跌停则卖不出。"""
    assert is_blocked_at_open(11.0, 10.0, "600519", "20240102", "buy") is True
    assert is_blocked_at_open(9.0, 10.0, "600519", "20240102", "sell") is True
    assert is_blocked_at_open(10.5, 10.0, "600519", "20240102", "buy") is False
    assert is_blocked_at_open(10.5, 10.0, "600519", "20240102", "sell") is False


def test_growth_board_uses_20pct_limit_for_one_word():
    """创业板按 20% 判定一字板：11.9 不封板、12.0 才是涨停。"""
    assert is_blocked_at_open(11.9, 10.0, "300750", "20240102", "buy") is False
    assert is_blocked_at_open(12.0, 10.0, "300750", "20240102", "buy") is True


# --------------------------------------------------------------------------- #
# 2. T+1 次日执行 + 真实开盘价
# --------------------------------------------------------------------------- #
def test_signal_executes_at_next_day_open_not_same_day_close(env_cfg):
    """T 日信号在 **T+1 日开盘价** 成交：第 0 步的成交日是第 1 个交易日。"""
    opens = np.array([20.0, 30.0, 31.0, 32.0, 33.0])
    closes = np.array([99.0, 31.0, 32.0, 33.0, 34.0])  # T 日收盘价故意与开盘差异极大
    bars = make_bars(5, open_offsets=opens, closes=closes)
    env = StrictExecutionEnv(make_obs(5), bars, env_cfg)
    env.reset()
    _, _, _, _, info = env.step(1)
    trade = env.trades[0]
    assert trade["date"] == bars.dates[1], "成交日必须是 T+1"
    assert trade["signal_date"] == bars.dates[0], "信号日必须是 T"
    # 成交价 = T+1 开盘价 * (1 + 滑点)
    assert trade["price"] == pytest.approx(bars.open[1] * (1 + env_cfg.slippage_rate))
    # 绝不是 T 日收盘价（99）或 T+1 收盘价（31）
    assert trade["price"] != pytest.approx(99.0)
    assert trade["price"] != pytest.approx(31.0)
    assert info["execution_date"] == bars.dates[1]


def test_executes_with_unadjusted_real_price_not_hfq(env_cfg):
    """特征用后复权价，成交必须用未复权真实开盘价。"""
    bars = make_bars(6, base=5.0)
    # 特征张量全部写成后复权量纲（放大 100 倍），成交金额仍应基于 5 元真实价
    env = StrictExecutionEnv(np.full((6, 3, 9), 137.0, dtype="float32"), bars, env_cfg)
    env.reset()
    env.step(1)
    trade = env.trades[0]
    assert trade["open_price"] == pytest.approx(bars.open[1])
    assert trade["notional"] == pytest.approx(trade["shares"] * trade["price"])
    assert env.state.cash < env_cfg.initial_cash
    # 全仓买入后总市值应接近 100 万（真实价量纲），而非千万级
    assert env.state.portfolio_value < env_cfg.initial_cash * 1.05


def test_costs_use_real_notional_with_commission_and_slippage(env_cfg):
    """成交金额、佣金、滑点都基于真实未复权价计算。"""
    bars = make_bars(5, base=10.0)
    env = StrictExecutionEnv(make_obs(5), bars, env_cfg)
    env.reset()
    env.step(1)
    trade = env.trades[0]
    expected_notional = trade["shares"] * trade["price"]
    assert trade["notional"] == pytest.approx(expected_notional)
    assert trade["fee"] == pytest.approx(max(expected_notional * env_cfg.commission_rate,
                                             env_cfg.min_commission))
    assert trade["price"] == pytest.approx(bars.open[1] * 1.001)


def test_sell_uses_next_day_open_and_pays_stamp_tax(env_cfg):
    """卖出同样在 T+1 开盘执行，并计印花税。"""
    bars = make_bars(9, base=10.0)
    env = StrictExecutionEnv(make_obs(9), bars, env_cfg)
    env.reset()
    env.step(1)   # T0 信号 -> T1 买入
    env.step(0)   # T1 信号 -> T2 执行（持有）
    env.step(2)   # T2 信号 -> T3 卖出（T+1 已解除）
    sells = [t for t in env.trades if t["side"] == "sell"]
    assert sells and sells[0]["date"] == bars.dates[3]
    notional = sells[0]["notional"]
    expected = max(notional * env_cfg.commission_rate, env_cfg.min_commission) + \
        notional * env_cfg.stamp_tax_rate
    assert sells[0]["fee"] == pytest.approx(expected)
    assert sells[0]["price"] == pytest.approx(bars.open[3] * (1 - env_cfg.slippage_rate))


# --------------------------------------------------------------------------- #
# 3. 涨跌停拒单
# --------------------------------------------------------------------------- #
def test_buy_rejected_when_next_open_is_limit_up(env_cfg):
    """T+1 开盘一字涨停 -> 买不进。"""
    closes = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    opens = np.array([10.0, 11.0, 10.0, 10.0, 10.0])  # 第 2 天开盘 11.00 == 涨停价
    bars = make_bars(5, open_offsets=opens, closes=closes)
    env = StrictExecutionEnv(make_obs(5), bars, env_cfg)
    env.reset()
    _, _, _, _, info = env.step(1)
    assert info["trade_ok"] is False
    assert "涨停" in info["blocked_reason"]
    assert env.trades == [] and env.state.shares == 0


def test_sell_rejected_when_next_open_is_limit_down(env_cfg):
    """T+1 开盘一字跌停 -> 卖不出，持仓保留。"""
    closes = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    opens = np.array([10.0, 10.0, 10.0, 9.0, 10.0])  # 第 4 天开盘 9.00 == 跌停价
    bars = make_bars(5, open_offsets=opens, closes=closes)
    env = StrictExecutionEnv(make_obs(5), bars, env_cfg)
    env.reset()
    env.step(1)          # T0 -> T1 买入
    env.step(0)          # T1 -> T2 无操作
    _, _, _, _, info = env.step(2)   # T2 -> T3 跌停拒单
    assert info["execution_date"] == bars.dates[3]
    assert info["trade_ok"] is False and "跌停" in info["blocked_reason"]
    assert env.state.shares > 0
    assert not [t for t in env.trades if t["side"] == "sell"]


def test_growth_board_buy_allowed_at_10pct_but_not_at_20pct(env_cfg):
    """创业板 +10% 开盘可买，+20% 开盘（一字板）拒单。"""
    closes = np.full(5, 10.0)
    opens_ok = np.array([10.0, 11.0, 10.0, 10.0, 10.0])
    env_ok = StrictExecutionEnv(make_obs(5), make_bars(5, symbol="300750",
                                                      open_offsets=opens_ok, closes=closes),
                               env_cfg)
    env_ok.reset()
    _, _, _, _, ok_info = env_ok.step(1)
    assert ok_info["trade_ok"] is True

    opens_block = np.array([10.0, 12.0, 10.0, 10.0, 10.0])
    env_bad = StrictExecutionEnv(make_obs(5), make_bars(5, symbol="300750",
                                                        open_offsets=opens_block, closes=closes),
                                 env_cfg)
    env_bad.reset()
    _, _, _, _, bad_info = env_bad.step(1)
    assert bad_info["trade_ok"] is False and "涨停" in bad_info["blocked_reason"]


def test_halted_day_blocks_trading(env_cfg):
    """真实成交量为 0（停牌）时不可买卖。"""
    volumes = [1e6, 0.0, 1e6, 1e6, 1e6]
    bars = make_bars(5, volumes=volumes)
    env = StrictExecutionEnv(make_obs(5), bars, env_cfg)
    env.reset()
    _, _, _, _, info = env.step(1)
    assert info["trade_ok"] is False and info["blocked_reason"] == "停牌"


# --------------------------------------------------------------------------- #
# 4. T+1 冻结
# --------------------------------------------------------------------------- #
def test_cannot_sell_on_the_same_execution_day(env_cfg):
    """T+1 冻结：买入执行日当天不可卖出。

    本环境每个 step 都推进一个交易日，正常路径下"买/卖同一执行日"不会自然发生，
    因此这里直接构造 ``last_buy_idx >= exec_idx`` 的状态验证守卫仍然生效。
    """
    bars = make_bars(6, base=10.0)
    env = StrictExecutionEnv(make_obs(6), bars, env_cfg)
    env.reset()
    env.step(1)                                  # T0 信号 -> T1 买入
    assert env.state.last_buy_idx == 1
    env.state.last_buy_idx = env.idx + 1          # 人为制造"将在下一执行日才买入"
    _, _, _, _, info = env.step(2)                # 当前执行日卖出 -> 应被 T+1 拦下
    assert info["execution_date"] == bars.dates[env.idx]
    assert info["trade_ok"] is False and "T+1" in info["blocked_reason"]
    assert env.state.shares > 0
    assert not [t for t in env.trades if t["side"] == "sell"]


def test_sell_allowed_on_day_after_buy_execution(env_cfg):
    """买入执行日的**次日**即可卖出（T+1 规则成立）。"""
    bars = make_bars(8, base=10.0)
    env = StrictExecutionEnv(make_obs(8), bars, env_cfg)
    env.reset()
    env.step(1)          # T0 -> T1 买入
    _, _, _, _, info = env.step(2)   # T1 -> T2 卖出
    assert info["trade_ok"] is True
    sells = [t for t in env.trades if t["side"] == "sell"]
    assert len(sells) == 1 and sells[0]["date"] == bars.dates[2]


# --------------------------------------------------------------------------- #
# 5. 磨损指标
# --------------------------------------------------------------------------- #
def test_friction_metrics_fields_and_baseline():
    """输出总换手、交易次数、成本拖累 bps、平均单笔磨损与基准对比。"""
    trades = [
        {"notional": 1_000_000.0, "fee": 250.0, "side": "buy"},
        {"notional": 1_000_000.0, "fee": 750.0, "side": "sell"},  # 含印花税
    ]
    equity = np.array([1_000_000.0, 1_010_000.0, 1_009_000.0])
    benchmark = np.array([1_000_000.0, 1_020_000.0, 1_005_000.0])
    m = friction_metrics(trades, 1_000_000.0, equity, benchmark)
    assert m["trades"] == 2
    assert m["total_notional"] == pytest.approx(2_000_000.0)
    assert m["total_turnover_ratio"] == pytest.approx(2.0)
    assert m["total_fee"] == pytest.approx(1000.0)
    assert m["cost_drag_bps"] == pytest.approx(1000.0 / equity.mean() * 1e4)
    assert m["avg_wear_bps"] == pytest.approx((250.0 + 750.0) / 2 / 1e6 * 1e4)
    assert m["strategy_return"] == pytest.approx(0.009)
    assert m["benchmark_return"] == pytest.approx(0.005)
    assert m["excess_return"] == pytest.approx(0.004)


def test_friction_metrics_without_benchmark_has_none_excess():
    """没有基准时 excess 为 None（不编造基准收益）。"""
    m = friction_metrics([], 1000.0, np.array([1000.0, 1000.0]), None)
    assert m["trades"] == 0 and m["avg_wear_bps"] == 0.0
    assert m["benchmark_return"] is None and m["excess_return"] is None


def test_env_metrics_reports_real_costs_and_buy_hold(env_cfg):
    """环境级 ``metrics()`` 汇总真实费用并与买入持有对比。"""
    bars = make_bars(40, base=10.0)
    env = StrictExecutionEnv(make_obs(40), bars, env_cfg)
    env.reset()
    env.step(1)
    env.step(0)
    env.step(2)
    m = env.metrics()
    assert m["trades"] == len(env.trades) == 2
    assert m["total_fee"] == pytest.approx(sum(t["fee"] for t in env.trades))
    assert m["cost_drag_bps"] > 0
    assert m["avg_wear_bps"] > 0
    assert m["benchmark_return"] is not None
    assert m["excess_return"] == pytest.approx(m["strategy_return"] - m["benchmark_return"])


def test_buy_and_hold_uses_real_open_and_costs(env_cfg):
    """买入持有基准从次日真实开盘买入，且扣除买入费用，不超过初始资金。"""
    bars = make_bars(30, base=25.0)
    env = StrictExecutionEnv(make_obs(30), bars, env_cfg)
    env.reset()
    curve = env.buy_and_hold()
    assert len(curve) == 30
    assert np.isfinite(curve).all()
    assert curve.max() <= env_cfg.initial_cash * 1.05


# --------------------------------------------------------------------------- #
# 6. 无未来函数
# --------------------------------------------------------------------------- #
def test_future_bars_do_not_affect_signal_day_decision(env_cfg):
    """改写 T+1 及之后的收盘价，不改变 T 信号的成交价与成交日。"""
    base_closes = np.linspace(10.0, 12.0, 12)
    opens = base_closes * 1.001
    env_a = StrictExecutionEnv(make_obs(12), make_bars(12, open_offsets=opens,
                                                        closes=base_closes), env_cfg)
    env_a.reset()
    _, _, _, _, info_a = env_a.step(1)
    trade_a = dict(env_a.trades[0])

    tampered = base_closes.copy()
    tampered[1:] *= 50.0        # 从 T+1 开始全部篡改
    env_b = StrictExecutionEnv(make_obs(12), make_bars(12, open_offsets=opens,
                                                        closes=tampered), env_cfg)
    env_b.reset()
    _, _, _, _, info_b = env_b.step(1)
    trade_b = dict(env_b.trades[0])

    assert info_a["execution_date"] == info_b["execution_date"]
    assert trade_a["date"] == trade_b["date"]
    assert trade_a["price"] == pytest.approx(trade_b["price"])
    assert trade_a["shares"] == trade_b["shares"]
    assert trade_a["notional"] == pytest.approx(trade_b["notional"])


def test_benchmark_return_is_close_to_close_and_causal(env_cfg):
    """``info`` 新增 ``benchmark_return`` / ``step_excess``，且只用 T、T+1 两日收盘。"""
    base_closes = np.linspace(10.0, 12.0, 12)
    env_a = StrictExecutionEnv(make_obs(12), make_bars(12, base=10.0, closes=base_closes),
                               env_cfg)
    env_a.reset()
    _, reward_a, _, _, info_a = env_a.step(0)
    assert "benchmark_return" in info_a and "step_excess" in info_a
    assert info_a["benchmark_return"] == pytest.approx(base_closes[1] / base_closes[0] - 1.0)

    tampered = base_closes.copy()
    tampered[2:] *= 99.0        # 只篡改 t+2 及以后
    env_b = StrictExecutionEnv(make_obs(12), make_bars(12, base=10.0, closes=tampered),
                               env_cfg)
    env_b.reset()
    _, reward_b, _, _, info_b = env_b.step(0)
    assert info_a["benchmark_return"] == pytest.approx(info_b["benchmark_return"])
    assert reward_a == pytest.approx(reward_b)


def test_benchmark_step_return_guards_bad_denominator(env_cfg):
    """分母 ≤1e-9 或非有限时基准收益取 0，不产生 NaN/Inf。"""
    bars = make_bars(5, base=10.0)
    bars.close[0] = 0.0
    env = StrictExecutionEnv(make_obs(5), bars, env_cfg)
    assert env._benchmark_step_return(0, 1) == 0.0
    bars.close[0] = float("nan")
    assert env._benchmark_step_return(0, 1) == 0.0


def test_signal_observation_contains_no_future_features(env_cfg):
    """第 t 步返回的观测来自第 t 天，不含 t+1 及以后的信息。"""
    bars = make_bars(10, base=10.0)
    obs_input = np.arange(10, dtype="float32").reshape(10, 1, 1) * np.ones((1, 3, 9), dtype="float32")
    env = StrictExecutionEnv(obs_input, bars, env_cfg)
    first, _ = env.reset()
    np.testing.assert_array_equal(first, obs_input[0])
    second, _, _, _, _ = env.step(0)
    np.testing.assert_array_equal(second, obs_input[1])
    assert not np.array_equal(second, obs_input[2])


def test_step_after_terminated_raises(env_cfg):
    """episode 结束后必须先 reset，不能越界读取未来 K 线。"""
    bars = make_bars(2, base=10.0)
    env = StrictExecutionEnv(make_obs(2), bars, env_cfg)
    env.reset()
    _, _, terminated, _, _ = env.step(0)
    assert terminated is True
    with pytest.raises(RuntimeError):
        env.step(0)


# --------------------------------------------------------------------------- #
# 7. gym 契约 & 池化 & 旧环境不变
# --------------------------------------------------------------------------- #
def test_check_env_passes_for_strict_and_pool(env_cfg):
    """通过 gymnasium ``check_env``。"""
    bars = make_bars(20)
    StrictExecutionEnv(make_obs(20), bars, env_cfg).close()
    pool = StrictEpisodePoolEnv([("600519_2024", make_obs(20), bars)], env_cfg)
    check_env(pool, skip_render_check=True)
    ids = {pool.reset()[1]["episode_id"] for _ in range(10)}
    assert ids == {"600519_2024"}
    pool.close()


def test_mismatched_obs_bars_length_rejected(env_cfg):
    """特征长度与真实 K 线长度不一致时立即报错（防错位撮合）。"""
    with pytest.raises(ValueError, match="长度不一致"):
        StrictExecutionEnv(make_obs(10), make_bars(8), env_cfg)
    with pytest.raises(ValueError, match="至少 2 根"):
        StrictExecutionEnv(make_obs(1), make_bars(1), env_cfg)


def test_old_trading_env_semantics_unchanged(env_cfg):
    """旧 TradingEnv 仍是"同日收盘模拟成交"，未被严格撮合改动。"""
    from stock_ai.env.trading_env import TradingEnv
    from stock_ai.samples.schema import SampleSplit

    n = 10
    dates = pd.bdate_range("2024-01-02", periods=n).strftime("%Y%m%d").to_numpy()
    sample = SampleSplit(obs=np.zeros((n, 3, 9), dtype="float32"), dates=dates,
                         symbols=np.repeat("600519", n),
                         prices=np.linspace(10.0, 11.0, n))
    env = TradingEnv(sample, env_cfg)
    env.reset()
    env.step(1)
    trade = env.trades[0]
    assert trade["date"] == dates[0], "旧环境仍是首日收盘成交"
    assert trade["price"] == pytest.approx(sample.prices[0] * 1.001)


# --------------------------------------------------------------------------- #
# 8. unified 严格撮合接线 & manifest
# --------------------------------------------------------------------------- #
def _strict_workspace(tmp_path):
    """构造 特征目录(data/raw_hfq) + 不复权目录(data/raw) 各一只股票。"""
    from stock_ai.scale.unified import build_dataset

    dates = pd.DatetimeIndex([])
    for year in (2021, 2022, 2023, 2024, 2025):
        dates = dates.append(pd.bdate_range(f"{year}-01-04", periods=12))
    n = len(dates)
    hfq_close = 10 + np.arange(n) * 0.03
    raw_close = hfq_close / 7.5                     # 未复权真实价（后复权放大 7.5 倍）
    hfq = pd.DataFrame({"date": dates, "open": hfq_close, "high": hfq_close * 1.01,
                        "low": hfq_close * 0.99, "close": hfq_close,
                        "volume": 1000.0 + np.arange(n)})
    raw = pd.DataFrame({"date": dates, "open": raw_close, "high": raw_close * 1.01,
                        "low": raw_close * 0.99, "close": raw_close,
                        "volume": 1000.0 + np.arange(n)})
    feature_dir = tmp_path / "data" / "raw_hfq"
    raw_dir = tmp_path / "data" / "raw"
    feature_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    hfq.to_parquet(feature_dir / "600519.parquet", index=False)
    raw.to_parquet(raw_dir / "600519.parquet", index=False)
    return build_dataset, feature_dir, raw_dir


def test_build_dataset_strict_uses_raw_bars_for_execution(tmp_path):
    """严格撮合模式下 episode 携带的是不复权真实 K 线（价格量纲为 raw）。"""
    build_dataset, feature_dir, raw_dir = _strict_workspace(tmp_path)
    data = build_dataset(feature_dir, 3, execution_dir=raw_dir,
                         execution_version="strict-t1-open-v1")
    assert data.execution_version == "strict-t1-open-v1"
    assert data.split["strict_episodes"]["train"] == len(data.strict_pools["train"]) >= 1
    uid, obs, bars = data.strict_pools["train"][0]
    assert obs.shape[0] == bars.n_steps
    raw_df = pd.read_parquet(raw_dir / "600519.parquet")
    assert bars.close[0] == pytest.approx(raw_df["close"].iloc[0])
    # 特征来源是后复权（量纲不同），确认两套价格确实不同
    hfq_df = pd.read_parquet(feature_dir / "600519.parquet")
    assert bars.close[0] != pytest.approx(hfq_df["close"].iloc[0])


def test_build_dataset_legacy_mode_unchanged(tmp_path):
    """旧模式不传 execution_dir 时行为与从前一致（无 strict episode）。"""
    build_dataset, feature_dir, _raw_dir = _strict_workspace(tmp_path)
    data = build_dataset(feature_dir, 3)
    assert data.execution_version == "legacy-same-day-close-v1"
    assert data.strict_pools == {"train": [], "val": [], "test": []}
    assert data.split["strict_episodes"] == {"train": 0, "val": 0, "test": 0}


def test_build_dataset_rejects_unknown_execution_version(tmp_path):
    """未知 execution_version 明确报错。"""
    build_dataset, feature_dir, raw_dir = _strict_workspace(tmp_path)
    with pytest.raises(Exception, match="execution_version"):
        build_dataset(feature_dir, 3, execution_dir=raw_dir, execution_version="v9")


def test_strict_requires_execution_dir(tmp_path):
    """声明 strict 却不给不复权目录时明确报错（防用复权价撮合）。"""
    build_dataset, feature_dir, _ = _strict_workspace(tmp_path)
    with pytest.raises(Exception, match="execution_dir"):
        build_dataset(feature_dir, 3, execution_version="strict-t1-open-v1")


def test_manifest_records_execution_version(tmp_path):
    """unified manifest 记录 execution_version / feature_dir / execution_dir。"""
    pytest.importorskip("stable_baselines3")
    from stock_ai.config.loader import ConfigLoader
    from stock_ai.scale.unified import EXECUTION_STRICT, UnifiedRunner

    cfg = AppConfig()
    cfg.paths.root = tmp_path
    ConfigLoader.resolve_paths(cfg)
    cfg.data.cache_dir = tmp_path / "data" / "raw"
    _build_dataset, feature_dir, raw_dir = _strict_workspace(tmp_path)
    cfg.sample.window = 3
    cfg.train.n_steps = 8
    cfg.train.batch_size = 8
    cfg.train.n_epochs = 1
    directory = UnifiedRunner(cfg).run(8, "strict_smoke", n_envs=1, checkpoint_steps=8,
                                       execution_version=EXECUTION_STRICT,
                                       feature_dir=str(feature_dir))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_version"] == EXECUTION_STRICT
    assert manifest["feature_dir"] == str(Path(feature_dir).resolve())
    assert manifest["execution_dir"] == str(Path(raw_dir).resolve())
    assert manifest["feature_adjust"] == "hfq"
    assert "未复权" in manifest["warning"]
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["execution_version"] == EXECUTION_STRICT
    for split in ("val", "test"):
        assert metrics["evaluation"][split]["episodes"][0]["execution_version"] == EXECUTION_STRICT
        assert "friction" in metrics["evaluation"][split]
        friction = metrics["evaluation"][split]["friction"]
        assert {"total_trades", "total_notional", "mean_cost_drag_bps",
                "mean_avg_wear_bps", "mean_strategy_return",
                "mean_benchmark_return"} <= set(friction)
        # 零交易单列 + 退化守卫（新字段不得破坏旧字段）
        assert {"zero_trade_episodes", "zero_trade_fraction", "mean_excess_return_traded",
                "mean_excess_return_zero_trade", "degenerate",
                "degenerate_reason"} <= set(friction)
        assert 0.0 <= friction["zero_trade_fraction"] <= 1.0
        # 占比超过 5% 时必须标记退化并给出理由；反之必须不标记
        over = friction["zero_trade_fraction"] > 0.05
        assert friction["degenerate"] is over
        assert (friction["degenerate_reason"] is not None) is over
        # 分口径一致性：有交易 + 零交易 == 全部；各自口径与计数自洽
        n_traded = friction["episodes"] - friction["zero_trade_episodes"]
        if friction["zero_trade_episodes"] == 0:
            assert friction["mean_excess_return_zero_trade"] is None
        if n_traded == 0:
            assert friction["mean_excess_return_traded"] is None
        if friction["mean_excess_return"] is not None and n_traded and friction["zero_trade_episodes"]:
            lo = min(friction["mean_excess_return_traded"], friction["mean_excess_return_zero_trade"])
            hi = max(friction["mean_excess_return_traded"], friction["mean_excess_return_zero_trade"])
            assert lo - 1e-12 <= friction["mean_excess_return"] <= hi + 1e-12
    assert (directory / "model.zip").is_file()


def test_cli_execution_flag_and_feature_dir(tmp_path):
    """``--execution strict --feature-dir`` 出现在 CLI 上。"""
    from click.testing import CliRunner

    from stock_ai.cli.scale_cmd import unified_train

    result = CliRunner().invoke(unified_train, ["--help"])
    assert result.exit_code == 0
    assert "--execution" in result.output and "--feature-dir" in result.output


def test_zero_trade_excess_is_definitional_artifact():
    """零交易 episode 的「超额」恒等于 −benchmark_return，是定义性假象。

    纯数学验证，不依赖训练：全现金权益曲线为水平线，friction_metrics 的
    excess_return = strategy − benchmark = 0 − benchmark。因此一个永不做交易的
    「策略」在基准下跌的年份会被记成正超额——必须单列，不能与有交易口径混算。
    """
    from stock_ai.env.strict_env import friction_metrics

    initial = 1_000_000.0
    closes = np.linspace(10.0, 6.0, 60)          # 基准全年下跌 40%
    equity = np.full(60, initial)                 # 全现金：权益水平线
    bench = initial * (closes / closes[0])        # 恒买基准
    m = friction_metrics([], initial, equity, bench)
    assert m["excess_return"] == pytest.approx(0.0 - (closes[-1] / closes[0] - 1.0))
    assert m["excess_return"] > 0.0               # 躺平在熊市被记成「跑赢」
    # 同样的全现金，在基准上涨年份会被记成负超额
    closes_up = np.linspace(10.0, 14.0, 60)
    bench_up = initial * (closes_up / closes_up[0])
    m_up = friction_metrics([], initial, equity, bench_up)
    assert m_up["excess_return"] == pytest.approx(0.0 - (closes_up[-1] / closes_up[0] - 1.0))
    assert m_up["excess_return"] < 0.0
    # 同一「策略」（不交易）在两个方向上都只是基准的镜像，不含任何技能
    assert m["excess_return"] + m_up["excess_return"] == pytest.approx(
        -(closes[-1] / closes[0] - 1.0) - (closes_up[-1] / closes_up[0] - 1.0))
