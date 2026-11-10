"""QA 独立对抗性验证（*由 QA/严过关 编写，不复用工程师断言*）。

目的：证明「奖励改造」（excess 超额奖励 + 换手惩罚 + 无前视 + 旧路径零回归）
确实正确，而非盖章。所有期望值均由本文件独立手算/独立推导。

覆盖：
A. 无未来函数（最强）：篡改 t+2 及以后**全部** OHLCV，第 t 步 reward/benchmark 必须逐位不变；
B. 窗口对齐：策略步收益与基准必须都落在同一 signal→exec 的 close-to-close 区间；
C. excess 逐步数值正确（含 dd / turnover / leverage 惩罚）；
D. 换手/仓位惩罚严格单调（excess 与 drawdown_penalty 都验证）；
E. 旧类型零回归：默认值不变 + gross/drawdown_penalty/sharpe 与文档公式**逐位一致**；
F. CLI 覆盖规则（含大小写归一、非 excess 不设预设、非法值明确报错）；
G. 边界与错误路径（close=0/负、极大换手、负仓位、非有限、bench=0 退化）；
H. resume 兼容（旧 manifest 无 reward_type → 按 drawdown_penalty；不一致 → 明确拒绝）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from stock_ai.config.schema import AppConfig, EnvConfig
from stock_ai.env.rewards import SUPPORTED_REWARDS, RewardCalculator
from stock_ai.env.strict_env import ExecutionBars, StrictExecutionEnv

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _bars(closes, opens=None, highs=None, lows=None, volumes=None, symbol="600519"):
    n = len(closes)
    dates = pd.bdate_range("2024-01-02", periods=n).strftime("%Y%m%d").to_numpy()
    close = np.asarray(closes, dtype="float64")
    opens = close * 1.001 if opens is None else np.asarray(opens, dtype="float64")
    highs = np.maximum(opens, close) * 1.01 if highs is None else np.asarray(highs, dtype="float64")
    lows = np.minimum(opens, close) * 0.99 if lows is None else np.asarray(lows, dtype="float64")
    vols = np.full(n, 1e6) if volumes is None else np.asarray(volumes, dtype="float64")
    return ExecutionBars(dates=dates, open=opens, high=highs, low=lows, close=close,
                         volume=vols, symbol=symbol)


def _obs(n, window=3, nf=9):
    return np.zeros((n, window, nf), dtype="float32")


def _cfg(**kw):
    base = dict(initial_cash=1_000_000.0, commission_rate=0.00025, stamp_tax_rate=0.0005,
                slippage_rate=0.001, min_commission=5.0, lot_size=100)
    base.update(kw)
    return EnvConfig(**base)


# =========================================================================== #
# A. 无未来函数（最强对抗：篡改 t+2 及以后的全部 OHLCV）
# =========================================================================== #
@pytest.mark.parametrize("mangle", ["x50", "random"])
def test_no_lookahead_tamper_all_ohlcv_from_t2(mangle):
    """第 0 步（signal_idx=0, exec_idx=1）只应依赖 idx∈{0,1}；篡改 idx>=2 全部无效。"""
    n = 15
    base = np.linspace(10.0, 12.0, n)
    opens = base * 1.001
    highs = np.maximum(opens, base) * 1.01
    lows = np.minimum(opens, base) * 0.99
    vols = np.full(n, 1e6)
    ec = _cfg(reward_type="excess", lambda_drawdown=0.0, lambda_turnover=0.002,
              lambda_leverage=0.01)

    env_a = StrictExecutionEnv(_obs(n), _bars(base, opens, highs, lows, vols), ec)
    env_a.reset()
    _, reward_a, _, _, info_a = env_a.step(1)  # 买入

    rng = np.random.default_rng(12345)
    cl, op, hi, lo, vo = base.copy(), opens.copy(), highs.copy(), lows.copy(), vols.copy()
    if mangle == "x50":
        for arr in (cl, op, hi, lo):
            arr[2:] *= 50.0
        vo[2:] *= 50.0
    else:
        for arr in (cl, op, hi, lo):
            arr[2:] = rng.uniform(0.1, 1e6, n - 2)
        vo[2:] = rng.uniform(0.0, 1e7, n - 2)

    env_b = StrictExecutionEnv(_obs(n), _bars(cl, op, hi, lo, vo), ec)
    env_b.reset()
    _, reward_b, _, _, info_b = env_b.step(1)

    # 第 0 步基准只由 close[0]、close[1] 决定 -> 逐位相等
    assert info_a["benchmark_return"] == info_b["benchmark_return"]
    assert info_a["benchmark_return"] == pytest.approx(base[1] / base[0] - 1.0)
    assert reward_a == reward_b
    assert np.isfinite(reward_a)
    # 成交价同样不受影响（只用 open[1]）
    assert env_a.trades[0]["price"] == pytest.approx(env_b.trades[0]["price"])


# =========================================================================== #
# B. 窗口对齐：策略估值区间 == 基准区间（皆为 close[signal]→close[exec]）
# =========================================================================== #
def test_strategy_and_benchmark_share_same_close_window():
    """若基准用 T→T+1 而策略用 T→T+2（窗口错位），本测试会失败。"""
    # 缓慢上涨，避免触发一字涨停（否则买入被拒，无持仓）
    closes = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5])
    ec = _cfg(reward_type="excess", lambda_drawdown=0.0, lambda_turnover=0.0,
              lambda_leverage=0.0)
    env = StrictExecutionEnv(_obs(6), _bars(closes), ec)
    env.reset()
    env.step(1)                                   # T0 信号 -> T1 成交；idx 变为 1
    shares, cash = int(env.state.shares), float(env.state.cash)
    assert shares > 0

    _, reward, _, _, info = env.step(0)           # signal_idx=1 -> exec_idx=2
    # 策略步收益应基于 close[1] -> close[2]
    prev = cash + shares * closes[1]
    val = cash + shares * closes[2]
    ret = (val - prev) / prev
    bench = closes[2] / closes[1] - 1.0

    assert info["benchmark_return"] == pytest.approx(bench)
    assert info["step_excess"] == pytest.approx(ret - bench)
    assert reward == pytest.approx(ret - bench)   # λ 全 0
    # 反向哨兵：绝不能是 T+1->T+2 或 T->T+1 的错位窗口
    assert info["benchmark_return"] != pytest.approx(closes[3] / closes[2] - 1.0)
    assert info["benchmark_return"] != pytest.approx(closes[1] / closes[0] - 1.0)


def test_env_rejects_step_after_terminated():
    """episode 结束后 step 必须报错，杜绝越界读取未来 K 线。"""
    env = StrictExecutionEnv(_obs(2), _bars([10.0, 11.0]), _cfg())
    env.reset()
    _, _, terminated, _, _ = env.step(0)
    assert terminated is True
    with pytest.raises(RuntimeError):
        env.step(0)


# =========================================================================== #
# C. excess 逐步数值正确（独立手算，含全部三项惩罚与回撤峰值状态）
# =========================================================================== #
def test_excess_multi_step_hand_calc_all_terms():
    lam_dd, lam_t, lam_lev = 0.2, 0.003, 0.05
    ec = _cfg(reward_type="excess", lambda_drawdown=lam_dd, lambda_turnover=lam_t,
              lambda_leverage=lam_lev)
    rc = RewardCalculator(ec)
    rc.reset(1_000_000.0)

    seq = [
        # (prev, value, benchmark, turnover, position)
        (1_000_000.0, 1_050_000.0, 0.03, 0.10, 0.9),   # 上涨，创新高
        (1_050_000.0, 980_000.0, -0.01, 0.00, 0.9),    # 回落 -> 有回撤
        (980_000.0, 1_120_000.0, 0.02, 0.20, 1.0),     # 反弹，仍低于峰值 -> 回撤
    ]
    peak = 1_000_000.0
    for prev, value, bench, turn, pos in seq:
        peak = max(peak, value)
        dd = max(0.0, (peak - value) / peak) if peak > 1e-9 else 0.0
        ret = (value - prev) / prev if prev > 1e-9 else 0.0
        expected = (ret - bench) - lam_dd * dd - lam_t * turn - lam_lev * abs(pos)
        got = rc.compute(prev, value, turnover=turn, position_ratio=pos,
                         benchmark_return=bench)
        assert got == pytest.approx(expected, rel=0.0, abs=1e-15), (prev, value, dd)


def test_excess_no_penalty_all_zero_lambdas():
    """λ 全 0 时 excess 就是纯超额收益。"""
    ec = _cfg(reward_type="excess", lambda_drawdown=0.0, lambda_turnover=0.0,
              lambda_leverage=0.0)
    rc = RewardCalculator(ec)
    rc.reset(1_000_000.0)
    ret = 1_010_000.0 / 1_000_000.0 - 1.0
    assert rc.compute(1_000_000.0, 1_010_000.0, benchmark_return=0.005) == pytest.approx(ret - 0.005)


# =========================================================================== #
# D. 惩罚严格单调
# =========================================================================== #
@pytest.mark.parametrize("rtype", ["excess", "drawdown_penalty"])
def test_turnover_strictly_monotonic(rtype):
    ec = _cfg(reward_type=rtype, lambda_drawdown=0.0, lambda_turnover=0.01,
              lambda_leverage=0.0)
    rc = RewardCalculator(ec)
    rc.reset(1_000_000.0)
    rewards = [rc.compute(1_000_000.0, 1_000_000.0, turnover=t, position_ratio=0.0,
                          benchmark_return=0.0) for t in (0.0, 0.3, 0.7, 1.0, 3.0)]
    assert all(a > b for a, b in zip(rewards, rewards[1:])), rewards
    # 期望值断言（独立手算）
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[-1] == pytest.approx(-0.01 * 3.0)


@pytest.mark.parametrize("rtype", ["excess", "drawdown_penalty"])
def test_position_penalty_strictly_monotonic(rtype):
    ec = _cfg(reward_type=rtype, lambda_drawdown=0.0, lambda_turnover=0.0,
              lambda_leverage=0.05)
    rc = RewardCalculator(ec)
    rc.reset(1_000_000.0)
    rewards = [rc.compute(1_000_000.0, 1_000_000.0, turnover=0.0, position_ratio=p,
                          benchmark_return=0.0) for p in (0.0, 0.5, 1.0, 2.0)]
    assert all(a > b for a, b in zip(rewards, rewards[1:])), rewards
    assert rewards[-1] == pytest.approx(-0.05 * 2.0)


# =========================================================================== #
# E. 旧类型零回归：默认值 + 文档公式逐位一致
# =========================================================================== #
def test_env_config_defaults_unchanged():
    c = EnvConfig()
    assert c.reward_type == "drawdown_penalty"
    assert c.lambda_drawdown == 0.1
    assert c.lambda_turnover == 0.001
    assert c.lambda_leverage == 0.01
    assert c.commission_rate == 0.00025 and c.stamp_tax_rate == 0.0005
    assert c.slippage_rate == 0.001 and c.min_commission == 5.0 and c.lot_size == 100
    assert SUPPORTED_REWARDS == frozenset({"gross", "drawdown_penalty", "sharpe", "excess"})


def test_legacy_types_bit_identical_to_documented_formula():
    """逐位一致（`==`）：gross / drawdown_penalty / sharpe 未被本次改动影响。"""
    # gross = ret
    ec = _cfg(reward_type="gross")
    rc = RewardCalculator(ec)
    rc.reset(1_000_000.0)
    ret = (1_020_000.0 - 1_000_000.0) / 1_000_000.0
    assert rc.compute(1_000_000.0, 1_020_000.0, turnover=0.5, position_ratio=0.8) == ret

    # drawdown_penalty = ret - λdd·dd - λt·turn - λlev·|pos|（含峰值状态）
    ec = _cfg(reward_type="drawdown_penalty", lambda_drawdown=0.1, lambda_turnover=0.002,
              lambda_leverage=0.01)
    rc = RewardCalculator(ec)
    rc.reset(1_000_000.0)
    rc.compute(1_000_000.0, 1_050_000.0, turnover=0.0, position_ratio=1.0)  # peak=1.05M
    peak, v, pv = 1_050_000.0, 1_000_000.0, 1_050_000.0
    ret = (v - pv) / pv
    dd = (peak - v) / peak
    turn, pos = 0.02, 0.8
    expected = ret - 0.1 * dd - 0.002 * turn - 0.01 * abs(pos)
    assert rc.compute(pv, v, turnover=turn, position_ratio=pos) == expected

    # sharpe = ret/(|ret|+1e-6) - λt·turn - λlev·|pos|
    ec = _cfg(reward_type="sharpe", lambda_turnover=0.002, lambda_leverage=0.01)
    rc = RewardCalculator(ec)
    rc.reset(1_000_000.0)
    ret = (980_000.0 - 1_000_000.0) / 1_000_000.0
    base = ret / (abs(ret) + 1e-6)
    expected = base - 0.002 * 0.3 - 0.01 * abs(0.7)
    assert rc.compute(1_000_000.0, 980_000.0, turnover=0.3, position_ratio=0.7) == expected


def test_legacy_types_ignore_benchmark_kwarg_everywhere():
    for rtype in ("gross", "drawdown_penalty", "sharpe"):
        ec = _cfg(reward_type=rtype, lambda_drawdown=0.1, lambda_turnover=0.002,
                  lambda_leverage=0.01)
        a = RewardCalculator(ec)
        b = RewardCalculator(ec)
        a.reset(1_000_000.0)
        b.reset(1_000_000.0)
        kw = dict(prev_value=1_000_000.0, value=1_030_000.0, turnover=0.4, position_ratio=0.6)
        assert a.compute(**kw) == b.compute(**kw, benchmark_return=0.11)


# =========================================================================== #
# F. CLI 覆盖规则
# =========================================================================== #
class _FakeRunner:
    captured: dict = {}

    def __init__(self, cfg):
        _FakeRunner.captured["cfg"] = cfg

    def run(self, *args, **kwargs):
        return "fake-run-dir"


@pytest.fixture
def cli_capture(monkeypatch):
    import stock_ai.scale.unified as unified

    _FakeRunner.captured = {}
    monkeypatch.setattr(unified, "UnifiedRunner", _FakeRunner)
    return _FakeRunner


def _invoke(args):
    from stock_ai.cli.scale_cmd import unified_train

    return CliRunner().invoke(unified_train, ["--config", str(DEFAULT_CONFIG)] + list(args))


def test_cli_no_flags_env_identical_to_yaml(cli_capture):
    """不传任何新参数：cfg.env 与改动前的 YAML 配置完全一致（逐字段）。"""
    res = _invoke([])
    assert res.exit_code == 0, res.output
    got = cli_capture.captured["cfg"].env
    expected = AppConfig.from_yaml(DEFAULT_CONFIG).env
    from dataclasses import asdict
    assert asdict(got) == asdict(expected)


def test_cli_reward_excess_presets(cli_capture):
    res = _invoke(["--reward", "excess"])
    assert res.exit_code == 0, res.output
    env = cli_capture.captured["cfg"].env
    assert env.reward_type == "excess"
    assert env.lambda_turnover == 0.002      # 预设
    assert env.lambda_drawdown == 0.0        # 预设
    # 持仓惩罚纳入预设：long-only 全仓下按步收会变成持仓时长税，
    # 量级为年基准波幅的 5.6 倍，令「永不买入」成为全局最优
    assert env.lambda_leverage == 0.0        # 预设


def test_cli_reward_uppercase_normalized(cli_capture):
    res = _invoke(["--reward", "Excess"])
    assert res.exit_code == 0, res.output
    env = cli_capture.captured["cfg"].env
    assert env.reward_type == "excess" and env.lambda_turnover == 0.002


def test_cli_explicit_lambda_overrides_preset(cli_capture):
    res = _invoke(["--reward", "excess", "--lambda-turnover", "0.003"])
    assert res.exit_code == 0, res.output
    env = cli_capture.captured["cfg"].env
    assert env.reward_type == "excess"
    assert env.lambda_turnover == 0.003      # 显式 > 预设
    assert env.lambda_drawdown == 0.0        # 未显式 -> 预设
    assert env.lambda_leverage == 0.0        # 未显式 -> 预设


def test_cli_lambda_only_without_reward_no_preset(cli_capture):
    res = _invoke(["--lambda-turnover", "0.007"])
    assert res.exit_code == 0, res.output
    env = cli_capture.captured["cfg"].env
    assert env.reward_type == "drawdown_penalty"   # 未给 --reward 不改
    assert env.lambda_turnover == 0.007
    assert env.lambda_drawdown == 0.1 and env.lambda_leverage == 0.01


def test_cli_reward_gross_no_presets(cli_capture):
    res = _invoke(["--reward", "gross"])
    assert res.exit_code == 0, res.output
    env = cli_capture.captured["cfg"].env
    assert env.reward_type == "gross"
    assert env.lambda_turnover == 0.001 and env.lambda_drawdown == 0.1  # 非 excess 不设预设


def test_cli_invalid_reward_rejected_not_silent(cli_capture):
    from stock_ai.scale.errors import ScaleError

    res = _invoke(["--reward", "foo"])
    assert res.exit_code != 0
    assert isinstance(res.exception, ScaleError)
    assert "不支持的奖励类型" in str(res.exception)
    assert cli_capture.captured == {}   # 非法值必须在训练前被拦下


# =========================================================================== #
# G. 边界与错误路径
# =========================================================================== #
def test_excess_degenerate_inputs_never_nan_inf():
    ec = _cfg(reward_type="excess")
    rc = RewardCalculator(ec)
    cases = [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1_000_000.0, 1_000_000.0, float("nan"), 0.0, 0.0),
        (1_000_000.0, float("inf"), 0.01, 0.0, 0.0),
        (1_000_000.0, 1_010_000.0, 0.02, float("inf"), float("nan")),
        (1_000_000.0, -500_000.0, -0.01, 0.0, -0.5),   # 负仓位 / 负市值
        (1_000_000.0, 1_000_000.0, float("inf"), float("inf"), float("-inf")),
    ]
    for prev, value, bench, turn, pos in cases:
        rc.reset(1_000_000.0)
        r = rc.compute(prev, value, turnover=turn, position_ratio=pos, benchmark_return=bench)
        assert np.isfinite(r), (prev, value, bench, turn, pos, r)


def test_excess_bench_zero_degenerates_to_abs_return_minus_penalty():
    """非 strict（bench=0）时 excess 退化为 ret − 惩罚项，且与 drawdown_penalty 同值。"""
    kw = dict(lambda_drawdown=0.1, lambda_turnover=0.002, lambda_leverage=0.01)
    ec_ex = _cfg(reward_type="excess", **kw)
    ec_dd = _cfg(reward_type="drawdown_penalty", **kw)
    r_ex = RewardCalculator(ec_ex)
    r_dd = RewardCalculator(ec_dd)
    r_ex.reset(1_000_000.0)
    r_dd.reset(1_000_000.0)
    r_ex.compute(1_000_000.0, 1_050_000.0, turnover=0.0, position_ratio=1.0, benchmark_return=0.0)
    r_dd.compute(1_000_000.0, 1_050_000.0, turnover=0.0, position_ratio=1.0)
    a = r_ex.compute(1_050_000.0, 1_010_000.0, turnover=0.05, position_ratio=0.8,
                     benchmark_return=0.0)
    b = r_dd.compute(1_050_000.0, 1_010_000.0, turnover=0.05, position_ratio=0.8)
    assert a == b


def test_benchmark_step_return_edge_cases():
    ec = _cfg()
    base = [10.0, 11.0, 12.0, 13.0]
    env = StrictExecutionEnv(_obs(4), _bars(base), ec)
    assert env._benchmark_step_return(0, 1) == pytest.approx(11.0 / 10.0 - 1.0)
    # 分母 <=1e-9 / 非有限 -> 0
    for bad in (0.0, -5.0, float("nan")):
        b = _bars(base)
        b.close[0] = bad
        assert StrictExecutionEnv(_obs(4), b, ec)._benchmark_step_return(0, 1) == 0.0
    # 分子非有限 -> 0
    b = _bars(base)
    b.close[1] = float("inf")
    assert StrictExecutionEnv(_obs(4), b, ec)._benchmark_step_return(0, 1) == 0.0


def test_reward_calculator_rejects_unknown_type():
    from stock_ai.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        RewardCalculator(EnvConfig(reward_type="totally_bogus"))


# =========================================================================== #
# H. resume 兼容（真实 run 目录 + 真实训练管线，1 只股票小工作区）
# =========================================================================== #
def _strict_workspace(tmp_path):
    dates = pd.DatetimeIndex([])
    for year in (2021, 2022, 2023, 2024, 2025):
        dates = dates.append(pd.bdate_range(f"{year}-01-04", periods=12))
    n = len(dates)
    hfq_close = 10 + np.arange(n) * 0.03
    raw_close = hfq_close / 7.5
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
    return feature_dir, raw_dir


def _cfg_for(tmp_path, raw_dir, reward_type, lam_t, lam_d, lam_lev):
    from stock_ai.config.loader import ConfigLoader

    cfg = AppConfig()
    cfg.paths.root = tmp_path
    ConfigLoader.resolve_paths(cfg)
    cfg.data.cache_dir = raw_dir
    cfg.sample.window = 3
    cfg.train.n_steps = 8
    cfg.train.batch_size = 8
    cfg.train.n_epochs = 1
    cfg.env.reward_type = reward_type
    cfg.env.lambda_turnover = lam_t
    cfg.env.lambda_drawdown = lam_d
    cfg.env.lambda_leverage = lam_lev
    return cfg


def _run_smoke(tmp_path, raw_dir, feature_dir, run_id, reward_type, lam_t, lam_d, lam_lev):
    from stock_ai.scale.unified import EXECUTION_STRICT, UnifiedRunner

    cfg = _cfg_for(tmp_path, raw_dir, reward_type, lam_t, lam_d, lam_lev)
    directory = UnifiedRunner(cfg).run(8, run_id, n_envs=1, checkpoint_steps=8,
                                       execution_version=EXECUTION_STRICT,
                                       feature_dir=str(feature_dir))
    return directory


def test_resume_legacy_manifest_without_reward_type_is_accepted(tmp_path):
    """旧 manifest（无 reward_type 字段）按 drawdown_penalty 处理，续训不报错。"""
    pytest.importorskip("stable_baselines3")
    from stock_ai.scale.unified import EXECUTION_STRICT, UnifiedRunner

    feature_dir, raw_dir = _strict_workspace(tmp_path)
    directory = _run_smoke(tmp_path, raw_dir, feature_dir, "qa_legacy", "drawdown_penalty",
                           0.001, 0.1, 0.01)
    man_path = directory / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for key in ("reward_type", "reward_version", "reward_params"):
        man.pop(key, None)                       # 模拟旧 run：无这些字段
    man_path.write_text(json.dumps(man), encoding="utf-8")

    cfg2 = _cfg_for(tmp_path, raw_dir, "drawdown_penalty", 0.001, 0.1, 0.01)
    # 不抛异常即为兼容成功
    UnifiedRunner(cfg2).run(8, "qa_legacy", resume=True, n_envs=1, checkpoint_steps=8,
                            execution_version=EXECUTION_STRICT, feature_dir=str(feature_dir))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] in ("done",)


def test_resume_rejects_inconsistent_reward_type(tmp_path):
    """冻结 manifest 的 reward_type 与当前不一致 -> 明确拒绝（专用校验分支）。"""
    pytest.importorskip("stable_baselines3")
    from stock_ai.scale.errors import ScaleError
    from stock_ai.scale.unified import EXECUTION_STRICT, UnifiedRunner

    feature_dir, raw_dir = _strict_workspace(tmp_path)
    directory = _run_smoke(tmp_path, raw_dir, feature_dir, "qa_rs", "excess", 0.002, 0.0, 0.01)
    man_path = directory / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["reward_type"] = "gross"                 # 顶层与 env_config 自相矛盾
    man_path.write_text(json.dumps(man), encoding="utf-8")

    # 当前 cfg 仍是 excess -> env_config（excess）匹配，从而命中专用 reward_type 校验
    cfg2 = _cfg_for(tmp_path, raw_dir, "excess", 0.002, 0.0, 0.01)
    with pytest.raises(ScaleError, match="reward_type"):
        UnifiedRunner(cfg2).run(8, "qa_rs", resume=True, n_envs=1, checkpoint_steps=8,
                                execution_version=EXECUTION_STRICT, feature_dir=str(feature_dir))


def test_resume_rejects_different_reward_changes_env_config(tmp_path):
    """用不同 reward 续训 -> 至少被 env_config 一致性校验拒绝（不得静默继承）。"""
    pytest.importorskip("stable_baselines3")
    from stock_ai.scale.errors import ScaleError
    from stock_ai.scale.unified import EXECUTION_STRICT, UnifiedRunner

    feature_dir, raw_dir = _strict_workspace(tmp_path)
    _run_smoke(tmp_path, raw_dir, feature_dir, "qa_rs2", "excess", 0.002, 0.0, 0.01)

    cfg2 = _cfg_for(tmp_path, raw_dir, "drawdown_penalty", 0.001, 0.1, 0.01)
    with pytest.raises(ScaleError):
        UnifiedRunner(cfg2).run(8, "qa_rs2", resume=True, n_envs=1, checkpoint_steps=8,
                                execution_version=EXECUTION_STRICT, feature_dir=str(feature_dir))
