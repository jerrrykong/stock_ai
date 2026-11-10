"""奖励改造：``excess``（相对买入持有基准的超额收益）+ 换手惩罚 + 无前视。

覆盖（对应改造规格 §测试）：
1. **无未来函数**：改写 ``close[t+2..]`` 不改变第 t 步的 ``benchmark_return`` 与 reward；
2. **excess 数值正确**：``r_excess = ret − benchmark_return`` 再减三项惩罚，与手算一致；
3. **换手惩罚单调**：固定 (ret, bench, position)，turnover 增大 → reward 严格递减；
4. **旧类型零回归**：gross / drawdown_penalty / sharpe 传入/不传 ``benchmark_return`` 一致；
5. **默认值未变**：``EnvConfig().reward_type == "drawdown_penalty"``、``lambda_turnover == 0.001``；
6. **CLI 覆盖规则**：``--reward excess`` 时 λ 预设生效；什么都不传时不改；
7. **端到端 smoke**：strict + excess 跑通且 manifest 记录 reward 字段。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from stock_ai.config.schema import EnvConfig
from stock_ai.env.rewards import SUPPORTED_REWARDS, RewardCalculator
from stock_ai.env.strict_env import ExecutionBars, StrictExecutionEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def make_bars(closes, opens=None, volumes=None, symbol="600519"):
    """构造未复权真实 K 线（长度 = ``len(closes)``）。"""
    n = len(closes)
    dates = pd.bdate_range("2024-01-02", periods=n).strftime("%Y%m%d").to_numpy()
    close = np.asarray(closes, dtype="float64")
    if opens is None:
        opens = close * 1.001
    return ExecutionBars(
        dates=dates, open=np.asarray(opens, dtype="float64"),
        high=np.maximum(np.asarray(opens, dtype="float64"), close) * 1.01,
        low=np.minimum(np.asarray(opens, dtype="float64"), close) * 0.99,
        close=close,
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
# 1. 白名单 & 默认值
# --------------------------------------------------------------------------- #
def test_supported_rewards_includes_excess():
    assert "excess" in SUPPORTED_REWARDS
    assert SUPPORTED_REWARDS == frozenset({"gross", "drawdown_penalty", "sharpe", "excess"})


def test_env_config_defaults_unchanged():
    """默认值未变：reward_type 仍 drawdown_penalty，lambda_turnover 仍 0.001。"""
    cfg = EnvConfig()
    assert cfg.reward_type == "drawdown_penalty"
    assert cfg.lambda_turnover == 0.001
    assert cfg.lambda_drawdown == 0.1
    assert cfg.lambda_leverage == 0.01


def test_reward_calculator_accepts_excess():
    calc = RewardCalculator(EnvConfig(reward_type="excess"))
    assert calc is not None


# --------------------------------------------------------------------------- #
# 2. excess 数值正确
# --------------------------------------------------------------------------- #
def test_excess_matches_hand_calculation_on_rise():
    cfg = EnvConfig(reward_type="excess", lambda_drawdown=0.1,
                    lambda_turnover=0.002, lambda_leverage=0.01)
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    prev, value = 1_000_000.0, 1_020_000.0
    bench, turnover, pos = 0.01, 0.5, 1.0
    ret = (value - prev) / prev
    dd = 0.0  # 上涨无回撤（peak 即当前值）
    expected = (ret - bench) - cfg.lambda_drawdown * dd \
        - cfg.lambda_turnover * turnover - cfg.lambda_leverage * abs(pos)
    got = calc.compute(prev, value, turnover=turnover, position_ratio=pos,
                       benchmark_return=bench)
    assert got == pytest.approx(expected)


def test_excess_matches_hand_calculation_with_drawdown():
    """先涨到峰值，再回落：drawdown 项参与惩罚。"""
    cfg = EnvConfig(reward_type="excess", lambda_drawdown=0.1,
                    lambda_turnover=0.002, lambda_leverage=0.0)
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    calc.compute(1_000_000.0, 1_050_000.0, turnover=0.0, position_ratio=0.0,
                 benchmark_return=0.0)   # 峰值 1.05M
    prev, value = 1_050_000.0, 1_000_000.0
    bench, turnover, pos = -0.02, 0.3, 0.5
    ret = (value - prev) / prev
    dd = (1_050_000.0 - 1_000_000.0) / 1_050_000.0
    expected = (ret - bench) - cfg.lambda_drawdown * dd \
        - cfg.lambda_turnover * turnover - cfg.lambda_leverage * abs(pos)
    got = calc.compute(prev, value, turnover=turnover, position_ratio=pos,
                       benchmark_return=bench)
    assert got == pytest.approx(expected)


def test_excess_default_benchmark_is_zero_cost_free():
    """不传 benchmark_return 时等价于基准收益 0（与 ret 绝对收益一致）。"""
    cfg = EnvConfig(reward_type="excess", lambda_drawdown=0.0,
                    lambda_turnover=0.0, lambda_leverage=0.0)
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    r = calc.compute(1_000_000.0, 1_010_000.0)
    assert r == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# 3. 换手惩罚单调 & 数值防御
# --------------------------------------------------------------------------- #
def test_excess_turnover_penalty_strictly_monotonic():
    cfg = EnvConfig(reward_type="excess", lambda_drawdown=0.0,
                    lambda_turnover=0.01, lambda_leverage=0.0)
    calc = RewardCalculator(cfg)
    calc.reset(1_000_000.0)
    prev, value, bench, pos = 1_000_000.0, 1_000_000.0, 0.0, 0.0
    rewards = [calc.compute(prev, value, turnover=t, position_ratio=pos,
                            benchmark_return=bench) for t in (0.0, 0.5, 1.0, 2.0)]
    assert all(a > b for a, b in zip(rewards, rewards[1:]))
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[-1] == pytest.approx(-0.01 * 2.0)


def test_excess_no_nan_or_inf_on_degenerate_inputs():
    cfg = EnvConfig(reward_type="excess")
    calc = RewardCalculator(cfg)
    calc.reset(0.0)
    for prev, value, bench, turnover, pos in [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1_000_000.0, 1_000_000.0, float("nan"), 0.0, 0.0),
        (1_000_000.0, float("inf"), 0.01, 0.0, 0.0),
        (1_000_000.0, 1_010_000.0, 0.02, float("inf"), float("nan")),
    ]:
        r = calc.compute(prev, value, turnover=turnover, position_ratio=pos,
                         benchmark_return=bench)
        assert np.isfinite(r), (prev, value, bench, turnover, pos, r)


# --------------------------------------------------------------------------- #
# 4. 旧类型零回归
# --------------------------------------------------------------------------- #
def test_legacy_reward_types_ignore_benchmark_param():
    """gross / drawdown_penalty / sharpe 传入 benchmark_return 后数值不变。"""
    for reward_type in ("gross", "drawdown_penalty", "sharpe"):
        cfg = EnvConfig(reward_type=reward_type, lambda_drawdown=0.1,
                        lambda_turnover=0.002, lambda_leverage=0.01)
        base = RewardCalculator(cfg)
        with_bench = RewardCalculator(cfg)
        base.reset(1_000_000.0)
        with_bench.reset(1_000_000.0)
        args = dict(prev_value=1_000_000.0, value=1_020_000.0,
                    turnover=0.5, position_ratio=0.8)
        assert base.compute(**args) == with_bench.compute(**args, benchmark_return=0.05)


# --------------------------------------------------------------------------- #
# 5. 无未来函数（环境级）
# --------------------------------------------------------------------------- #
def test_benchmark_step_return_is_close_to_close():
    closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    env = StrictExecutionEnv(make_obs(5), make_bars(closes), EnvConfig())
    assert env._benchmark_step_return(0, 1) == pytest.approx(11.0 / 10.0 - 1.0)
    assert env._benchmark_step_return(1, 2) == pytest.approx(12.0 / 11.0 - 1.0)
    # 分母非正 / 非有限 → 0
    bad = make_bars(closes)
    bad.close[0] = 0.0
    env_bad = StrictExecutionEnv(make_obs(5), bad, EnvConfig())
    assert env_bad._benchmark_step_return(0, 1) == 0.0


def test_future_closes_do_not_change_step_t_benchmark_and_reward(env_cfg):
    """改写 T+1 之后的收盘价，不改变第 T 步的 benchmark_return 与 reward。"""
    base_closes = np.linspace(10.0, 12.0, 12)
    opens = base_closes * 1.001
    env_a = StrictExecutionEnv(make_obs(12), make_bars(base_closes, opens=opens), env_cfg)
    env_a.reset()
    _, reward_a, _, _, info_a = env_a.step(0)

    tampered = base_closes.copy()
    tampered[2:] *= 50.0                       # 从 t+2 起全部篡改
    env_b = StrictExecutionEnv(make_obs(12), make_bars(tampered, opens=opens), env_cfg)
    env_b.reset()
    _, reward_b, _, _, info_b = env_b.step(0)

    assert info_a["benchmark_return"] == pytest.approx(info_b["benchmark_return"])
    assert reward_a == pytest.approx(reward_b)
    # 第 0 步基准只由 close[0]、close[1] 决定
    assert info_a["benchmark_return"] == pytest.approx(base_closes[1] / base_closes[0] - 1.0)
    assert "benchmark_return" in info_a and "step_excess" in info_a
    assert np.isfinite(reward_a)


def test_excess_reward_in_env_uses_benchmark(env_cfg):
    """excess 环境：λ 全 0 时 reward == step_excess == 策略步收益 − 基准收益。"""
    env_cfg.reward_type = "excess"
    env_cfg.lambda_drawdown = 0.0
    env_cfg.lambda_turnover = 0.0
    env_cfg.lambda_leverage = 0.0
    closes = np.array([10.0, 11.0, 12.0, 13.0])
    env = StrictExecutionEnv(make_obs(4), make_bars(closes), env_cfg)
    env.reset()
    _, reward, _, _, info = env.step(1)   # 买入；第 0 步估值基于 close[0] -> close[1]
    ret_value = (info["portfolio_value"] - env_cfg.initial_cash) / env_cfg.initial_cash
    assert info["step_excess"] == pytest.approx(ret_value - info["benchmark_return"])
    assert reward == pytest.approx(info["step_excess"])
    assert info["benchmark_return"] == pytest.approx(closes[1] / closes[0] - 1.0)


# --------------------------------------------------------------------------- #
# 6. CLI 覆盖规则
# --------------------------------------------------------------------------- #
class _FakeRunner:
    """捕获构造时的 cfg，不真正训练。"""

    captured = {}

    def __init__(self, cfg):
        _FakeRunner.captured["cfg"] = cfg

    def run(self, *args, **kwargs):
        return "fake-run-dir"


@pytest.fixture
def patched_runner(monkeypatch):
    import stock_ai.scale.unified as unified

    _FakeRunner.captured = {}
    monkeypatch.setattr(unified, "UnifiedRunner", _FakeRunner)
    return _FakeRunner


def _invoke_train(args):
    from stock_ai.cli.scale_cmd import unified_train

    return CliRunner().invoke(
        unified_train, ["--config", str(DEFAULT_CONFIG)] + args
    )


def test_cli_excess_applies_lambda_presets(patched_runner):
    result = _invoke_train(["--reward", "excess"])
    assert result.exit_code == 0, result.output
    cfg = patched_runner.captured["cfg"]
    assert cfg.env.reward_type == "excess"
    assert cfg.env.lambda_turnover == 0.002
    assert cfg.env.lambda_drawdown == 0.0
    # lambda_leverage 也纳入 excess 预设：本环境 long-only 全仓，按步收的
    # 持仓惩罚会退化成「持仓时长税」（年成本 5.6 倍于基准波幅），
    # 使「永不买入」成为全局最优。详见 scale_cmd.py 预设处注释。
    assert cfg.env.lambda_leverage == 0.0


def test_cli_explicit_lambda_overrides_preset(patched_runner):
    result = _invoke_train(["--reward", "excess", "--lambda-turnover", "0.5",
                            "--lambda-drawdown", "0.3", "--lambda-leverage", "0.2"])
    assert result.exit_code == 0, result.output
    cfg = patched_runner.captured["cfg"]
    assert cfg.env.lambda_turnover == 0.5
    assert cfg.env.lambda_drawdown == 0.3
    assert cfg.env.lambda_leverage == 0.2


def test_cli_lambda_only_without_reward(patched_runner):
    result = _invoke_train(["--lambda-turnover", "0.9"])
    assert result.exit_code == 0, result.output
    cfg = patched_runner.captured["cfg"]
    assert cfg.env.reward_type == "drawdown_penalty"   # 未给 --reward 不改
    assert cfg.env.lambda_turnover == 0.9
    assert cfg.env.lambda_drawdown == 0.1              # 未给则保持默认


def test_cli_no_flags_leaves_config_untouched(patched_runner):
    result = _invoke_train([])
    assert result.exit_code == 0, result.output
    cfg = patched_runner.captured["cfg"]
    assert cfg.env.reward_type == "drawdown_penalty"
    assert cfg.env.lambda_turnover == 0.001
    assert cfg.env.lambda_drawdown == 0.1
    assert cfg.env.lambda_leverage == 0.01


def _write_excess_yaml(tmp_path, lam_turnover, lam_drawdown, lam_leverage):
    """写一个 ``reward_type: excess`` 且自定义 λ 的 YAML，返回其路径。"""
    import yaml

    from stock_ai.config.schema import AppConfig

    cfg = AppConfig.from_yaml(DEFAULT_CONFIG)
    cfg.env.reward_type = "excess"
    cfg.env.lambda_turnover = lam_turnover
    cfg.env.lambda_drawdown = lam_drawdown
    cfg.env.lambda_leverage = lam_leverage
    path = tmp_path / "excess_custom.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    return path


def test_cli_yaml_excess_without_flag_keeps_yaml_lambdas(tmp_path, patched_runner):
    """YAML 里 reward_type=excess + 自定义 λ，**不带任何新 flag** → 尊重 YAML 的 λ（P2 修复）。"""
    from stock_ai.cli.scale_cmd import unified_train

    cfg_file = _write_excess_yaml(tmp_path, 0.005, 0.07, 0.02)
    result = CliRunner().invoke(unified_train, ["--config", str(cfg_file)])
    assert result.exit_code == 0, result.output
    env = patched_runner.captured["cfg"].env
    assert env.reward_type == "excess"
    # 关键：λ 不被 excess 预设静默覆盖成 0.002 / 0.0
    assert env.lambda_turnover == 0.005
    assert env.lambda_drawdown == 0.07
    assert env.lambda_leverage == 0.02


def test_cli_yaml_excess_with_explicit_reward_flag_applies_preset(tmp_path, patched_runner):
    """同一 YAML，但显式带 ``--reward excess`` → 预设生效（覆盖 YAML 的 λ）。"""
    from stock_ai.cli.scale_cmd import unified_train

    cfg_file = _write_excess_yaml(tmp_path, 0.005, 0.07, 0.02)
    result = CliRunner().invoke(unified_train, ["--config", str(cfg_file), "--reward", "excess"])
    assert result.exit_code == 0, result.output
    env = patched_runner.captured["cfg"].env
    assert env.reward_type == "excess"
    assert env.lambda_turnover == 0.002      # 预设生效
    assert env.lambda_drawdown == 0.0        # 预设生效
    assert env.lambda_leverage == 0.0        # 预设生效（持仓惩罚同为预设项）


def test_cli_invalid_reward_reports_clear_error():
    result = _invoke_train(["--reward", "bogus"])
    assert result.exit_code != 0
    assert "不支持的奖励类型" in str(result.exception)


def test_cli_help_lists_reward_options():
    from stock_ai.cli.scale_cmd import unified_train

    result = CliRunner().invoke(unified_train, ["--help"])
    assert result.exit_code == 0
    for flag in ("--reward", "--lambda-turnover", "--lambda-drawdown", "--lambda-leverage"):
        assert flag in result.output


# --------------------------------------------------------------------------- #
# 7. 端到端 smoke（strict + excess）
# --------------------------------------------------------------------------- #
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


def test_unified_excess_run_records_manifest(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stock_ai.config.loader import ConfigLoader
    from stock_ai.config.schema import AppConfig
    from stock_ai.scale.unified import (
        EXECUTION_STRICT, REWARD_VERSION_EXCESS, UnifiedRunner,
    )

    cfg = AppConfig()
    cfg.paths.root = tmp_path
    ConfigLoader.resolve_paths(cfg)
    cfg.data.cache_dir = tmp_path / "data" / "raw"
    feature_dir, _raw_dir = _strict_workspace(tmp_path)
    cfg.sample.window = 3
    cfg.train.n_steps = 8
    cfg.train.batch_size = 8
    cfg.train.n_epochs = 1
    cfg.env.reward_type = "excess"
    cfg.env.lambda_turnover = 0.002
    cfg.env.lambda_drawdown = 0.0

    directory = UnifiedRunner(cfg).run(8, "excess_smoke", n_envs=1, checkpoint_steps=8,
                                       execution_version=EXECUTION_STRICT,
                                       feature_dir=str(feature_dir))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reward_type"] == "excess"
    assert manifest["reward_version"] == REWARD_VERSION_EXCESS == "excess-causal-close-v1"
    assert manifest["reward_params"]["reward_type"] == "excess"
    assert manifest["reward_params"]["lambda_turnover"] == 0.002
    assert manifest["reward_params"]["lambda_drawdown"] == 0.0
    assert manifest["reward_params"]["lambda_leverage"] == cfg.env.lambda_leverage

    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["reward_type"] == "excess"
    assert metrics["reward_version"] == "excess-causal-close-v1"
    assert metrics["reward_params"]["reward_type"] == "excess"
    for split in ("val", "test"):
        record = metrics["evaluation"][split]["episodes"][0]
        assert "benchmark_return_cc" in record


def test_unified_legacy_reward_version_defaults(tmp_path):
    """非 excess 类型写 reward_version=legacy，且默认 reward_type 可区分。"""
    from stock_ai.config.schema import EnvConfig
    from stock_ai.scale.unified import _reward_payload

    rt, ver, params = _reward_payload(EnvConfig())
    assert rt == "drawdown_penalty"
    assert ver == "legacy"
    assert params["reward_type"] == "drawdown_penalty"
