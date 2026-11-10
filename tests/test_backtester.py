"""回测器单测。"""

from __future__ import annotations

import numpy as np
import pytest

from stock_ai.config.schema import TrainConfig
from stock_ai.env.trading_env import TradingEnv
from stock_ai.eval.backtester import Backtester
from stock_ai.train.ppo_numpy import NumpyPPOTrainer


@pytest.fixture
def trained_model(small_split, env_cfg, tmp_path):
    """训练一个极小的 NumPy PPO 模型并返回。"""
    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="ppo", timesteps=150, n_steps=32, batch_size=16, n_epochs=1)
    trainer = NumpyPPOTrainer(env, cfg, seed=42)
    trainer.learn(150, seed=42)
    return trainer


def test_backtester_run_returns_metrics(trained_model, small_split, env_cfg):
    """回测返回完整指标字段。"""
    bt = Backtester(trained_model, env_cfg, small_split, mode="discrete")
    result = bt.run(backend="numpy", algo="ppo")
    for key in (
        "annual_return", "total_return", "max_drawdown", "sharpe",
        "win_rate", "turnover", "num_trades", "benchmark_return",
    ):
        assert key in result.metrics
    assert len(result.equity_curve) > 0
    assert len(result.baseline_equity) > 0


def test_backtester_compare_baseline(trained_model, small_split, env_cfg):
    """基线对比返回 excess_return。"""
    bt = Backtester(trained_model, env_cfg, small_split, mode="discrete")
    cmp = bt.compare_with_baseline()
    assert "strategy" in cmp
    assert "baseline" in cmp
    assert "excess_return" in cmp


def test_backtest_result_to_json(trained_model, small_split, env_cfg, tmp_path):
    """报告 JSON 结构完整。"""
    from stock_ai.utils.io import read_json

    bt = Backtester(trained_model, env_cfg, small_split, mode="discrete")
    result = bt.run(backend="numpy", algo="ppo", data_hash="abc123")
    out = tmp_path / "report.json"
    result.to_json(out)
    data = read_json(out)
    assert "meta" in data
    assert "metrics" in data
    assert "equity_curve" in data
    assert data["meta"]["data_hash"] == "abc123"


def test_plot_equity_png(trained_model, small_split, env_cfg, tmp_path):
    """权益曲线 PNG 能被写出。"""
    from stock_ai.eval.plot import EquityPlotter

    bt = Backtester(trained_model, env_cfg, small_split, mode="discrete")
    result = bt.run()
    out = tmp_path / "equity.png"
    EquityPlotter().plot(result, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_training_curve(tmp_path):
    """训练曲线 PNG 能被写出。"""
    from stock_ai.eval.plot import EquityPlotter

    out = tmp_path / "curve.png"
    EquityPlotter().plot_training_curve({"reward": [1.0, 2.0, 1.5]}, out)
    assert out.exists()


def test_plot_handles_missing_cjk(tmp_path):
    """无数据/无中文环境不崩。"""
    from stock_ai.eval.plot import EquityPlotter

    out = tmp_path / "empty.png"
    EquityPlotter().plot_training_curve({}, out)
    assert out.exists()
