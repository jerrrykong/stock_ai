"""共享测试夹具。"""

from __future__ import annotations

import numpy as np
import pytest

from stock_ai.config.schema import EnvConfig, SampleConfig
from stock_ai.samples.schema import SampleSplit


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    """把工作目录切到临时目录，避免污染真实 data/。"""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def env_cfg() -> EnvConfig:
    """标准环境配置。"""
    return EnvConfig(
        initial_cash=1_000_000.0,
        commission_rate=0.00025,
        stamp_tax_rate=0.0005,
        slippage_rate=0.001,
        min_commission=5.0,
        lot_size=100,
    )


@pytest.fixture
def sample_cfg() -> SampleConfig:
    """标准样本配置（小窗口）。"""
    return SampleConfig(window=5, features=["close", "volume", "ma5", "momentum"])


@pytest.fixture
def raw_df():
    """构造 120 天的确定性日线 DataFrame。"""
    import pandas as pd

    n = 120
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    dates = pd.bdate_range("2023-01-03", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + rng.normal(0, 0.003, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "volume": np.abs(rng.normal(2e6, 4e5, n)),
            "amount": close * 2e6,
            "adj_factor": 1.0,
        }
    )


@pytest.fixture
def small_split(raw_df, sample_cfg) -> SampleSplit:
    """由 raw_df 构造一个小 SampleSplit。"""
    from stock_ai.samples.features import FeatureEngineer

    eng = FeatureEngineer(sample_cfg.features)
    state = eng.fit(raw_df)
    obs = eng.make_obs_matrix(raw_df, state, sample_cfg.window)
    dates = raw_df["date"].dt.strftime("%Y%m%d").to_numpy()
    return SampleSplit(
        obs=obs,
        dates=dates,
        symbols=np.array(["600519"] * len(raw_df)),
        prices=raw_df["close"].to_numpy(dtype="float64"),
        requested_start="20230103",
        requested_end="20231231",
    )
