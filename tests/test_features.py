"""特征工程无泄漏验证。"""

from __future__ import annotations

import numpy as np
import pytest

from stock_ai.samples.features import FeatureEngineer


def test_fit_only_uses_given_data(raw_df):
    """fit 只用传入数据：改动测试段不改变 mean/std（防泄漏核心）。"""
    features = ["close", "volume", "ma5", "momentum"]
    eng = FeatureEngineer(features)

    train = raw_df.iloc[:80].reset_index(drop=True)
    state_a = eng.fit(train)

    # 人为篡改训练段之后的数据（模拟测试集）
    tampered = raw_df.copy()
    tampered.loc[80:, "close"] = tampered.loc[80:, "close"] * 10.0
    train_again = tampered.iloc[:80].reset_index(drop=True)
    state_b = eng.fit(train_again)

    np.testing.assert_allclose(state_a.mean, state_b.mean)
    np.testing.assert_allclose(state_a.std, state_b.std)


def test_transform_uses_frozen_state(raw_df):
    """transform 不重新 fit：传入不同数据用同一 state，输出仅受 state 影响。"""
    features = ["close", "volume", "ma5"]
    eng = FeatureEngineer(features)
    train = raw_df.iloc[:60].reset_index(drop=True)
    state = eng.fit(train)

    out1 = eng.transform(raw_df.iloc[60:90].reset_index(drop=True), state)
    out2 = eng.transform(raw_df.iloc[60:90].reset_index(drop=True), state)
    np.testing.assert_allclose(out1, out2)


def test_obs_matrix_shape_and_no_nan(raw_df):
    """滑窗张量形状正确且无 NaN/Inf。"""
    eng = FeatureEngineer(["close", "volume", "ma5", "ma10", "momentum", "volatility", "volume_ratio"])
    state = eng.fit(raw_df)
    obs = eng.make_obs_matrix(raw_df, state, window=10)
    assert obs.shape == (len(raw_df), 10, 7)
    assert np.all(np.isfinite(obs))


def test_feature_order_matches_request(raw_df):
    """特征列顺序严格按请求顺序（K8）。"""
    names = ["volume_ratio", "close", "ma5", "momentum"]
    eng = FeatureEngineer(names)
    state = eng.fit(raw_df)
    assert list(state.columns) == names
    obs = eng.make_obs_matrix(raw_df, state, window=5)
    assert obs.shape[2] == len(names)


def test_unsupported_feature_raises():
    """不支持的特征名抛 ArtifactError。"""
    from stock_ai.utils.errors import ArtifactError

    with pytest.raises(ArtifactError):
        FeatureEngineer(["close", "not_a_feature"])


def test_constant_column_no_div_zero():
    """常数列不会被除零（std 兜底为 1）。"""
    import pandas as pd

    n = 40
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-03", periods=n),
            "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
            "volume": 0.0, "amount": 0.0, "adj_factor": 1.0,
        }
    )
    eng = FeatureEngineer(["close", "volume"])
    state = eng.fit(df)
    obs = eng.transform(df, state)
    assert np.all(np.isfinite(obs))
