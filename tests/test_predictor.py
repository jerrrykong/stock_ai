"""预测器单测：复用冻结归一化状态（K9 断言）。"""

from __future__ import annotations

import numpy as np
import pytest

from stock_ai.config.schema import TrainConfig
from stock_ai.env.trading_env import TradingEnv
from stock_ai.infer.predictor import Predictor
from stock_ai.samples.features import FeatureEngineer
from stock_ai.train.ppo_numpy import NumpyPPOTrainer
from stock_ai.utils.errors import ArtifactError


@pytest.fixture
def frozen_state(raw_df, sample_cfg):
    """冻结的归一化状态。"""
    eng = FeatureEngineer(sample_cfg.features)
    return eng.fit(raw_df), sample_cfg


@pytest.fixture
def trained(small_split, env_cfg):
    """训练好的小模型。"""
    env = TradingEnv(small_split, env_cfg, seed=42)
    cfg = TrainConfig(algo="ppo", timesteps=150, n_steps=32, batch_size=16, n_epochs=1)
    tr = NumpyPPOTrainer(env, cfg, seed=42)
    tr.learn(150, seed=42)
    return tr


def test_predict_from_frames_columns(trained, frozen_state, env_cfg, raw_df):
    """预测行字段与 PRD §6 契约一致。"""
    state, sample_cfg = frozen_state
    pred = Predictor(trained, env_cfg, state, sample_cfg.features,
                     sample_cfg.window, mode="discrete")
    rows = pred.predict_from_frames({"600519": raw_df})
    assert len(rows) == len(raw_df)
    for r in rows[:5]:
        assert r.action in {"买", "卖", "持有"}
        assert 0.0 <= r.target_position <= 1.0
        assert -1.0 <= r.signal <= 1.0
        d = r.to_dict()
        assert set(d.keys()) == {
            "date", "symbol", "action", "target_position", "signal", "price",
        }


def test_predictor_reuses_frozen_state_no_refit(trained, frozen_state, env_cfg, raw_df, monkeypatch):
    """predict 绝不重新 fit（K9）：monkeypatch fit 使其抛错，调用仍应成功。"""
    state, sample_cfg = frozen_state
    pred = Predictor(trained, env_cfg, state, sample_cfg.features,
                     sample_cfg.window, mode="discrete")

    def _boom(self, df):
        raise AssertionError("predict 不得调用 fit！")

    monkeypatch.setattr(FeatureEngineer, "fit", _boom, raising=True)
    rows = pred.predict_from_frames({"600519": raw_df})
    assert len(rows) > 0


def test_predictor_requires_frozen_state(trained, env_cfg):
    """无冻结状态时抛 ArtifactError。"""
    from stock_ai.samples.normalizer import NormalizerState
    empty = NormalizerState(mean=np.array([]), std=np.array([]), columns=[])
    with pytest.raises(ArtifactError):
        Predictor(trained, env_cfg, empty, ["close"], 5)


def test_predictor_stateful_position(trained, frozen_state, env_cfg, raw_df):
    """逐日 stateful：买入后仓位 >0，卖出后归 0。"""
    state, sample_cfg = frozen_state
    pred = Predictor(trained, env_cfg, state, sample_cfg.features,
                     sample_cfg.window, mode="discrete")
    rows = pred.predict_from_frames({"600519": raw_df})
    positions = [r.target_position for r in rows]
    assert all(0.0 <= p <= 1.0 for p in positions)


def test_to_csv_and_json(trained, frozen_state, env_cfg, raw_df, tmp_path):
    """CSV / JSON 落盘。"""
    import csv

    from stock_ai.utils.io import read_json

    state, sample_cfg = frozen_state
    pred = Predictor(trained, env_cfg, state, sample_cfg.features,
                     sample_cfg.window, mode="discrete")
    rows = pred.predict_from_frames({"600519": raw_df})
    csv_path = pred.to_csv(rows, tmp_path / "p.csv")
    json_path = pred.to_json(rows, tmp_path / "p.json")
    assert csv_path.exists() and json_path.exists()

    with csv_path.open("r", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == ["date", "symbol", "action", "target_position", "signal", "price"]

    data = read_json(json_path)
    assert data["count"] == len(rows)
