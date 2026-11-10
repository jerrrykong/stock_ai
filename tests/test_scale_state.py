"""状态原子写 + 崩溃恢复 + 控制/归一化/模型策略用例（T02）。"""

from __future__ import annotations

import numpy as np
import pytest

from stock_ai.scale.control import ControlStore
from stock_ai.scale.model_policy import ModelSelector
from stock_ai.scale.normalizer_store import NormalizerStore
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.state import ScaleState, StateStore, UnitState
from stock_ai.samples.normalizer import NormalizerState


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #
def test_state_roundtrip_and_counts(tmp_path):
    paths = ScalePaths(tmp_path)
    paths.ensure()
    store = StateStore(paths)
    state = ScaleState(phase="train", total=3, seed=42)
    state.units = {
        "a_2020": UnitState(status="done", finished_at="20200101_000000"),
        "a_2021": UnitState(status="failed", error="boom"),
        "a_2022": UnitState(status="pending"),
    }
    store.save(state)
    loaded = store.load("train")
    assert loaded.total == 3
    assert loaded.counts["done"] == 1
    assert loaded.counts["failed"] == 1
    assert loaded.counts["pending"] == 1
    assert loaded.units["a_2021"].error == "boom"


def test_state_load_missing_returns_empty(tmp_path):
    paths = ScalePaths(tmp_path)
    paths.ensure()
    state = StateStore(paths).load("detect")
    assert state.phase == "detect"
    assert state.total == 0
    assert state.units == {}


def test_recover_running_and_done(tmp_path):
    paths = ScalePaths(tmp_path)
    paths.ensure()
    store = StateStore(paths)
    state = ScaleState(phase="train", total=3)
    state.units = {
        "a_2020": UnitState(status="running"),   # 产物无效 → pending
        "a_2021": UnitState(status="done"),      # 产物缺失 → pending
        "a_2022": UnitState(status="running"),   # 产物有效 → done（悬挂兜底）
    }
    info = store.recover(state, verify=lambda uid: uid == "a_2022")
    assert state.units["a_2020"].status == "pending"
    assert state.units["a_2021"].status == "pending"
    assert state.units["a_2022"].status == "done"
    assert "a_2020" in info["reset_to_pending"]
    assert "a_2021" in info["reset_done"]


def test_state_refresh_progress_eta(tmp_path):
    state = ScaleState(phase="train", total=4, elapsed_sec=10.0)
    state.units = {
        "a": UnitState(status="done"),
        "b": UnitState(status="done"),
        "c": UnitState(status="failed"),
        "d": UnitState(status="pending"),
    }
    counts = state.refresh_progress()
    assert counts["done"] == 2 and counts["failed"] == 1 and counts["pending"] == 1
    # 平均 = 10 / (2+1) 秒；剩余 = 4 - 3 = 1
    assert state.avg_unit_sec == pytest.approx(10.0 / 3.0)
    assert state.eta_sec == pytest.approx(10.0 / 3.0 * 1)


# --------------------------------------------------------------------------- #
# 控制信号
# --------------------------------------------------------------------------- #
def test_control_poll_missing_and_wrong_phase(tmp_path):
    paths = ScalePaths(tmp_path)
    paths.ensure()
    cs = ControlStore(paths)
    assert cs.poll("train") == ""
    cs.write("train", "pause")
    assert cs.poll("train") == "pause"
    assert cs.poll("detect") == ""      # phase 不匹配忽略
    cs.clear()
    assert cs.poll("train") == ""


def test_control_write_rejects_bad_cmd(tmp_path):
    cs = ControlStore(ScalePaths(tmp_path))
    with pytest.raises(ValueError):
        cs.write("train", "explode")


# --------------------------------------------------------------------------- #
# 归一化存储
# --------------------------------------------------------------------------- #
def test_normalizer_store_roundtrip(tmp_path):
    paths = ScalePaths(tmp_path)
    paths.ensure()
    ns = NormalizerStore(paths)
    assert ns.exists("a_2020") is False
    state = NormalizerState(
        mean=np.array([1.0, 2.0]), std=np.array([0.5, 0.25]),
        columns=["close", "volume"], data_hash="abc",
    )
    ns.save("a_2020", state)
    assert ns.exists("a_2020") is True
    back = ns.load("a_2020")
    assert np.allclose(back.mean, state.mean)
    assert np.allclose(back.std, state.std)
    assert back.columns == ["close", "volume"]
    assert back.data_hash == "abc"


# --------------------------------------------------------------------------- #
# 模型策略
# --------------------------------------------------------------------------- #
def test_model_selector_policies():
    train_units = ["600519_2015", "600519_2018", "600519_2020", "000001_2019"]
    available = set(train_units)

    # nearest_past：同代码 ≤ 检测年 最大
    assert ModelSelector.select("nearest_past", "600519", 2019, train_units, available) == "600519_2018"
    # 检测 2014 → 无 past → 回退 max_year
    assert ModelSelector.select("nearest_past", "600519", 2014, train_units, available) == "600519_2020"
    # max_year
    assert ModelSelector.select("max_year", "600519", 2019, train_units, available) == "600519_2020"
    # nearest：|Δ| 最小
    assert ModelSelector.select("nearest", "600519", 2019, train_units, available) == "600519_2018"
    assert ModelSelector.select("nearest", "600519", 2021, train_units, available) == "600519_2020"
    # none → 全跳过
    assert ModelSelector.select("none", "600519", 2019, train_units, available) == ""
    # 无同代码 → ""
    assert ModelSelector.select("nearest_past", "300750", 2019, train_units, available) == ""
    # 模型未就绪（不在 available）→ ""
    assert ModelSelector.select("nearest_past", "000001", 2019, train_units, set()) == ""


def test_model_selector_rejects_unknown_policy():
    """非法策略必须显式报错，绝不静默回退到 nearest_past（P2-1 回归）。"""
    from stock_ai.scale.errors import ScaleError

    with pytest.raises(ScaleError):
        ModelSelector.select(
            "bogus", "600519", 2019, ["600519_2018"], {"600519_2018"}
        )
