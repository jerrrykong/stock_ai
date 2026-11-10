"""单单元训练产物 + norm 持久化 + 检测复用 norm + 编排（中断/续跑/隔离/恢复）用例（T03）。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from stock_ai.config.schema import AppConfig
from stock_ai.data.cache import DataCache
from stock_ai.samples.builder import SampleBuilder
from stock_ai.scale.artifacts import UnitArtifacts
from stock_ai.scale.atomic import atomic_write_json
from stock_ai.scale.normalizer_store import NormalizerStore
from stock_ai.scale.orchestrator import ScaleOrchestrator
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.sampler import UnitSampler
from stock_ai.scale.state import StateStore, UnitState, _now
from stock_ai.scale.unit_detector import UnitDetector
from stock_ai.scale.unit_trainer import UnitTrainer
from stock_ai.scale.units import SplitPlan, Unit


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
def _make_df(start: str, periods: int, seed: int = 3):
    """构造确定性日线 DataFrame（跨数年）。"""
    import pandas as pd

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, periods)))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + rng.normal(0, 0.002, periods)),
            "high": close * (1 + np.abs(rng.normal(0, 0.004, periods))),
            "low": close * (1 - np.abs(rng.normal(0, 0.004, periods))),
            "close": close,
            "volume": np.abs(rng.normal(2e6, 3e5, periods)),
        }
    )


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.paths.root = tmp_path
    cfg.data.cache_dir = tmp_path / "data" / "raw"
    cfg.data.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.sample.window = 5
    cfg.sample.features = ["close", "volume", "ma5", "momentum"]
    cfg.scale.train.timesteps = 32
    cfg.scale.train.backend = "numpy"
    cfg.scale.poll_sec = 0.05
    return cfg


def _write_cache(cfg: AppConfig, symbol: str, df) -> None:
    DataCache(cfg.data.cache_dir).write(symbol, df)


def _write_split(paths: ScalePaths, train_units: list[str], detect_units: list[str]) -> None:
    plan = SplitPlan(
        random_seed=42, train_units=train_units, detect_units=detect_units,
        units_total=len(train_units) + len(detect_units),
    )
    atomic_write_json(plan.to_dict(), paths.split_json)


# --------------------------------------------------------------------------- #
# 单元训练
# --------------------------------------------------------------------------- #
def test_unit_trainer_produces_all_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    df = _make_df("2020-01-02", 262)
    _write_cache(cfg, "600519", df)

    paths = ScalePaths(tmp_path)
    paths.ensure()
    sampler = UnitSampler(SampleBuilder(cfg.sample))
    trainer = UnitTrainer(
        cfg, paths, sampler, NormalizerStore(paths), UnitArtifacts(paths), 42
    )
    res = trainer.run_unit(Unit.make("600519", 2020), df, 0, 1)

    assert res.status == "done", res.error
    assert paths.model_zip("600519_2020").exists()
    assert paths.sidecar("600519_2020").exists()
    assert paths.norm("600519_2020").exists()          # Q2：norm 持久化
    assert paths.curve("600519_2020").exists()
    assert res.data_hash
    assert UnitArtifacts(paths).is_model_done("600519_2020")


def test_unit_trainer_failure_is_isolated(tmp_path):
    """单元级异常返回 failed 且不上抛。"""
    cfg = _cfg(tmp_path)
    paths = ScalePaths(tmp_path)
    paths.ensure()

    class _BoomSampler:
        builder = None

        def sample(self, unit, raw_df, val_ratio):
            raise RuntimeError("boom")

    trainer = UnitTrainer(
        cfg, paths, _BoomSampler(), NormalizerStore(paths), UnitArtifacts(paths), 42
    )
    res = trainer.run_unit(Unit.make("600519", 2020), _make_df("2020-01-02", 262), 0, 1)
    assert res.status == "failed"
    assert "boom" in res.error


# --------------------------------------------------------------------------- #
# 检测（复用训练时 norm，绝不重新 fit）
# --------------------------------------------------------------------------- #
def _train_one(cfg, paths, df, unit):
    sampler = UnitSampler(SampleBuilder(cfg.sample))
    trainer = UnitTrainer(
        cfg, paths, sampler, NormalizerStore(paths), UnitArtifacts(paths), 42
    )
    return trainer.run_unit(unit, df, 0, 1)


def test_detector_reuses_assigned_norm_never_refits(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    df = _make_df("2020-01-02", 520)
    _write_cache(cfg, "600519", df)
    paths = ScalePaths(tmp_path)
    paths.ensure()
    _write_split(paths, ["600519_2020"], ["600519_2021"])

    assert _train_one(cfg, paths, df, Unit.make("600519", 2020)).status == "done"

    # 检测阶段严禁重新 fit：一旦调用 fit 立即失败
    import stock_ai.samples.features as feats

    def _boom(self, df):
        raise AssertionError("检测阶段不得重新 fit（Q2 违规）")

    monkeypatch.setattr(feats.FeatureEngineer, "fit", _boom)

    plan = SplitPlan(train_units=["600519_2020"], detect_units=["600519_2021"])
    detector = UnitDetector(
        cfg, paths, UnitSampler(SampleBuilder(cfg.sample)),
        NormalizerStore(paths), UnitArtifacts(paths), plan, 42,
    )
    res = detector.run_unit(Unit.make("600519", 2021), df, 0, 1)

    assert res.status == "done", res.error
    assert res.assigned_model == "600519_2020"
    assert paths.report("600519_2021").exists()
    assert res.metrics and "sharpe" in res.metrics


def test_detector_skipped_when_normalizer_missing(tmp_path):
    cfg = _cfg(tmp_path)
    df = _make_df("2020-01-02", 520)
    _write_cache(cfg, "600519", df)
    paths = ScalePaths(tmp_path)
    paths.ensure()
    assert _train_one(cfg, paths, df, Unit.make("600519", 2020)).status == "done"
    # 关键：删除 assigned 的 norm → 必须 skipped + normalizer_missing，绝不静默重新 fit
    paths.norm("600519_2020").unlink()

    plan = SplitPlan(train_units=["600519_2020"], detect_units=["600519_2021"])
    detector = UnitDetector(
        cfg, paths, UnitSampler(SampleBuilder(cfg.sample)),
        NormalizerStore(paths), UnitArtifacts(paths), plan, 42,
    )
    res = detector.run_unit(Unit.make("600519", 2021), df, 0, 1)
    assert res.status == "skipped"
    assert res.reason == "normalizer_missing"


def test_detector_skipped_no_model(tmp_path):
    cfg = _cfg(tmp_path)
    df = _make_df("2020-01-02", 520)
    paths = ScalePaths(tmp_path)
    paths.ensure()
    plan = SplitPlan(train_units=[], detect_units=["600519_2021"])
    detector = UnitDetector(
        cfg, paths, UnitSampler(SampleBuilder(cfg.sample)),
        NormalizerStore(paths), UnitArtifacts(paths), plan, 42,
    )
    res = detector.run_unit(Unit.make("600519", 2021), df, 0, 1)
    assert res.status == "skipped"
    assert res.reason.startswith("no_model")


def test_detector_mode_not_supported(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.scale.detect.mode = "predict"   # U4：仅预留
    paths = ScalePaths(tmp_path)
    paths.ensure()
    plan = SplitPlan(train_units=["600519_2020"], detect_units=["600519_2021"])
    detector = UnitDetector(
        cfg, paths, UnitSampler(SampleBuilder(cfg.sample)),
        NormalizerStore(paths), UnitArtifacts(paths), plan, 42,
    )
    res = detector.run_unit(Unit.make("600519", 2021), _make_df("2020-01-02", 520), 0, 1)
    assert res.status == "skipped"
    assert "mode_not_supported" in res.reason


# --------------------------------------------------------------------------- #
# 编排骨架
# --------------------------------------------------------------------------- #
class _StubRunner:
    """快速桩执行器：写非空产物并返回 done，支持注入失败 / 中断。"""

    def __init__(self, paths: ScalePaths, fail_on: str | None = None,
                 interrupt_on_call: int | None = None) -> None:
        self.paths = paths
        self.fail_on = fail_on
        self.interrupt_on_call = interrupt_on_call
        self.calls: list[str] = []

    def run_unit(self, unit: Unit, raw_df, index: int, total: int) -> UnitState:
        self.calls.append(unit.unit_id)
        if self.interrupt_on_call is not None and len(self.calls) == self.interrupt_on_call:
            raise KeyboardInterrupt()
        if self.fail_on and unit.unit_id == self.fail_on:
            raise RuntimeError("injected failure")
        p = self.paths
        p.model_zip(unit.unit_id).write_bytes(b"model")
        p.sidecar(unit.unit_id).write_text("{}", encoding="utf-8")
        p.norm(unit.unit_id).write_text("{}", encoding="utf-8")
        return UnitState(status="done", finished_at=_now(), duration_sec=0.01,
                         model_path=str(p.model_zip(unit.unit_id)))


def _orchestrator(cfg, paths, runner, only_failed=False):
    import logging

    return ScaleOrchestrator(
        cfg, "train", runner, paths, logging.getLogger("test-scale"), 42,
        only_failed=only_failed,
    )


def _prep(tmp_path):
    cfg = _cfg(tmp_path)
    df = _make_df("2018-01-02", 1000)
    _write_cache(cfg, "600519", df)
    paths = ScalePaths(tmp_path)
    paths.ensure()
    _write_split(paths, ["600519_2019", "600519_2020", "600519_2021"], [])
    return cfg, paths, df


def test_orchestrator_interrupt_rollback_and_resume_no_overwrite(tmp_path):
    cfg, paths, _ = _prep(tmp_path)
    runner1 = _StubRunner(paths, interrupt_on_call=2)
    code1 = _orchestrator(cfg, paths, runner1).run()
    assert code1 == 130

    state1 = StateStore(paths).load("train")
    assert state1.units["600519_2019"].status == "done"
    assert state1.units["600519_2020"].status == "pending"   # 中断回滚
    assert state1.units["600519_2021"].status == "pending"
    mtime_before = paths.model_zip("600519_2019").stat().st_mtime

    time.sleep(0.01)
    runner2 = _StubRunner(paths)
    code2 = _orchestrator(cfg, paths, runner2).run()
    assert code2 == 0
    assert runner2.calls == ["600519_2020", "600519_2021"]     # 已完成单元被跳过

    state2 = StateStore(paths).load("train")
    assert state2.units["600519_2019"].finished_at == state1.units["600519_2019"].finished_at
    assert paths.model_zip("600519_2019").stat().st_mtime == mtime_before  # 绝不覆盖
    assert state2.counts["done"] == 3


def test_orchestrator_failed_isolation(tmp_path):
    cfg, paths, _ = _prep(tmp_path)
    runner = _StubRunner(paths, fail_on="600519_2020")
    code = _orchestrator(cfg, paths, runner).run()
    assert code == 0
    state = StateStore(paths).load("train")
    assert state.units["600519_2020"].status == "failed"
    assert "injected failure" in state.units["600519_2020"].error
    assert state.counts["done"] == 2
    assert state.counts["failed"] == 1


def test_orchestrator_only_failed(tmp_path):
    cfg, paths, _ = _prep(tmp_path)
    _orchestrator(cfg, paths, _StubRunner(paths, fail_on="600519_2020")).run()
    # 第二次仅重跑 failed
    runner = _StubRunner(paths)
    _orchestrator(cfg, paths, runner, only_failed=True).run()
    assert runner.calls == ["600519_2020"]
    assert StateStore(paths).load("train").counts["done"] == 3


def test_orchestrator_crash_recovery(tmp_path):
    """模拟上次被杀的 running 悬挂 → 重启后回滚 pending 并重跑。"""
    cfg, paths, _ = _prep(tmp_path)
    store = StateStore(paths)
    state = store.load("train")
    from stock_ai.scale.state import ScaleState

    state = ScaleState(phase="train", total=3, seed=42)
    state.units = {
        "600519_2019": UnitState(status="running"),   # 悬挂，无产物
        "600519_2020": UnitState(status="pending"),
        "600519_2021": UnitState(status="pending"),
    }
    store.save(state)

    runner = _StubRunner(paths)
    code = _orchestrator(cfg, paths, runner).run()
    assert code == 0
    assert set(runner.calls) == {"600519_2019", "600519_2020", "600519_2021"}
    assert StateStore(paths).load("train").counts["done"] == 3


def test_orchestrator_missing_split_raises(tmp_path):
    cfg = _cfg(tmp_path)
    paths = ScalePaths(tmp_path)
    paths.ensure()
    from stock_ai.scale.errors import ScaleError

    try:
        _orchestrator(cfg, paths, _StubRunner(paths)).run()
    except ScaleError:
        return
    raise AssertionError("缺少 split.json 未报错")
