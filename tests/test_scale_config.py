"""``scale`` 配置段解析 / 覆盖 / 路径解析 / 名单校验用例（T01）。"""

from __future__ import annotations

from pathlib import Path

from stock_ai.config.loader import ConfigLoader
from stock_ai.config.schema import AppConfig, ScaleConfig, ScaleDetectConfig, ScaleTrainConfig
from stock_ai.scale.errors import ScaleError
from stock_ai.scale.symbols import load_symbols, resolve_year_range

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "configs" / "default.yaml"
SYMBOLS_YAML = REPO / "configs" / "symbols_hs300_full.yaml"


def test_scale_dataclass_defaults():
    """``ScaleConfig`` 代码默认值符合 Q4/§7.6。"""
    sc = ScaleConfig()
    assert sc.years is None
    assert sc.symbols_file == Path("configs/symbols_hs300_full.yaml")
    assert sc.split_ratio == {"train": 0.8, "detect": 0.2}
    assert sc.workers == 1
    assert sc.poll_sec == 2.0
    assert isinstance(sc.train, ScaleTrainConfig)
    assert isinstance(sc.detect, ScaleDetectConfig)
    assert sc.train.timesteps == 20_000
    assert sc.train.algo == "ppo"
    assert sc.train.backend == "auto"
    assert sc.detect.model_policy == "nearest_past"
    assert sc.detect.mode == "backtest"


def test_default_yaml_has_scale_section():
    """``configs/default.yaml`` 含 ``scale`` 且取值符合预期。"""
    cfg = AppConfig.from_yaml(DEFAULT_YAML)
    assert cfg.scale.train.timesteps == 20000
    assert cfg.scale.train.algo == "ppo"
    assert cfg.scale.train.backend == "auto"
    assert cfg.scale.detect.model_policy == "nearest_past"
    assert cfg.scale.detect.mode == "backtest"
    assert cfg.scale.workers == 1
    assert cfg.scale.split_ratio == {"train": 0.8, "detect": 0.2}
    assert cfg.scale.years is None


def test_resolve_paths_symbols_file_is_absolute():
    """``ConfigLoader.resolve_paths`` 后 ``scale.symbols_file`` 为绝对路径。"""
    cfg = ConfigLoader.load(DEFAULT_YAML)
    assert Path(cfg.scale.symbols_file).is_absolute()


def test_symbols_file_has_enough_real_codes():
    """名单文件 ≥250 只，且均为 6 位数字真实代码。"""
    syms = load_symbols(SYMBOLS_YAML)
    assert len(syms) >= 250
    assert all(len(s) == 6 and s.isdigit() for s in syms)
    assert len(set(syms)) == len(syms)
    # 抽查若干典型真实代码
    for code in ("600519", "000001"):
        assert code in syms


def test_load_symbols_rejects_bad_code(tmp_path):
    """非法代码（非 6 位数字）应报错，严禁编造。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text('symbols:\n  - "60051"\n', encoding="utf-8")
    try:
        load_symbols(bad)
    except ScaleError:
        return
    raise AssertionError("非法代码未被拒绝")


def test_load_symbols_missing_file(tmp_path):
    """名单文件缺失应抛 :class:`ScaleError`。"""
    try:
        load_symbols(tmp_path / "nope.yaml")
    except ScaleError:
        return
    raise AssertionError("缺失文件未报错")


def test_resolve_year_range_dynamic_and_override():
    """U1 更正：动态取系统时钟；当前 2026 → (2006, 2025)；显式配置可覆盖。"""
    from datetime import date

    assert resolve_year_range(None, date(2026, 10, 10)) == (2006, 2025)
    assert resolve_year_range(None, date(2025, 1, 1)) == (2005, 2024)
    assert resolve_year_range([2010, 2015]) == (2010, 2015)
    assert resolve_year_range([2015, 2010]) == (2010, 2015)


def test_scale_cli_overrides_do_not_touch_legacy_fields():
    """新增 scale 段不影响既有配置字段默认值。"""
    cfg = AppConfig()
    assert cfg.data.adjust == "none"
    assert cfg.sample.window == 20
    assert cfg.train.algo == "ppo"
    # 显式覆盖 scale 子字段
    cfg.scale.train.timesteps = 1234
    assert cfg.scale.train.timesteps == 1234
