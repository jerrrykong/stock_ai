"""``scale --help`` / 6 条既有 CLI ``--help`` 不变 / 端到端冒烟用例（T04）。"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from stock_ai.cli.main import cli

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
# --help
# --------------------------------------------------------------------------- #
def test_main_help_lists_scale():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("data", "sample", "train", "backtest", "predict", "doctor", "scale"):
        assert cmd in result.output


def test_scale_group_and_subcommands_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["scale", "--help"])
    assert r.exit_code == 0
    for sub in ("fetch", "units", "train", "detect", "status", "ui"):
        assert sub in r.output
    for args in (
        ["scale", "fetch", "--help"], ["scale", "units", "--help"],
        ["scale", "train", "--help"], ["scale", "detect", "--help"],
        ["scale", "status", "--help"], ["scale", "ui", "--help"],
    ):
        r = runner.invoke(cli, args)
        assert r.exit_code == 0, f"{args} 失败: {r.output}"


def test_existing_six_clis_help_contract_unchanged():
    """既有 6 条 CLI 的 ``--help`` 契约不变（关键选项仍在）。"""
    runner = CliRunner()
    cases = {
        ("data", "--help"): ["fetch", "refresh", "info"],
        ("sample", "--help"): ["build"],
        ("train", "--help"): ["--timesteps", "--algo", "--backend"],
        ("backtest", "--help"): ["--model"],
        ("predict", "--help"): ["--model"],
        ("doctor", "--help"): ["--config"],
    }
    for args, needles in cases.items():
        r = runner.invoke(cli, list(args))
        assert r.exit_code == 0, f"{args} 失败"
        for n in needles:
            assert n in r.output, f"{args} 缺少 {n}"


# --------------------------------------------------------------------------- #
# 端到端冒烟（synthetic，小样本）
# --------------------------------------------------------------------------- #
def _write_workspace(tmp_path: Path) -> Path:
    """在 tmp 工作区写最小名单 + 配置，返回配置路径。"""
    syms = tmp_path / "scale_symbols.yaml"
    syms.write_text('symbols:\n  - "600519"\n', encoding="utf-8")
    cfg = tmp_path / "scale_test.yaml"
    cfg.write_text(
        "random_seed: 42\n"
        "log_level: WARNING\n"
        "data:\n"
        "  source: synthetic\n"
        "  cache_dir: data/raw\n"
        "  adjust: none\n"
        "sample:\n"
        "  window: 5\n"
        "  features: [close, volume, ma5, momentum]\n"
        "scale:\n"
        "  years: [2018, 2021]\n"
        f"  symbols_file: {syms.as_posix()}\n"
        "  split_ratio: { train: 0.8, detect: 0.2 }\n"
        "  poll_sec: 0.05\n"
        "  train:\n"
        "    timesteps: 16\n"
        "    algo: ppo\n"
        "    backend: numpy\n"
        "  detect:\n"
        "    model_policy: nearest_past\n"
        "    mode: backtest\n"
        "paths:\n"
        "  root: .\n"
        "  raw: data/raw\n"
        "  samples: data/samples\n"
        "  models: models\n"
        "  logs: logs\n"
        "  reports: reports\n",
        encoding="utf-8",
    )
    return cfg


def test_scale_end_to_end_synthetic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_workspace(tmp_path)
    runner = CliRunner()

    r = runner.invoke(cli, ["scale", "fetch", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "data" / "raw" / "600519.parquet").exists()

    r = runner.invoke(cli, ["scale", "units", "--seed", "42", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "data" / "scale" / "units.json").exists()
    assert (tmp_path / "data" / "scale" / "split.json").exists()

    r = runner.invoke(cli, ["scale", "status", "--phase", "train", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    assert "total" in r.output

    r = runner.invoke(cli, ["scale", "train", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    models = list((tmp_path / "models" / "units").glob("*.zip"))
    assert len(models) >= 1, f"未生成单元模型: {list((tmp_path/'models'/'units').glob('*'))}"

    r = runner.invoke(cli, ["scale", "detect", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    reports = list((tmp_path / "reports" / "units").glob("*.json"))
    assert len(reports) >= 1, "未生成检测报告"


def test_scale_status_after_train_shows_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_workspace(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["scale", "fetch", "--config", str(cfg)])
    runner.invoke(cli, ["scale", "units", "--seed", "42", "--config", str(cfg)])
    runner.invoke(cli, ["scale", "train", "--config", str(cfg)])
    r = runner.invoke(cli, ["scale", "status", "--phase", "train", "--config", str(cfg)])
    assert r.exit_code == 0
    assert "done" in r.output
    assert "ETA" in r.output
