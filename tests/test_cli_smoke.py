"""CLI 冒烟测试：--help 与端到端（synthetic 数据）。"""

from __future__ import annotations

import subprocess
import sys

from click.testing import CliRunner

from stock_ai.cli.main import cli

PY = sys.executable


# ------------------------- --help -------------------------
def test_main_help_lists_all_commands():
    """主命令 --help 列出 6 类子命令 + doctor。"""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("data", "sample", "train", "backtest", "predict", "doctor"):
        assert cmd in result.output


def test_subcommand_help():
    """每条子命令 --help 可用。"""
    runner = CliRunner()
    for args in (
        ["data", "--help"], ["data", "fetch", "--help"], ["data", "refresh", "--help"],
        ["data", "info", "--help"], ["sample", "--help"], ["sample", "build", "--help"],
        ["train", "--help"], ["backtest", "--help"], ["predict", "--help"],
        ["doctor", "--help"],
    ):
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, f"{args} 失败: {result.output}"


def test_module_entrypoint_help():
    """python -m stock_ai --help 可用（子进程）。"""
    proc = subprocess.run(
        [PY, "-m", "stock_ai", "--help"],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert proc.returncode == 0
    assert "data" in proc.stdout


def test_doctor_runs():
    """doctor 命令可运行。"""
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "doctor" in result.output


# ------------------------- 端到端（synthetic，离线）-------------------------
def test_end_to_end_synthetic(tmp_path, monkeypatch):
    """synthetic 数据走完 fetch -> sample -> train -> backtest -> predict。"""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    sym = "600519"
    start, end = "20200101", "20241231"

    # 1) data fetch（合成数据，无需网络）
    r = runner.invoke(cli, [
        "data", "fetch", "--symbols", sym, "--start", start, "--end", end,
        "--source", "synthetic", "--config", "configs/default.yaml",
    ])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "data" / "raw" / f"{sym}.parquet").exists()

    # 2) sample build
    r = runner.invoke(cli, [
        "sample", "build", "--symbols", sym,
        "--train-start", "20200101", "--train-end", "20221231",
        "--test-start", "20230101", "--test-end", "20241231",
        "--window", "10", "--seed", "42", "--config", "configs/default.yaml",
    ])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "data" / "samples" / "train.npz").exists()
    assert (tmp_path / "data" / "samples" / "test.npz").exists()
    assert (tmp_path / "data" / "samples" / "meta.json").exists()

    # 3) train（小步数）
    r = runner.invoke(cli, [
        "train", "--algo", "ppo", "--timesteps", "300", "--seed", "42",
        "--config", "configs/default.yaml",
    ])
    assert r.exit_code == 0, r.output
    models = list((tmp_path / "models").glob("ppo_*.zip"))
    assert len(models) == 1, f"未生成模型: {list((tmp_path/'models').iterdir())}"
    model_path = models[0]

    # 4) backtest
    r = runner.invoke(cli, [
        "backtest", "--model", str(model_path), "--config", "configs/default.yaml",
    ])
    assert r.exit_code == 0, r.output
    assert len(list((tmp_path / "reports").glob("backtest_*.json"))) == 1

    # 5) predict
    r = runner.invoke(cli, [
        "predict", "--model", str(model_path), "--symbols", sym,
        "--start", "20240101", "--end", "20241231",
        "--config", "configs/default.yaml",
    ])
    assert r.exit_code == 0, r.output
    assert len(list((tmp_path / "reports").glob("predict_*.csv"))) == 1
    assert len(list((tmp_path / "reports").glob("predict_*.json"))) == 1


def test_data_hash_reproducible(tmp_path, monkeypatch):
    """相同输入两次 sample build 的 data_hash 完全一致（G2）。"""
    import json

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    runner.invoke(cli, ["data", "fetch", "--symbols", "600519",
                        "--start", "20200101", "--end", "20241231",
                        "--source", "synthetic"])
    args = ["sample", "build", "--symbols", "600519",
            "--train-start", "20200101", "--train-end", "20221231",
            "--test-start", "20230101", "--test-end", "20241231", "--window", "10"]

    runner.invoke(cli, args)
    meta1 = json.loads((tmp_path / "data" / "samples" / "meta.json").read_text(encoding="utf-8"))
    runner.invoke(cli, args)
    meta2 = json.loads((tmp_path / "data" / "samples" / "meta.json").read_text(encoding="utf-8"))
    assert meta1["data_hash"] == meta2["data_hash"]
    assert meta1["normalizer"]["mean"] == meta2["normalizer"]["mean"]
