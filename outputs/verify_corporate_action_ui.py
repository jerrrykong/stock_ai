"""隔离真实浏览器验收：合成行情及账本、假 policy，不触发任何训练。

当前 UI 仅在随机回环端口提供服务；所有模型和行情 fixture 位于 outputs 临时目录，
不会读写生产 data/models，也不操作已有服务。页面控制请求由浏览器拦截，
独立后端拒绝门另行验证；每条浏览器操作及失败均留存证据。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import threading
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

print("[隔离浏览器验收] 正在加载 NumPy / Pandas 与项目只读回放组件", flush=True)
import numpy as np
import pandas as pd

from stock_ai.config.schema import AppConfig
from stock_ai.samples.normalizer import NormalizerState
from stock_ai.scale import replay, unified
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.ui.server import ScaleUI

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NODE = Path("C:/Users/jerrykong/.workbuddy/binaries/node/versions/22.22.2-6/node.exe")
CHROME = Path("C:/Users/jerrykong/.agent-browser/browsers/chrome-146.0.7680.153/chrome-win64/chrome.exe")
PLAYWRIGHT = Path("C:/Users/jerrykong/.workbuddy/binaries/node/workspace/node_modules/playwright-core")
BROWSER_SCRIPT = OUT / "verify_corporate_action_browser.cjs"


class FakeModel:
    obs_shape = (3, len(unified.FEATURES))
    n_actions = 3

    def __init__(self):
        self.index = 0

    def predict(self, obs, deterministic=True):
        assert deterministic and obs.shape == self.obs_shape
        actions = [1, 0, 2, 0]
        action = actions[self.index] if self.index < len(actions) else 0
        self.index += 1
        return action, None


class ReadOnlyUI(ScaleUI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_attempts = []

    def unified_control(self, payload):
        self.control_attempts.append(payload)
        return {"ok": False, "msg": "Isolated acceptance: all training controls denied"}

    def control(self, phase, action):
        self.control_attempts.append({"phase": phase, "action": action})
        return {"ok": False, "msg": "Isolated acceptance: all training controls denied"}

    def start(self, phase):
        raise AssertionError("Training subprocesses are forbidden")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(base):
    cfg = AppConfig()
    cfg.paths.root = base
    cfg.paths.raw = base / "data/raw"
    cfg.data.cache_dir = cfg.paths.raw
    feature_dir, ledger_dir = base / "data/raw_hfq", base / "data/corporate_actions"
    for directory in (cfg.paths.raw, feature_dir, ledger_dir):
        directory.mkdir(parents=True)
    dates = pd.bdate_range("2023-01-02", periods=80)
    prices = 10 + np.arange(80) * .05
    raw = pd.DataFrame({"date": dates, "open": prices - .02, "high": prices + .1,
                        "low": prices - .1, "close": prices, "volume": 1000 + np.arange(80)})
    hfq = raw.copy()
    hfq.loc[:, ["open", "high", "low", "close"]] *= np.repeat((2 + np.arange(80) * .025)[:, None], 4, axis=1)
    hfq["volume"] = 2000 + np.arange(80) ** 2
    reference = round((raw.close.iloc[31] - 1) / 1.5, 2)
    for index in range(32, 80):
        raw.loc[index, ["open", "high", "low", "close"]] = reference + (index - 32) * .02
    raw_path, feature_path = cfg.paths.raw / "600519.parquet", feature_dir / "600519.parquet"
    raw.to_parquet(raw_path, index=False)
    hfq.to_parquet(feature_path, index=False)
    day = lambda index: dates[index].strftime("%Y%m%d")
    ledger = {"schema": "corporate-actions-v1", "symbol": "600519",
              "coverage": {"start": day(0), "end": day(79), "status": "verified",
                           "scope": "all_shareholder_distributions_and_capital_changes",
                           "sources": ["https://invalid.example/synthetic-coverage"],
                           "verified_at": "2026-10-16", "notes": "SYNTHETIC MECHANISM FIXTURE ONLY; NOT REAL 600519 DATA"},
              "events": [{"event_id": "SYNTHETIC-distribution", "kind": "distribution", "status": "implemented",
                          "record_date": day(31), "ex_date": day(32), "pay_date": day(34), "listing_date": day(34),
                          "cash_per_share_gross": "1", "bonus_ratio": "0", "transfer_ratio": "0.5",
                          "reference_method": "standard_cash_bonus", "fractional_policy": "reject",
                          "sources": ["https://invalid.example/synthetic-distribution"]}]}
    ledger_path = ledger_dir / "600519.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    norm = NormalizerState(np.linspace(-.02, .02, len(unified.FEATURES)),
                           np.linspace(.03, .1, len(unified.FEATURES)), list(unified.FEATURES))
    env = asdict(cfg.env)
    env.update(initial_cash=100000., reward_type="excess")
    for run_id, version in (("ca_fixture", unified.EXECUTION_CORPORATE_ACTIONS), ("v1_fixture", unified.EXECUTION_STRICT)):
        directory = base / "models/unified" / run_id
        directory.mkdir(parents=True)
        source = {"symbol": "600519", "sha256": sha(feature_path)}
        if version == unified.EXECUTION_CORPORATE_ACTIONS:
            source.update(execution_sha256=sha(raw_path), corporate_actions_sha256=sha(ledger_path))
        manifest = {"run_id": run_id, "mode": "unified", "contract": unified.CONTRACT, "backend": "sb3", "algo": "ppo",
                    "window": 3, "features": unified.FEATURES, "observation_shape": [3, len(unified.FEATURES)],
                    "ranges": {"train": [2006, 2021], "val": [2022, 2023], "test": [2024, 2025]},
                    "env_config": env, "execution_version": version, "feature_adjust": "hfq",
                    "execution_adjust": "none(真实未复权价)", "feature_dir": str(feature_dir),
                    "execution_dir": str(cfg.paths.raw), "corporate_actions_dir": "data/corporate_actions",
                    "reward_type": "excess", "reward_version": "excess-ca-constant-buy-v2" if version == unified.EXECUTION_CORPORATE_ACTIONS else "excess-causal-close-v1",
                    "reward_params": {key: env[key] for key in ("reward_type", "lambda_drawdown", "lambda_turnover", "lambda_leverage")},
                    "created_at": "2026-10-16T20:00:00" if run_id == "ca_fixture" else "2026-10-15T20:00:00",
                    "sources": [source]}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "normalizer.json").write_text(json.dumps(norm.to_dict()), encoding="utf-8")
        (directory / "model.zip").write_bytes(b"FAKE POLICY FOR BROWSER ACCEPTANCE; NOT A TRAINED MODEL")
    return cfg, day


def run(target: Path | None = None):
    target = (target or OUT / "corporate_action_ui_acceptance_final").resolve()
    target.mkdir(exist_ok=False)
    process_env = dict(os.environ)
    process_env["PATH"] = str(NODE.parent) + os.pathsep + process_env.get("PATH", "")
    process_env["PYTHONDONTWRITEBYTECODE"] = "1"
    process_env["PYTHONIOENCODING"] = "utf-8"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        process_env[key] = "1"
    results = {"fixture_type": "合成机制验收：假 policy、假行情及账本，不是真实 600519 业绩；不触发训练",
               "generated_at": datetime.now().astimezone().isoformat(), "checks": {},
               "passed": False, "browser_closed": False, "server_closed": False}

    def save_json(filename, value):
        (target / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def progress(message):
        line = json.dumps({"time": datetime.now().astimezone().isoformat(), "message": message}, ensure_ascii=False)
        with (target / "script_progress.jsonl").open("a", encoding="utf-8") as log:
            log.write(line + "\n")
        print(line, flush=True)

    def fingerprint(base):
        return {p.relative_to(base).as_posix(): {"sha256": sha(p), "mtime_ns": p.stat().st_mtime_ns,
                                                "size": p.stat().st_size}
                for p in sorted(base.rglob("*")) if p.is_file()}

    original = replay._load_model
    server = thread = ui = None
    error = None
    try:
        progress("建立 outputs 内隔离合成 fixture；不读取生产模型，不连接 8766/8767")
        with tempfile.TemporaryDirectory(prefix="ca-ui-fixture-final-", dir=OUT) as temporary:
            cfg, day = fixture(Path(temporary))
            results["fixture_root"] = str(cfg.paths.root)
            replay._load_model = lambda _: FakeModel()
            ui = ReadOnlyUI(cfg, ScalePaths(cfg.paths.root), host="127.0.0.1", port=0)
            server = ui.make_server()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            start, end = (pd.Timestamp(day(index)).strftime("%Y-%m-%d") for index in (30, 34))
            url = (f"http://127.0.0.1:{server.server_port}/?model_id=unified:ca_fixture"
                   f"&symbol=600519&start={start}&end={end}#replay")
            results["isolated_server"] = url
            save_json("browser_config.json", {"url": url, "start": start, "end": end,
                                               "output_dir": str(target), "chrome": str(CHROME),
                                               "playwright": str(PLAYWRIGHT)})
            before = fingerprint(cfg.paths.root)
            save_json("fixture_before.json", before)
            try:
                progress(f"启动单次 Playwright 辅助进程；逐命令进度见 browser_progress.jsonl；端口 {server.server_port}")
                # 文件句柄不会因 daemon 子进程继承 stdout 管道而卡在 communicate()。
                with (target / "browser_stdout.jsonl").open("w", encoding="utf-8") as stdout, \
                        (target / "browser_stderr.log").open("w", encoding="utf-8") as stderr:
                    child = subprocess.run([str(NODE), str(BROWSER_SCRIPT), str(target / "browser_config.json")],
                                           env=process_env, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, timeout=210)
                results["browser_exit_code"] = child.returncode
                browser_results = json.loads((target / "browser_results.json").read_text(encoding="utf-8"))
                results["browser_closed"] = browser_results.get("browser_closed", False)
                results["checks"].update(browser_results.get("checks", {}))
                results["page_errors"] = browser_results.get("page_errors", [])
                results["console_errors"] = browser_results.get("console_errors", [])
                assert child.returncode == 0 and browser_results["passed"], browser_results.get("error", "浏览器验收失败")
                assert results["browser_closed"]
                progress("浏览器验收已结束，交叉校验实际下载的 CSV 与完整 JSON")
                exported = json.loads((target / "replay.json").read_text(encoding="utf-8-sig"))
                daily = list(csv.DictReader(io.StringIO((target / "daily.csv").read_text(encoding="utf-8-sig"))))
                trades = list(csv.DictReader(io.StringIO((target / "trades.csv").read_text(encoding="utf-8-sig"))))
                assert exported["model"]["id"] == "unified:ca_fixture"
                assert exported["execution_version"] == unified.EXECUTION_CORPORATE_ACTIONS
                assert len(exported["daily"]) == len(daily) == 5
                assert len(exported["trades"]) == len(trades) == 2
                assert exported["source"]["training_source_audit"]["status"] == "matched"
                for csv_row in daily + trades:
                    assert csv_row["模型"] == "unified:ca_fixture"
                    assert csv_row["执行版本"] == unified.EXECUTION_CORPORATE_ACTIONS
                    assert csv_row["训练执行版本"] == exported["training_execution_version"]
                    for heading, key in (("公司行为账本SHA256", "corporate_actions_sha256"),
                                         ("当前后复权SHA256", "feature_sha256"),
                                         ("当前未复权SHA256", "execution_sha256")):
                        assert csv_row[heading] == exported["source"][key]
                    assert json.loads(csv_row["冻结训练来源核对_JSON"]) == exported["source"]["training_source_audit"]
                for csv_row, json_row in zip(daily, exported["daily"]):
                    assert csv_row["日期"] == json_row["date"]
                    for heading, key in (("权益_CNY", "equity"), ("现金_CNY", "cash"), ("持股数", "shares"),
                                         ("可卖股数", "available_shares"), ("待上市锁股数", "locked_shares"),
                                         ("股息应收_CNY", "dividend_receivable")):
                        assert np.isclose(float(csv_row[heading]), json_row[key])
                    assert json.loads(csv_row["公司行为完整事件_JSON"]) == json_row["corporate_actions"]
                    assert np.isclose(json_row["equity"], json_row["cash"] + json_row["shares"] * json_row["close"]
                                      + json_row["dividend_receivable"])
                for csv_row, json_row in zip(trades, exported["trades"]):
                    assert csv_row["成交日"] == json_row["date"] and csv_row["方向"] == json_row["side"]
                    assert np.isclose(float(csv_row["成交股数"]), json_row["shares"])
                    assert np.isclose(float(csv_row["成交金额_CNY"]), json_row["notional"])
                assert exported["daily"][2]["locked_shares"] > 0 and exported["daily"][2]["dividend_receivable"] > 0
                assert exported["daily"][3]["available_shares"] == 0 and exported["daily"][3]["locked_shares"] > 0
                controls = [item["payload"] for item in browser_results["control_payloads"]]
                assert ui.control_attempts == controls and len(controls) == 2 and ui.procs == {}
                results.update(row_count=len(daily), trade_count=len(trades), control_payloads=controls,
                               backend_control_attempts=ui.control_attempts, training_process_count=len(ui.procs))
                results["checks"].update(csv_json_bound_to_completed_result=True, full_training_hashes_matched=True,
                                         sell_only_available_shares=True, accounting_identity=True,
                                         no_training_process_started=True)
                results["passed"] = True
            finally:
                try:
                    progress("finally 关闭随机端口隔离服务器")
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)
                    results["server_closed"] = not thread.is_alive() and server.socket.fileno() == -1
                    assert results["server_closed"], "隔离服务器未完成关闭"
                finally:
                    after = fingerprint(cfg.paths.root)
                    save_json("fixture_after.json", after)
                    results["fixture_files_unchanged"] = before == after
                    results["checks"]["fixture_hash_mtime_unchanged"] = before == after
                    results["fixture_file_count"] = len(before)
                    assert before == after, "隔离 fixture 的 SHA256、mtime 或文件集合改变"
    except BaseException as exc:
        error = exc
        results["passed"] = False
        results["error"] = traceback.format_exc()
    finally:
        replay._load_model = original
        if server is not None and not results["server_closed"]:
            server.shutdown()
            server.server_close()
            if thread is not None:
                thread.join(timeout=5)
            results["server_closed"] = (thread is None or not thread.is_alive()) and server.socket.fileno() == -1
        if (target / "browser_results.json").exists():
            browser_results = json.loads((target / "browser_results.json").read_text(encoding="utf-8"))
            results["browser_closed"] = browser_results.get("browser_closed", False)
        save_json("acceptance.json", results)
        progress(f"验收结束：passed={results['passed']}；浏览器已关闭={results['browser_closed']}；服务器已关闭={results['server_closed']}")
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="全新的证据目录；已存在时拒绝覆盖。")
    run(parser.parse_args().output_dir)
