"""Run the final corporate-action regression set with real learning hard-disabled."""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FILES = [
    "tests/test_corporate_action_env.py",
    "tests/test_corporate_action_adversarial.py",
    "tests/test_corporate_action_data.py",
    "tests/test_corporate_action_sources.py",
    "tests/test_execution_env.py",
    "tests/test_excess_reward.py",
    "tests/test_unified.py",
    "tests/test_rewards.py",
    "tests/test_scale_ui.py",
    "tests/test_scale_replay.py",
]
EXCLUDED = [
    "test_manifest_records_execution_version",
    "test_unified_excess_run_records_manifest",
    "test_real_sb3_train_resume_replay",
    "test_pause_resume_stop_checkpoint",
    "test_cli_aliases_and_short_train",
]
learning_attempts = []


def no_learning(*args, **kwargs):
    learning_attempts.append("Real learning was attempted")
    pytest.fail("No real learning is authorized in this acceptance run")


def main():
    # Guard both public wrappers and concrete algorithms. Synthetic tests can
    # monkeypatch their own fakes, but accidentally entering actual learning fails.
    from stock_ai.train import sb3_trainer
    from stock_ai.train.ppo_numpy import NumpyPPOTrainer
    from stable_baselines3 import DQN, PPO, SAC

    sb3_trainer.learn_unified = no_learning
    sb3_trainer.SB3Trainer.learn = no_learning
    NumpyPPOTrainer.learn = no_learning
    for algorithm in (PPO, SAC, DQN):
        algorithm.learn = no_learning
    junit = OUT / "corporate_action_regression_final.xml"
    summary = OUT / "corporate_action_regression_final.json"
    if junit.exists() or summary.exists():
        raise FileExistsError("Final regression evidence already exists; choose a new filename")
    arguments = ["-p", "no:cacheprovider", "-o", "addopts=", "-q", "--junitxml", str(junit),
                 *FILES, "-k", " and ".join("not " + name for name in EXCLUDED)]
    code = pytest.main(arguments)
    suites = ET.parse(junit).getroot().findall("testsuite")
    record = {"generated_at": datetime.now().astimezone().isoformat(), "exit_code": int(code),
              "files": FILES, "excluded_real_training_tests": EXCLUDED,
              "python": sys.executable, "python_version": sys.version,
              "learning_guard": True, "learning_attempts": learning_attempts,
              "tests": sum(int(s.get("tests", 0)) for s in suites),
              "failures": sum(int(s.get("failures", 0)) for s in suites),
              "errors": sum(int(s.get("errors", 0)) for s in suites),
              "skipped": sum(int(s.get("skipped", 0)) for s in suites),
              "duration_seconds": sum(float(s.get("time", 0)) for s in suites),
              "environment": {key: os.getenv(key) for key in
                              ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    summary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return int(code) or bool(learning_attempts)


if __name__ == "__main__":
    raise SystemExit(main())
