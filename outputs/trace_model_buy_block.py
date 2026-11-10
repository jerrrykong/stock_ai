"""逐步追踪：model 为什么发起买入却被「现金不足」阻断。

打印单个 episode 前若干步的 action / blocked_reason / cash / shares / 价格，
定位现金到底去了哪里。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_ai.config.schema import EnvConfig  # noqa: E402
from stock_ai.scale.unified import (  # noqa: E402
    DEFAULT_RANGES, EXECUTION_CORPORATE_ACTIONS, build_dataset, make_strict_env,
)

RUN = ROOT / "models" / "unified" / "20261019_105053_7383897c"


def training_cfg() -> EnvConfig:
    params = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["reward_params"]
    cfg = EnvConfig()
    cfg.reward_type = params["reward_type"]
    cfg.lambda_drawdown = float(params["lambda_drawdown"])
    cfg.lambda_turnover = float(params["lambda_turnover"])
    cfg.lambda_leverage = float(params["lambda_leverage"])
    return cfg


def main() -> int:
    cfg = training_cfg()
    data = build_dataset(ROOT / "data" / "raw_hfq_v2_train", 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")
    from stable_baselines3 import PPO
    model = PPO.load(str(RUN / "model.zip"))

    sample = data.strict_pools["train"][:6]
    for uid, obs, bars in sample:
        env = make_strict_env(obs, bars, cfg,
                              execution_version=EXECUTION_CORPORATE_ACTIONS)
        o, info = env.reset()
        print(f"\n===== {uid}  n_steps={env.n_steps}  "
              f"首日开盘={float(env.bars.open[1]):.2f} 首日收盘={float(env.bars.close[0]):.2f} =====")
        print(f"{'步':>4s} {'日期':>9s} {'动作':>4s} {'阻断':>34s} {'现金':>12s} {'股数':>8s} "
              f"{'开盘':>9s} {'1手金额':>11s}")
        done = False
        step = 0
        shown = 0
        while not done:
            action, _ = model.predict(o, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            o, reward, terminated, _, info = env.step(action)
            if shown < 45 or info["trade_ok"]:
                price = float(info["open_price"])
                print(f"{step:4d} {info['execution_date']:>9s} {action:4d} "
                      f"{info['blocked_reason'][:34]:>34s} {info['cash']:12.0f} "
                      f"{info['shares']:8d} {price:9.2f} {price * 100:11.0f}")
                shown += 1
            done = bool(terminated)
            step += 1
        print(f"  共 {step} 步；成交流水 {len(env.trades)} 笔")
        for t in env.trades[:8]:
            print(f"    {t['date']} {t['side']:4s} {t['shares']:8d} 股 @ {t['price']:.2f}"
                  f"  手续费 {t['fee']:.2f}")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
