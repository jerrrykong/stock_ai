"""奖励函数诊断：在训练切分上比较固定策略的累计奖励，判断「空仓」是否为局部最优。

三种策略：
  cash      全程 action 0（永不买入）
  buyhold   首步 action 1，其后 action 0
  model     已训练的 v2 模型

直接累加环境返回的 reward，不靠推测。
"""
from __future__ import annotations

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


def rollout(env, policy, model=None):
    obs, _ = env.reset()
    total = 0.0
    done = False
    step = 0
    while not done:
        if policy == "cash":
            action = 0
        elif policy == "buyhold":
            action = 1 if step == 0 else 0
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
        _, reward, terminated, _, _ = env.step(action)
        total += float(reward)
        done = bool(terminated)
        step += 1
    return total


def main() -> int:
    cfg = EnvConfig()
    cfg.reward_type = "excess"
    data = build_dataset(ROOT / "data" / "raw_hfq_v2_train", 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")
    from stable_baselines3 import PPO
    model = PPO.load(str(RUN / "model.zip"))

    sample = data.strict_pools["train"][:60]
    results = {"cash": [], "buyhold": [], "model": []}
    for uid, obs, bars in sample:
        for policy in ("cash", "buyhold", "model"):
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS,
                                  _with_benchmark=False)
            results[policy].append(rollout(env, policy, model))
            env.close()

    print(f"训练切分前 {len(sample)} 个 episode 的累计奖励均值：")
    for policy in ("cash", "buyhold", "model"):
        values = results[policy]
        print(f"  {policy:9s}: {np.mean(values):+.4f}  (中位数 {np.median(values):+.4f}, "
              f">0 占比 {sum(1 for v in values if v > 0) / len(values):.1%})")
    print()
    best = max(results, key=lambda k: float(np.mean(results[k])))
    print(f"均值最高策略: {best}")
    win = sum(1 for c, b in zip(results["cash"], results["buyhold"]) if c > b)
    print(f"cash 优于 buyhold 的 episode: {win}/{len(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
