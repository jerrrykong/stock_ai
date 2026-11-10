"""排除性检查：v2 的零交易 episode 是奖励导致的策略选择，还是买入被结构性阻断？

对同一批 episode 分别跑 cash / buyhold / model，统计 info['blocked_reason']
分布。若 buyhold 能正常买入而 model 选择不买，则零交易是策略选择而非环境阻断。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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


def rollout(env, policy, model=None):
    obs, _ = env.reset()
    blocked = Counter()
    buys_attempted = 0
    buys_filled = 0
    done = False
    step = 0
    while not done:
        if policy == "buyhold":
            action = 1 if step == 0 else 0
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
        if action == 1:
            buys_attempted += 1
        _, _, terminated, _, info = env.step(action)
        if info["trade_ok"] and info["action"] == 1:
            buys_filled += 1
        if info["blocked_reason"] and info["blocked_reason"] != "reset":
            blocked[info["blocked_reason"]] += 1
        done = bool(terminated)
        step += 1
    return blocked, buys_attempted, buys_filled


def main() -> int:
    cfg = training_cfg()
    data = build_dataset(ROOT / "data" / "raw_hfq_v2_train", 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")
    from stable_baselines3 import PPO
    model = PPO.load(str(RUN / "model.zip"))

    sample = data.strict_pools["train"][:30]
    stats = {}
    for policy in ("buyhold", "model"):
        blocked = Counter()
        attempted = filled = 0
        for uid, obs, bars in sample:
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            b, a, f = rollout(env, policy, model)
            env.close()
            blocked += b
            attempted += a
            filled += f
        stats[policy] = (blocked, attempted, filled)
        print(f"=== {policy}（{len(sample)} 个 episode）===")
        print(f"  发起买入 {attempted} 次，成交 {filled} 次，"
              f"买入被阻断 {attempted - filled} 次")
        print(f"  买入成功率: {filled / max(attempted, 1):.1%}")
        print(f"  被阻断原因分布:")
        for reason, n in blocked.most_common(12):
            print(f"    {n:6d}  {reason}")
        print()

    # 关键判据：model 若极少发起买入，则零交易是策略选择
    m_blocked, m_att, m_fill = stats["model"]
    b_blocked, b_att, b_fill = stats["buyhold"]
    print("判据：")
    print(f"  buyhold 每 episode 平均发起买入 {b_att / len(sample):.2f} 次、"
          f"成交 {b_fill / len(sample):.2f} 次")
    print(f"  model   每 episode 平均发起买入 {m_att / len(sample):.2f} 次、"
          f"成交 {m_fill / len(sample):.2f} 次")
    if m_att / len(sample) < 1.0:
        print("  → model 几乎不发起买入：零交易是**策略选择**，不是环境阻断")
    else:
        print("  → model 有发起买入，需进一步看成交流失")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
