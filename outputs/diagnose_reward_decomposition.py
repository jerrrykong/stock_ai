"""奖励分量拆解：把累计奖励拆成 excess / 回撤 / 换手 / 持仓四项贡献。

**配置必须与训练完全一致**（此前一版误用 EnvConfig() 默认值，λ_drawdown 用了
0.1 而训练是 0.0，结论被放大）。本版从 run manifest 读取真实 reward_params，
并且开启 _with_benchmark=True —— StrictEpisodePoolEnv 不传该参数，默认 True，
真实训练带同环境恒买基准。

不修改环境、不包装 RewardCalculator：所有分量从 step() 返回的 info 与
上一步的 portfolio_value 外部重构，最后用「分量之和 == 环境返回奖励之和」
做自校验，确保拆解与真实奖励逐位一致。
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

RUN = ROOT / "models" / "unified" / (
    sys.argv[1] if len(sys.argv) > 1 else "20261019_105053_7383897c")


def training_cfg() -> EnvConfig:
    """按 run manifest 的 reward_params 复原训练时的 EnvConfig。"""
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["reward_params"]
    cfg = EnvConfig()
    cfg.reward_type = params["reward_type"]
    cfg.lambda_drawdown = float(params["lambda_drawdown"])
    cfg.lambda_turnover = float(params["lambda_turnover"])
    cfg.lambda_leverage = float(params["lambda_leverage"])
    return cfg


def rollout_traced(env, policy, model=None):
    """跑一个 episode，逐步记录奖励的四个分量。返回 (总奖励, 分量字典)。

    注意：model 策略必须把 step() 返回的新观测喂回 predict ——
    复用 reset() 的初始观测会让模型退化成恒定动作策略（曾因此得出错误结论）。
    cash / buyhold 不依赖观测，不受此影响。
    """
    obs, info = env.reset()
    initial = float(info["portfolio_value"])
    peak = initial
    prev_value = initial
    comp = {"excess": 0.0, "drawdown": 0.0, "turnover": 0.0, "leverage": 0.0}
    n_steps = 0
    done = False
    step = 0
    reward_sum = 0.0
    bench_sum = 0.0
    pos_sum = 0.0
    dd_sum = 0.0
    actions = {0: 0, 1: 0, 2: 0}
    while not done:
        if policy == "cash":
            action = 0
        elif policy == "buyhold":
            action = 1 if step == 0 else 0
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
        actions[action] = actions.get(action, 0) + 1
        new_obs, reward, terminated, _, info = env.step(action)
        obs = new_obs  # 必须更新：下一步的预测要用当前观测
        value = float(info["portfolio_value"])
        ret = (value - prev_value) / prev_value if prev_value > 1e-9 else 0.0
        peak = max(peak, value)
        dd = max(0.0, (peak - value) / peak) if peak > 1e-9 else 0.0
        bench = float(info["benchmark_return"])
        turn = float(info["turnover"])
        pos = abs(float(info["position_ratio"]))
        cfg = env.cfg
        comp["excess"] += ret - bench
        comp["drawdown"] += -cfg.lambda_drawdown * dd
        comp["turnover"] += -cfg.lambda_turnover * turn
        comp["leverage"] += -cfg.lambda_leverage * pos
        bench_sum += bench
        dd_sum += dd
        pos_sum += pos
        reward_sum += float(reward)
        prev_value = value
        done = bool(terminated)
        step += 1
        n_steps = step
    comp["_reward_sum"] = reward_sum
    comp["_bench_sum"] = bench_sum
    comp["_steps"] = n_steps
    comp["_pos_sum"] = pos_sum
    comp["_dd_sum"] = dd_sum
    comp["_actions"] = actions
    return reward_sum, comp


def main() -> int:
    cfg = training_cfg()
    print("训练真实奖励配置（来自 manifest.reward_params）：")
    print(f"  reward_type      = {cfg.reward_type}")
    print(f"  lambda_drawdown  = {cfg.lambda_drawdown}")
    print(f"  lambda_turnover  = {cfg.lambda_turnover}")
    print(f"  lambda_leverage  = {cfg.lambda_leverage}")
    print(f"  基准             = 同环境恒买（_with_benchmark=True，与训练一致）")
    print()

    data = build_dataset(ROOT / "data" / "raw_hfq_v2_train", 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")
    from stable_baselines3 import PPO
    model = PPO.load(str(RUN / "model.zip"))

    sample = data.strict_pools["train"][:60]
    policies = ("cash", "buyhold", "model")
    agg = {p: {k: [] for k in ("excess", "drawdown", "turnover", "leverage",
                               "reward", "bench", "steps", "pos_sum", "dd_sum",
                               "actions")} for p in policies}
    mismatch = 0
    for uid, obs, bars in sample:
        for policy in policies:
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            reward, comp = rollout_traced(env, policy, model)
            env.close()
            parts = comp["excess"] + comp["drawdown"] + comp["turnover"] + comp["leverage"]
            if abs(parts - reward) > 1e-6:
                mismatch += 1
                print(f"  [自校验失败] {uid} {policy}: 分量之和 {parts:.6f} != 奖励 {reward:.6f}")
            for key in ("excess", "drawdown", "turnover", "leverage"):
                agg[policy][key].append(comp[key])
            agg[policy]["reward"].append(reward)
            agg[policy]["bench"].append(comp["_bench_sum"])
            agg[policy]["steps"].append(comp["_steps"])
            agg[policy]["pos_sum"].append(comp["_pos_sum"])
            agg[policy]["dd_sum"].append(comp["_dd_sum"])
            agg[policy]["actions"].append(comp["_actions"])

    n_eps = len(sample)
    print(f"共 {n_eps} 个 episode × {len(policies)} 个策略；"
          f"自校验不一致 {mismatch} 例（0 = 拆解与真实奖励逐位一致）")
    print()
    print(f"平均 episode 步数: {np.mean(agg['cash']['steps']):.1f}")
    print()

    header = (f"{'策略':9s} {'总奖励':>10s} {'excess':>10s} {'回撤项':>10s} "
              f"{'换手项':>10s} {'持仓项':>10s} {'bench合计':>11s}")
    print(header)
    print("-" * len(header))
    for policy in policies:
        a = agg[policy]
        print(f"{policy:9s} {np.mean(a['reward']):+10.4f} {np.mean(a['excess']):+10.4f} "
              f"{np.mean(a['drawdown']):+10.4f} {np.mean(a['turnover']):+10.4f} "
              f"{np.mean(a['leverage']):+10.4f} {np.mean(a['bench']):+11.4f}")
    print()

    # 反事实：把某一惩罚项置零后重算，看哪一项造成「空仓更优」
    print("反事实实验（把某一惩罚项置零后重算 buyhold 与 cash 的差距）：")
    base_gap = np.mean(agg["buyhold"]["reward"]) - np.mean(agg["cash"]["reward"])
    print(f"  现状                      : buyhold - cash = {base_gap:+.4f}")
    for name in ("drawdown", "turnover", "leverage"):
        bh_total = np.array(agg["buyhold"]["reward"]) - np.array(agg["buyhold"][name])
        cash_total = np.array(agg["cash"]["reward"]) - np.array(agg["cash"][name])
        gap = float(np.mean(bh_total) - np.mean(cash_total))
        print(f"  去掉 {name:9s}惩罚      : buyhold - cash = {gap:+.4f}")
    print()

    bh_lev = -np.mean(agg["buyhold"]["leverage"])
    bh_exc = np.mean(agg["buyhold"]["excess"])
    bench_abs = np.mean(np.abs(agg["buyhold"]["bench"]))
    steps = float(np.mean(agg["buyhold"]["steps"]))
    print("量级对比（buyhold，单位：奖励点数/年）：")
    print(f"  持仓惩罚年化成本          : {bh_lev:+.4f}")
    print(f"  超额收益（含费用磨损）    : {bh_exc:+.4f}")
    print(f"  同期基准收益绝对量级      : {bench_abs:+.4f}")
    print(f"  持仓惩罚 / |基准收益|     : {bh_lev / bench_abs:.2f} 倍")
    print(f"  平均仓位                  : {np.mean(agg['buyhold']['pos_sum']) / steps:.3f}"
          f"（≈1 表示全程满仓）")
    print(f"  满仓一步的理论惩罚        : {-cfg.lambda_leverage:+.4f}"
          f"  vs 单步基准收益典型量级 {bench_abs / steps:+.6f}")
    print()

    a = agg["model"]
    buys = np.mean([x.get(1, 0) for x in a["actions"]])
    sells = np.mean([x.get(2, 0) for x in a["actions"]])
    holds = np.mean([x.get(0, 0) for x in a["actions"]])
    print("模型动作分布（train 前 60 个 episode，每 episode 平均次数）：")
    print(f"  持有 {holds:8.1f}   买入 {buys:7.1f}   卖出 {sells:7.1f}")
    print(f"  模型持仓惩罚              : {np.mean(a['leverage']):+.4f}")
    print(f"  模型超额收益              : {np.mean(a['excess']):+.4f}")
    print(f"  模型总奖励                : {np.mean(a['reward']):+.4f}")
    print()

    # 在 val / test 切分上核对零交易现象（评估协议报 38% 零交易）
    for split in ("val", "test"):
        pool = data.strict_pools[split]
        rewards, excesses, zero, traded_ex, benches, steps_l = [], [], 0, [], [], []
        n_trade = []
        for uid, obs, bars in pool:
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            reward, comp = rollout_traced(env, "model", model)
            env.close()
            rewards.append(reward)
            excesses.append(comp["excess"])
            benches.append(comp["_bench_sum"])
            steps_l.append(comp["_steps"])
            nt = len(env.trades)
            n_trade.append(nt)
            if nt == 0:
                zero += 1
            else:
                traded_ex.append(comp["excess"])
        print(f"[{split}] {len(pool)} 个 episode：")
        print(f"  零交易 episode          : {zero} ({zero / len(pool):.1%})")
        print(f"  平均交易笔数            : {np.mean(n_trade):.1f}")
        print(f"  超额收益 全部           : {np.mean(excesses):+.4f}")
        print(f"  超额收益 仅有交易        : {np.mean(traded_ex):+.4f}"
              f"   ← 诚实口径")
        print(f"  超额收益 零交易         : "
              f"{np.mean([e for e, n in zip(excesses, n_trade) if n == 0]):+.4f}"
              f"   ← 躺平假象")
        print(f"  基准收益合计            : {np.mean(benches):+.4f}")
        print(f"  模型总奖励              : {np.mean(rewards):+.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
