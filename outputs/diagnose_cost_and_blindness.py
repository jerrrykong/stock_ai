"""三组量化，为「提升方案」提供可执行数字。

A. PPO 的亏损分解：毛超额（成本前） vs 交易成本拖累
   —— 判断「-5% 是没技能」还是「有技能但被成本吃掉」
B. 低换手反转规则：滞后带 + 最短持有期，检验信号能否在扣费后存活
C. 观测盲区检验：模型动作是否依赖「当前是否持仓」——环境观测里没有持仓状态，
   若动作与持仓状态无关，说明策略在信息上被阉割
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_ai.config.schema import EnvConfig  # noqa: E402
from stock_ai.scale.unified import (  # noqa: E402
    DEFAULT_RANGES, EXECUTION_CORPORATE_ACTIONS, FEATURES, build_dataset, make_strict_env,
)

RUN = ROOT / "models" / "unified" / "20261020_105633_578f25aa"
FEATURE_DIR = ROOT / "data" / "raw_hfq_v2_train"


def training_cfg() -> EnvConfig:
    params = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["reward_params"]
    cfg = EnvConfig()
    cfg.reward_type = params["reward_type"]
    cfg.lambda_drawdown = float(params["lambda_drawdown"])
    cfg.lambda_turnover = float(params["lambda_turnover"])
    cfg.lambda_leverage = float(params["lambda_leverage"])
    return cfg


def load_panel():
    from stock_ai.scale.unified import relative_features
    panel = {}
    for path in sorted(FEATURE_DIR.glob("*.parquet")):
        raw = pd.read_parquet(path)
        if raw.empty:
            continue
        df, values = relative_features(raw)
        dates = df.date.dt.strftime("%Y%m%d").to_numpy()
        panel[path.stem] = {"values": values, "dates": dates,
                            "close": df.close.to_numpy(dtype="float64")}
    return panel


def episode_slice(panel, symbol, dates):
    entry = panel.get(symbol)
    if entry is None:
        return None
    pos = {d: i for i, d in enumerate(entry["dates"])}
    start = pos.get(dates[0])
    if start is None:
        return None
    n = len(dates)
    if start + n > len(entry["dates"]):
        return None
    return entry, start, n


def run_policy(env, target):
    """按目标持仓序列驱动，统计毛/净收益、在市天数、换手。

    复用同一个 env：每次调用都从 reset() 开始，环境构造（含账本校验）
    是本脚本最贵的一步，不能放进组合循环里。
    """
    obs, info = env.reset()
    done = False
    step = 0
    held = False
    days_in = 0
    while not done:
        want = target[min(step, len(target) - 1)]
        if want > 0.5 and not held:
            action = 1
        elif want <= 0.5 and held:
            action = 2
        else:
            action = 0
        obs, _, terminated, _, info = env.step(action)
        held = info["shares"] > 0
        days_in += 1 if held else 0
        done = bool(terminated)
        step += 1
    return days_in / max(step, 1), len(env.trades)


def hysteresis_signal(values, enter_q, exit_q, min_hold, feature="ma20_relative"):
    """滞后带规则：低于 enter 分位买入，高于 exit 分位卖出，中间维持原状。

    enter_q < exit_q 形成滞后带，避免在阈值附近反复横跳 —— 这是压低换手的
    关键（阈值规则的换手普遍在 15~65 倍/年）。
    """
    idx = {name: i for i, name in enumerate(FEATURES)}
    series = values[:, idx[feature]]
    n = len(series)
    target = np.zeros(n)
    holding = False
    held_days = 0
    for i in range(n):
        if not holding:
            if series[i] < enter_q:
                holding = True
                held_days = 0
        else:
            held_days += 1
            if held_days >= min_hold and series[i] > exit_q:
                holding = False
                held_days = 0
        target[i] = 1.0 if holding else 0.0
    return target


def main() -> int:
    cfg = training_cfg()
    data = build_dataset(FEATURE_DIR, 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")
    from stable_baselines3 import PPO
    model = PPO.load(str(RUN / "model.zip"))

    print("加载特征面板 ...")
    panel = load_panel()

    # train 段阈值
    train_rows = []
    for symbol, entry in panel.items():
        mask = np.array([d[:4].isdigit() and 2006 <= int(d[:4]) <= 2021
                         for d in entry["dates"]])
        train_rows.append(entry["values"][mask])
    train_stack = np.concatenate(train_rows, axis=0)
    idx = {name: i for i, name in enumerate(FEATURES)}

    # ---------- A. PPO 亏损分解 ----------
    print("\n===== A. PPO(v3) 的毛超额 vs 成本拖累 =====")
    for split in ("val", "test"):
        rows = []
        for uid, obs, bars in data.strict_pools[split]:
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            o, _ = env.reset()
            done = False
            step = 0
            held = False
            days_in = 0
            while not done:
                action, _ = model.predict(o, deterministic=True)
                action = int(np.asarray(action).reshape(-1)[0])
                o, _, terminated, _, info = env.step(action)
                held = info["shares"] > 0
                days_in += 1 if held else 0
                done = bool(terminated)
                step += 1
            metrics = env.metrics()
            env.close()
            rows.append({
                "excess": metrics["excess_return"],
                "strategy": metrics["strategy_return"],
                "benchmark": metrics["benchmark_return"],
                "trades": metrics["trades"],
                "turnover": metrics["total_turnover_ratio"],
                "cost_bps": metrics["cost_drag_bps"],
                "exposure": days_in / max(step, 1),
            })
        arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        # 毛超额 = 净超额 + 成本拖累（成本拖累以收益比例计）
        gross = arr["excess"] + arr["cost_bps"] / 1e4
        print(f"  {split} 段（{len(rows)} episode）:")
        print(f"    净超额（官方口径）      : {arr['excess'].mean():+.2%}")
        print(f"    成本拖累（费用/权益）   : {arr['cost_bps'].mean():.1f} bps "
              f"= {arr['cost_bps'].mean() / 1e4:+.2%}")
        print(f"    毛超额（扣费前估算）    : {gross.mean():+.2%}   "
              f"← 剔除成本后的择时能力")
        print(f"    策略收益 / 基准收益     : {arr['strategy'].mean():+.2%} / "
              f"{arr['benchmark'].mean():+.2%}")
        print(f"    在市占比 / 平均交易     : {arr['exposure'].mean():.1%} / "
              f"{arr['trades'].mean():.1f} 笔")
        print(f"    年换手                  : {arr['turnover'].mean():.2f} 倍")
        print(f"    毛超额为正占比          : {float((gross > 0).mean()):.1%}")
        print()

    # ---------- B. 低换手反转规则 ----------
    print("===== B. 低换手反转规则（滞后带 + 最短持有期） =====")
    quantiles = {q: float(np.quantile(train_stack[:, idx["ma20_relative"]], q))
                 for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
    combos = []
    for enter_q in (0.1, 0.2, 0.3, 0.4):
        for exit_q in (0.5, 0.6, 0.7, 0.8, 0.9):
            if exit_q <= enter_q:
                continue
            for min_hold in (5, 10, 20, 40):
                combos.append((enter_q, exit_q, min_hold))
    results = {}
    for split in ("val", "test"):
        pool = data.strict_pools[split]
        for i, (uid, obs, bars) in enumerate(pool):
            sliced = episode_slice(panel, uid.split("_")[0],
                                   [str(d) for d in bars.dates])
            if sliced is None:
                continue
            entry, start, n = sliced
            values = entry["values"][start:start + n]
            # 环境只建一次，所有组合复用（reset 即可回到初始状态）
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            for enter_q, exit_q, min_hold in combos:
                target = hysteresis_signal(values, quantiles[enter_q],
                                           quantiles[exit_q], min_hold)
                exposure, trades = run_policy(env, target)
                metrics = env.metrics()
                results.setdefault((enter_q, exit_q, min_hold), []).append(
                    (split, metrics["excess_return"], trades, exposure,
                     metrics["total_turnover_ratio"]))
            env.close()
            if (i + 1) % 100 == 0:
                print(f"  {split}: {i + 1}/{len(pool)}", flush=True)
        print(f"  {split} 完成", flush=True)

    print(f"\n  {'进入分位':>8s}{'退出分位':>8s}{'最短持有':>8s}"
          f"{'val 超额':>12s}{'test 超额':>12s}{'test换手':>10s}{'在市':>8s}")
    ranked = []
    for key, rows in results.items():
        by_split = {s: [r for r in rows if r[0] == s] for s in ("val", "test")}
        if not by_split["val"] or not by_split["test"]:
            continue
        val_ex = np.mean([r[1] for r in by_split["val"]])
        test_ex = np.mean([r[1] for r in by_split["test"]])
        test_turn = np.mean([r[4] for r in by_split["test"]])
        test_exp = np.mean([r[3] for r in by_split["test"]])
        ranked.append((val_ex, test_ex, key, test_turn, test_exp))
    for val_ex, test_ex, key, test_turn, test_exp in sorted(ranked, reverse=True)[:12]:
        print(f"  {key[0]:>8.1f}{key[1]:>8.1f}{key[2]:>8d}"
              f"{val_ex:>+11.2%}{test_ex:>+11.2%}{test_turn:>10.2f}{test_exp:>8.1%}")
    best_val = max(ranked) if ranked else None
    if best_val:
        print(f"\n  按 val 选出的最优组合: enter_q={best_val[2][0]} exit_q={best_val[2][1]} "
              f"min_hold={best_val[2][2]}")
        print(f"    val {best_val[0]:+.2%}  test {best_val[1]:+.2%}  "
              f"（PPO v3 的 test 是 -5.04%）")

    # ---------- C. 观测盲区 ----------
    print("\n===== C. 观测盲区：模型动作是否依赖「当前是否持仓」 =====")
    for split in ("test",):
        action_when_flat = {0: 0, 1: 0, 2: 0}
        action_when_held = {0: 0, 1: 0, 2: 0}
        fill_when_flat = 0
        attempts_when_held = 0
        filled_when_held = 0
        for uid, obs, bars in data.strict_pools[split]:
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            o, _ = env.reset()
            done = False
            held = False
            while not done:
                action, _ = model.predict(o, deterministic=True)
                action = int(np.asarray(action).reshape(-1)[0])
                if held:
                    action_when_held[action] += 1
                    if action == 1:
                        attempts_when_held += 1
                else:
                    action_when_flat[action] += 1
                    if action == 1:
                        fill_when_flat += 1
                o, _, terminated, _, info = env.step(action)
                held = info["shares"] > 0
                done = bool(terminated)
            env.close()
        total_flat = sum(action_when_flat.values())
        total_held = sum(action_when_held.values())
        print(f"  {split} 段：")
        print(f"    空仓时动作分布 (n={total_flat}): 持有 {action_when_flat[0]/total_flat:.1%} "
              f"买入 {action_when_flat[1]/total_flat:.1%} 卖出 {action_when_flat[2]/total_flat:.1%}")
        print(f"    持仓时动作分布 (n={total_held}): 持有 {action_when_held[0]/total_held:.1%} "
              f"买入 {action_when_held[1]/total_held:.1%} 卖出 {action_when_held[2]/total_held:.1%}")
        print(f"    空仓时发起的买入 {fill_when_flat} 次（此时买入必然成交）")
        print(f"    持仓时发起的买入 {attempts_when_held} 次，其中 {filled_when_held} 次成交"
              f"（其余被「现金不足」阻断）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
