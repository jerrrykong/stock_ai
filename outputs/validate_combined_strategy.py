"""组合方案验证：regime 门控 + 滞后带反转，train 选参 → val/test 验证。

诊断已确立的三条事实：
  1. ma20_relative 的 20 日 IC 在牛熊之间**符号翻转**（牛 +0.023 / 熊 −0.080）
  2. 单一静态规则必然在某些 regime 失效（naive 规则 test −26.5%）
  3. 高换手（15~27 倍/年）是规则失效的第二杀手

本脚本检验：把「市场状态门控」与「滞后带 + 最短持有期」组合，
在 train 段选参后，val/test 段能拿到多少超额。

参��网格在 train 段上评估并按 train 超额排序，**不允许看 val/test 选参**。
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
IDX = {name: i for i, name in enumerate(FEATURES)}


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
        panel[path.stem] = {"values": values,
                            "dates": df.date.dt.strftime("%Y%m%d").to_numpy(),
                            "close": df.close.to_numpy(dtype="float64")}
    return panel


def build_market(panel):
    """197 只等权市场代理：日收益横截面均值累成净值，再算 20 日动量。"""
    rows = []
    for symbol, entry in panel.items():
        close = entry["close"]
        ret = np.zeros(len(close))
        ret[1:] = close[1:] / close[:-1] - 1.0
        rows.append(pd.DataFrame({"date": entry["dates"], "ret": ret}))
    frame = pd.concat(rows, ignore_index=True)
    daily = frame.groupby("date")["ret"].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    values = equity.to_numpy()
    dates = equity.index.to_numpy()
    out = {}
    for i, date in enumerate(dates):
        start = max(0, i - 19)
        out[date] = values[i] / values[start] - 1.0 if i > 0 else 0.0
    return out


def signal(values, dates, market, enter_q, exit_q, min_hold, regime_gate):
    """滞后带反转 + regime 门控。

    regime_gate > 0：市场动量高于阈值时强制持有（不看个股信号）
    regime_gate < 0：市场动量低于阈值时才允许择时，其余时间强制持有
    regime_gate == 0：不启用门控
    """
    n = len(values)
    ma20 = values[:, IDX["ma20_relative"]]
    target = np.zeros(n)
    holding = False
    held_days = 0
    for k in range(n):
        mom = market.get(dates[k], 0.0)
        if regime_gate > 0 and mom > regime_gate:
            holding = True
            held_days = 0
        elif regime_gate < 0 and mom <= regime_gate:
            holding = True
            held_days = 0
        else:
            if not holding:
                if ma20[k] < enter_q:
                    holding = True
                    held_days = 0
            else:
                held_days += 1
                if held_days >= min_hold and ma20[k] > exit_q:
                    holding = False
                    held_days = 0
        target[k] = 1.0 if holding else 0.0
    return target


def run(env, target):
    env.reset()
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
        _, _, terminated, _, info = env.step(action)
        held = info["shares"] > 0
        days_in += 1 if held else 0
        done = bool(terminated)
        step += 1
    metrics = env.metrics()
    return metrics, days_in / max(step, 1)


def main() -> int:
    cfg = training_cfg()
    print("加载面板 ...")
    panel = load_panel()
    market = build_market(panel)

    data = build_dataset(FEATURE_DIR, 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")

    # train 段分位阈值
    train_rows = []
    for symbol, entry in panel.items():
        mask = np.array([d[:4].isdigit() and 2006 <= int(d[:4]) <= 2021
                         for d in entry["dates"]])
        train_rows.append(entry["values"][mask])
    train_stack = np.concatenate(train_rows, axis=0)
    ma20_train = train_stack[:, IDX["ma20_relative"]]
    quantiles = {q: float(np.quantile(ma20_train, q))
                 for q in (0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.85, 0.9, 0.95)}

    # 参数网格（收敛到 24 组：全网格 160 组 × 2636 episode 的回放量级太大）
    grid = []
    for enter_q in (0.1, 0.2):
        for exit_q in (0.7, 0.85):
            for min_hold in (20, 40):
                for gate in (0.03, 0.0):
                    grid.append((enter_q, exit_q, min_hold, gate))
    print(f"参数网格 {len(grid)} 组")
    # train 上用子样本粗筛（每 3 个取 1），候选再进全量 val/test 精评
    train_stride = 3
    screen = {}

    def episodes_of(split, stride=1):
        pool = data.strict_pools[split]
        return pool[::stride]

    for split, stride in (("train", train_stride), ("val", 1), ("test", 1)):
        pool = episodes_of(split, stride)
        for i, (uid, obs, bars) in enumerate(pool):
            symbol = uid.split("_")[0]
            entry = panel.get(symbol)
            if entry is None:
                continue
            pos = {d: j for j, d in enumerate(entry["dates"])}
            start = pos.get(str(bars.dates[0]))
            if start is None:
                continue
            n = len(bars.dates)
            if start + n > len(entry["dates"]):
                continue
            values = entry["values"][start:start + n]
            dates = entry["dates"][start:start + n]
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            for enter_q, exit_q, min_hold, gate in grid:
                target = signal(values, dates, market, quantiles[enter_q],
                                quantiles[exit_q], min_hold, gate)
                metrics, exposure = run(env, target)
                key = (enter_q, exit_q, min_hold, gate)
                bucket = screen.setdefault(key, {"train": [], "val": [], "test": []})
                bucket[split].append(
                    (metrics["excess_return"], metrics["total_turnover_ratio"], exposure))
            env.close()
            if (i + 1) % 200 == 0:
                print(f"  {split}: {i + 1}/{len(pool)}", flush=True)
        print(f"  {split} 完成（{len(pool)} episode）", flush=True)

    def agg(g, split):
        rows = screen[g][split]
        if not rows:
            return None
        return (float(np.mean([r[0] for r in rows])),
                float(np.mean([r[1] for r in rows])),
                float(np.mean([r[2] for r in rows])))

    print("\n===== 按 train 超额排序的前 15 组 =====")
    print(f"    {'enter':>6s}{'exit':>6s}{'持有':>6s}{'gate':>7s}"
          f"{'train':>10s}{'val':>10s}{'test':>10s}{'test换手':>10s}{'在市':>8s}")
    ranked = []
    for g in grid:
        t = agg(g, "train")
        if t is None:
            continue
        ranked.append((t[0], g, t))
    ranked.sort(reverse=True)
    for train_ex, g, t in ranked[:15]:
        v = agg(g, "val")
        s = agg(g, "test")
        print(f"    {g[0]:>6.2f}{g[1]:>6.2f}{g[2]:>6d}{g[3]:>+7.2f}"
              f"{train_ex:>+10.2%}{v[0]:>+10.2%}{s[0]:>+10.2%}{s[1]:>10.2f}{s[2]:>8.1%}")

    # 只用 train 选出的最优
    best = ranked[0]
    g = best[1]
    v, s = agg(g, "val"), agg(g, "test")
    print(f"\n  ★ train 选出的最优: enter_q={g[0]} exit_q={g[1]} min_hold={g[2]} gate={g[3]:+}")
    print(f"    train {best[0]:+.2%}   val {v[0]:+.2%}   test {s[0]:+.2%}"
          f"（换手 {s[1]:.1f}，在市 {s[2]:.1%}）")
    print(f"    对照 PPO(v3): val −1.11%  test −5.04%")

    # train 前 10 组在 val/test 上的表现分布（评估选参稳健性）
    top = [agg(g, "val")[0] for _, g, _ in ranked[:10]]
    tops = [agg(g, "test")[0] for _, g, _ in ranked[:10]]
    print(f"\n  train 前 10 组在 val 上: 均值 {np.mean(top):+.2%}  "
          f"为正 {sum(1 for x in top if x > 0)}/10")
    print(f"  train 前 10 组在 test 上: 均值 {np.mean(tops):+.2%}  "
          f"为正 {sum(1 for x in tops if x > 0)}/10")
    print(f"  （若 test 上大面积为负，说明单一静态规则不足以穿越 regime）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
