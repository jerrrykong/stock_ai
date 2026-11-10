"""亏损分解（对数收益口径，正确版）。

把策略相对恒买基准的超额严格拆成两项：

    L_p − L_b = [暴露度项] + [择时技能项]
    暴露度项 = −Σ (1−held_t)·log(1+r_b,t)      ← 空仓放弃掉的基准收益
    择时技能项 = L_p − Σ held_t·log(1+r_b,t)    ← 持仓日相对基准当日的超出

对数收益可加，复合收益不可加；上一版用 f×复合收益是量纲错误。
摩擦被计入「持仓日超出」里（持仓日的实际收益含成本），这是有意的：
「择时技能」= 持仓日真正赚到的超额，摩擦是它的一部分。

对照朴素等式：无技能时 excess ≈ −(1−f)·r_bench，用于校验分解是否自洽。
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
                            "dates": df.date.dt.strftime("%Y%m%d").to_numpy()}
    return panel


def rollout_capture(env, target=None, model=None):
    obs, info = env.reset()
    initial = float(info["portfolio_value"])
    prev_value = initial
    done = False
    step = 0
    held = False
    rows = []
    while not done:
        if target is None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
        else:
            want = target[min(step, len(target) - 1)]
            if want > 0.5 and not held:
                action = 1
            elif want <= 0.5 and held:
                action = 2
            else:
                action = 0
        obs, _, terminated, _, info = env.step(action)
        held = info["shares"] > 0
        value = float(info["portfolio_value"])
        r_p = value / prev_value - 1.0 if prev_value > 1e-9 else 0.0
        r_b = float(info["benchmark_return"])
        rows.append((r_p, r_b, held))
        prev_value = value
        done = bool(terminated)
        step += 1
    metrics = env.metrics()
    return metrics, rows


def decompose(metrics, rows):
    r_p = np.array([r[0] for r in rows])
    r_b = np.array([r[1] for r in rows])
    held = np.array([r[2] for r in rows], dtype=float)
    log_b = np.log1p(r_b)
    log_p = np.log1p(r_p)
    L_p = float(log_p.sum())
    L_b = float(log_b.sum())
    exposure_term = float(-((1.0 - held) * log_b).sum())
    skill_term = float(L_p - (held * log_b).sum())
    f = float(held.mean())
    r_bench = float(np.expm1(L_b))
    r_strat = float(np.expm1(L_p))
    # 自洽校验：无技能假设下 exposure_term ≈ −(1−f)·log(1+r_bench)
    naive = float(-(1.0 - f) * np.log1p(r_bench))
    return {
        "excess": metrics["excess_return"],
        "f": f,
        "r_bench": r_bench,
        "r_strat": r_strat,
        "exposure_term": exposure_term,
        "skill_term": skill_term,
        "naive_check": naive,
        "total": L_p - L_b,
    }


def main() -> int:
    cfg = training_cfg()
    data = build_dataset(FEATURE_DIR, 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")
    from stable_baselines3 import PPO
    model = PPO.load(str(RUN / "model.zip"))
    panel = load_panel()

    train_rows = []
    for symbol, entry in panel.items():
        raw = pd.read_parquet(FEATURE_DIR / f"{symbol}.parquet", columns=["date"])
        mask = ((raw.date.dt.year >= 2006) & (raw.date.dt.year <= 2021)).to_numpy()
        train_rows.append(entry["values"][mask])
    train_stack = np.concatenate(train_rows, axis=0)
    q20 = float(np.quantile(train_stack[:, IDX["ma20_relative"]], 0.2))
    q80 = float(np.quantile(train_stack[:, IDX["ma20_relative"]], 0.8))

    buckets = {}
    for split in ("val", "test"):
        for uid, obs, bars in data.strict_pools[split]:
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
            ma20 = values[:, IDX["ma20_relative"]]
            target = np.zeros(n)
            holding = False
            held_days = 0
            for k in range(n):
                if not holding:
                    if ma20[k] < q20:
                        holding = True
                        held_days = 0
                else:
                    held_days += 1
                    if held_days >= 20 and ma20[k] > q80:
                        holding = False
                        held_days = 0
                target[k] = 1.0 if holding else 0.0

            for name, tgt, mdl in (("PPO(v3)", None, model),
                                   ("ma20_band", target, None),
                                   ("buyhold", np.ones(n), None)):
                env = make_strict_env(obs, bars, cfg,
                                      execution_version=EXECUTION_CORPORATE_ACTIONS)
                metrics, rows = rollout_capture(env, tgt, mdl)
                env.close()
                buckets.setdefault((name, split), []).append(decompose(metrics, rows))
        print(f"  {split} 完成", flush=True)

    print("\n===== 亏损分解（对数收益口径）=====")
    for name in ("PPO(v3)", "ma20_band", "buyhold"):
        for split in ("val", "test"):
            rows = buckets.get((name, split), [])
            if not rows:
                continue
            agg = {k: float(np.mean([r[k] for r in rows]))
                   for k in ("excess", "f", "r_bench", "r_strat",
                             "exposure_term", "skill_term", "naive_check", "total")}
            print(f"\n  {name} / {split}（{len(rows)} episode）")
            print(f"    净超额（官方）           : {agg['excess']:+7.2%}")
            print(f"    分解合计（对数口径）     : {agg['total']:+7.2%}"
                  f"   ← 应与净超额同向同量级")
            print(f"    ├ 暴露度项（空仓放弃）   : {agg['exposure_term']:+7.2%}")
            print(f"    └ 择时技能项（含摩擦）   : {agg['skill_term']:+7.2%}"
                  f"   ← 真正的本事")
            print(f"      自洽校验 −(1−f)·ln(1+r_b): {agg['naive_check']:+7.2%}")
            print(f"      在市 {agg['f']:.1%}  策略 {agg['r_strat']:+.2%}  "
                  f"基准 {agg['r_bench']:+.2%}")
    print()
    print("  暴露度项 = 空仓期间放弃的基准收益（仓位选择的必然结果）")
    print("  择时技能项 = 持仓日相对基准当日的超出（含摩擦），负值说明选时反而更差")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
