"""regime 门控规则的正超额来自「真技能」还是「低暴露在牛市不吃亏」？

唯一正面结果（§validate_combined_strategy）：
  门控规则（市场 20 日动量 > 3% 时强制持有）+ 滞后带反转
  → train 选参后 val +0.22% / test +0.17%（对照 PPO v3: −1.11% / −5.04%）
  → 在市占比 92.8%，年换手 2.2

在 test 段（基准 +27.66% 的牛市）里，低暴露本身就是优势。
所以必须用**技能项**（持仓日相对基准当日的超出，与暴露度无关）来判定真伪。

本脚本用统一台架算三组：
  ① 门控规则            ② 恒定满仓（基准的代理）  ③ PPO v3 模型
并给出「若门控规则的技能项 > 0 且高于恒定满仓 → 有真技能」的判据。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "outputs"))

from stock_ai.config.schema import EnvConfig  # noqa: E402
from stock_ai.scale.unified import (  # noqa: E402
    DEFAULT_RANGES, EXECUTION_CORPORATE_ACTIONS, FEATURES, build_dataset, make_strict_env,
)
from timing_backtest import TimingBacktest  # noqa: E402

RUN = ROOT / "models" / "unified" / "20261020_105633_578f25aa"
FEATURE_DIR = ROOT / "data" / "raw_hfq_v2_train"
SPLITS = {"train": (2006, 2021), "val": (2022, 2023), "test": (2024, 2025)}
ENTER_Q, EXIT_Q, MIN_HOLD, GATE = 0.2, 0.7, 40, 0.03


def build_panel():
    from stock_ai.scale.unified import relative_features
    frames = []
    for path in sorted(FEATURE_DIR.glob("*.parquet")):
        raw = pd.read_parquet(path)
        if raw.empty:
            continue
        df, values = relative_features(raw)
        out = df[["date", "close"]].copy()
        out["symbol"] = path.stem
        for i, name in enumerate(FEATURES):
            out[name] = values[:, i]
        frames.append(out)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["ret"] = panel.groupby("symbol")["close"].pct_change()
    panel["fwd"] = (panel.groupby("symbol")["close"].shift(-20)
                    / panel["close"] - 1.0)
    panel["year"] = panel["date"].dt.year
    return panel


def market_mom_map(panel):
    """等权市场 20 日动量（只用当日及以前）。"""
    rows = []
    for symbol, group in panel.groupby("symbol", sort=False):
        rows.append(pd.DataFrame({"date": group["date"].to_numpy(),
                                   "ret": group["ret"].to_numpy()}))
    frame = pd.concat(rows, ignore_index=True)
    daily = frame.groupby("date")["ret"].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    vals = equity.to_numpy()
    dts = equity.index.to_numpy()
    out = {}
    for i, d in enumerate(dts):
        s = max(0, i - 19)
        out[d] = vals[i] / vals[s] - 1.0 if i > 0 else 0.0
    return out


def skill_decomposition(log_p, log_b, held):
    """对数收益口径的技能项分解（与 diagnose_exposure_vs_skill.py 同口径）。

    输入都是**每步的对数收益**序列（不是资金曲线）：
      log_p: 组合的逐日对数收益
      log_b: 基准的逐日对数收益
    暴露度项 = −Σ(1−held)·log_b   空仓放弃的基准收益
    技能项   = Σ log_p − Σ held·log_b   持仓日相对基准当日的超出（含摩擦）
    """
    log_b = np.asarray(log_b, dtype="float64")
    log_p = np.asarray(log_p, dtype="float64")
    held = np.asarray(held, dtype="float64")
    exposure_term = float(-((1.0 - held) * log_b).sum())
    skill_term = float(log_p.sum() - (held * log_b).sum())
    return exposure_term, skill_term, float(log_p.sum() - log_b.sum())


def main() -> int:
    print("加载面板 ...")
    panel = build_panel()
    mkt_mom = market_mom_map(panel)
    panel["mkt_mom20"] = panel["date"].map(mkt_mom).fillna(0.0)

    # train 段分位阈值
    train_rows = panel[(panel.year >= 2006) & (panel.year <= 2021)]
    idx = {name: i for i, name in enumerate(FEATURES)}
    ma20_all = []
    for symbol, group in panel.groupby("symbol", sort=False):
        ma20_all.append(group["ma20_relative"].to_numpy())
    ma20_all = np.concatenate(ma20_all)
    mask = np.concatenate([((panel[panel.symbol == s].year >= 2006) &
                             (panel[panel.symbol == s].year <= 2021)).to_numpy()
                           for s in panel.symbol.unique()])
    q_enter = float(np.quantile(ma20_all[mask], ENTER_Q))
    q_exit = float(np.quantile(ma20_all[mask], EXIT_Q))
    print(f"  阈值: enter_q={ENTER_Q} → {q_enter:+.4f}  "
          f"exit_q={EXIT_Q} → {q_exit:+.4f}  gate={GATE:+.0%}  min_hold={MIN_HOLD}")
    print()

    data = panel[panel["fwd"].notna()].copy()
    data = data[data["fwd"].abs() < 1.0].reset_index(drop=True)
    data["_row"] = np.arange(len(data))

    def gated_signal(group: pd.DataFrame) -> np.ndarray:
        ma20 = group["ma20_relative"].to_numpy()
        mom = group["mkt_mom20"].to_numpy()
        n = len(group)
        target = np.zeros(n)
        holding = False
        held_days = 0
        for k in range(n):
            if mom[k] > GATE:
                holding = True
                held_days = 0
            else:
                if not holding:
                    if ma20[k] < q_enter:
                        holding = True
                        held_days = 0
                else:
                    held_days += 1
                    if held_days >= MIN_HOLD and ma20[k] > q_exit:
                        holding = False
                        held_days = 0
            target[k] = 1.0 if holding else 0.0
        return target

    print("===== 统一台架对比（调仓间隔 20）=====")
    print(f"    {'策略':16s}{'切分':6s}{'净超额':>10s}{'毛超额':>10s}"
          f"{'年换手':>9s}{'在市':>8s}")
    for split in ("val", "test"):
        sub = data[(data.year >= SPLITS[split][0]) & (data.year <= SPLITS[split][1])].copy()
        bt = TimingBacktest(sub, rebalance=20, global_len=len(data))
        r = bt.run(gated_signal)
        print(f"    {'门控规则':16s}{split:6s}{r.net_excess:>+10.2%}"
              f"{r.gross_excess:>+10.2%}{r.switches:>9.1f}{r.exposure:>8.1%}")
        r2 = bt.run(lambda g: np.ones(len(g)))
        print(f"    {'恒定满仓':16s}{split:6s}{r2.net_excess:>+10.2%}"
              f"{r2.gross_excess:>+10.2%}{r2.switches:>9.1f}{r2.exposure:>8.1%}")
    print()

    # ---- 用真实 CA 环境算技能项（与 PPO 同口径）----
    print("===== 真实 CA 环境下的技能项分解（与 PPO v3 同口径）=====")
    cfg_params = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["reward_params"]
    cfg = EnvConfig()
    cfg.reward_type = cfg_params["reward_type"]
    cfg.lambda_drawdown = float(cfg_params["lambda_drawdown"])
    cfg.lambda_turnover = float(cfg_params["lambda_turnover"])
    cfg.lambda_leverage = float(cfg_params["lambda_leverage"])
    data_env = build_dataset(FEATURE_DIR, 20, DEFAULT_RANGES, None,
                              execution_dir=ROOT / "data" / "raw",
                              execution_version=EXECUTION_CORPORATE_ACTIONS,
                              corporate_actions_dir=ROOT / "data" / "corporate_actions")
    from stable_baselines3 import PPO
    model = PPO.load(str(RUN / "model.zip"))

    panel_dates = {s: dict(zip(panel[panel.symbol == s]["date"].dt.strftime("%Y%m%d"),
                               panel[panel.symbol == s]["mkt_mom20"]))
                   for s in panel.symbol.unique()}

    for split in ("val", "test"):
        agg = {"gated": [], "ppo": []}
        for uid, obs, bars in data_env.strict_pools[split]:
            symbol = uid.split("_")[0]
            dates = [str(d) for d in bars.dates]
            mom_map = panel_dates.get(symbol, {})
            mom = np.array([mom_map.get(d, 0.0) for d in dates])
            entry = panel[panel.symbol == symbol]
            pos_map = dict(zip(entry["date"].dt.strftime("%Y%m%d"),
                               entry["ma20_relative"]))
            ma20 = np.array([pos_map.get(d, 0.0) for d in dates])
            # 门控目标
            n = len(dates)
            target = np.zeros(n)
            holding = False
            held_days = 0
            for k in range(n):
                if mom[k] > GATE:
                    holding = True
                    held_days = 0
                else:
                    if not holding:
                        if ma20[k] < q_enter:
                            holding = True
                            held_days = 0
                    else:
                        held_days += 1
                        if held_days >= MIN_HOLD and ma20[k] > q_exit:
                            holding = False
                            held_days = 0
                target[k] = 1.0 if holding else 0.0

            for name, tgt, mdl in (("gated", target, None), ("ppo", None, model)):
                env = make_strict_env(obs, bars, cfg,
                                      execution_version=EXECUTION_CORPORATE_ACTIONS)
                o, _ = env.reset()
                done = False
                step = 0
                held = False
                r_b, held_flags = [], []
                prev_value = float(env.equity[0])
                log_p = []
                while not done:
                    if tgt is None:
                        action, _ = model.predict(o, deterministic=True)
                        action = int(np.asarray(action).reshape(-1)[0])
                    else:
                        want = tgt[min(step, n - 1)]
                        if want > 0.5 and not held:
                            action = 1
                        elif want <= 0.5 and held:
                            action = 2
                        else:
                            action = 0
                    o, _, terminated, _, info = env.step(action)
                    held = info["shares"] > 0
                    value = float(info["portfolio_value"])
                    log_p.append(np.log1p(value / prev_value - 1.0)
                                 if prev_value > 1e-9 else 0.0)
                    prev_value = value
                    r_b.append(float(info["benchmark_return"]))
                    held_flags.append(1.0 if held else 0.0)
                    done = bool(terminated)
                    step += 1
                env.close()
                log_b = np.log1p(np.array(r_b))
                exposure_term, skill_term, total = skill_decomposition(
                    np.array(log_p), log_b, np.array(held_flags))
                agg[name].append({
                    "exposure_term": exposure_term,
                    "skill_term": skill_term,
                    "total": total,
                    "exposure": float(np.mean(held_flags)),
                })
        for name in ("gated", "ppo"):
            rows = agg[name]
            if not rows:
                continue
            ex = float(np.mean([r["exposure_term"] for r in rows]))
            sk = float(np.mean([r["skill_term"] for r in rows]))
            tt = float(np.mean([r["total"] for r in rows]))
            ep = float(np.mean([r["exposure"] for r in rows]))
            print(f"  {split:5s} {name:6s}: 合计 {tt:+.2%} = 暴露度项 {ex:+.2%} "
                  f"+ 技能项 {sk:+.2%}   在市 {ep:.1%}")
        print()
    print("  判据：门控规则的**技能项**若显著 > 0，说明它真的选对了时点；")
    print("        若技能项 ≈ 0 或为负，说明正超额只是「低暴露在牛市不吃亏」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
