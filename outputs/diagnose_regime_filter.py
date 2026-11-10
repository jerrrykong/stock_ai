"""市场状态（regime）过滤能否修复择时信号。

背景：naive 反转规则在 val 段 +2.35%，在 test 段却是 −26.5%。
诊断假设是「test 是牛市（基准 +27.66%），反转信号在强势上涨中反复被甩下车」。

检验：
  1. 用 197 只等权构造市场代理，算市场 20 日动量
  2. 分市场状态看时序 IC（牛/熊/震荡下信号是否一致）
  3. 加 regime 过滤后的规则表现：市场动量为正时强制持有，否则才允许择时
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
        panel[path.stem] = {"values": values,
                            "dates": df.date.dt.strftime("%Y%m%d").to_numpy(),
                            "close": df.close.to_numpy(dtype="float64")}
    return panel


def build_market_proxy(panel) -> pd.DataFrame:
    """197 只等权市场代理：按日期取横截面平均收益，累成净值。"""
    rows = []
    for symbol, entry in panel.items():
        close = entry["close"]
        ret = np.zeros(len(close))
        ret[1:] = close[1:] / close[:-1] - 1.0
        rows.append(pd.DataFrame({"date": entry["dates"], "ret": ret}))
    frame = pd.concat(rows, ignore_index=True)
    frame = frame[frame["date"] != "nan"]
    daily = frame.groupby("date")["ret"].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    momentum = {}
    values = equity.to_numpy()
    dates = equity.index.to_numpy()
    for i, date in enumerate(dates):
        start = max(0, i - 19)
        momentum[date] = values[i] / values[start] - 1.0 if i > 0 else 0.0
    out = pd.DataFrame({"date": dates, "mkt_mom20": [momentum[d] for d in dates]})
    out["date"] = out["date"].astype(str)
    return out.set_index("date")["mkt_mom20"].to_dict(), daily


def regime_of(momentum: float, bull=0.05, bear=-0.05) -> str:
    if momentum >= bull:
        return "牛"
    if momentum <= bear:
        return "熊"
    return "震荡"


def main() -> int:
    cfg = training_cfg()
    print("加载特征面板 ...")
    panel = load_panel()
    market, daily = build_market_proxy(panel)

    # ---------- 1. 市场状态分段 ----------
    print("\n===== 1. 市场代理（197 只等权）的年度表现 =====")
    for year in range(2006, 2026):
        slice_ = daily[daily.index.str[:4] == str(year)]
        if slice_.empty:
            continue
        ret = float((1 + slice_).prod() - 1)
        print(f"    {year}: {ret:+7.2%}  {'牛' if ret >= 0.05 else ('熊' if ret <= -0.05 else '震荡')}")
    print()

    # ---------- 2. 分市场状态的时序 IC ----------
    print("===== 2. ma20_relative 的 20 日 IC，按市场状态拆分 =====")
    idx = {name: i for i, name in enumerate(FEATURES)}
    buckets = {r: [] for r in ("牛", "熊", "震荡")}
    for symbol, entry in panel.items():
        values = entry["values"]
        close = entry["close"]
        dates = entry["dates"]
        n = len(close)
        fwd = np.full(n, np.nan)
        for k in range(n):
            j = k + 20
            if j < n:
                fwd[k] = close[j] / close[k] - 1.0
        for k in range(1, n):
            if not np.isfinite(fwd[k]):
                continue
            mom = market.get(dates[k], 0.0)
            regime = regime_of(mom)
            x = values[k, idx["ma20_relative"]]
            y = fwd[k]
            buckets[regime].append((x, y, dates[k]))
    print(f"    {'状态':<8s}{'样本数':>10s}{'IC':>10s}{'IC 标准误':>12s}{'t 值':>8s}")
    for regime, rows in buckets.items():
        if len(rows) < 200:
            print(f"    {regime:<8s}{len(rows):>10d}{'—':>10s}{'—':>12s}{'—':>8s}")
            continue
        x = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        xr = np.argsort(np.argsort(x))
        yr = np.argsort(np.argsort(y))
        ic = float(np.corrcoef(xr, yr)[0, 1])
        se = 1.0 / np.sqrt(len(rows))
        print(f"    {regime:<8s}{len(rows):>10d}{ic:>+10.4f}{se:>12.4f}{ic / se:>8.1f}")
    print()

    # ---------- 3. 加 regime 过滤的规则 ----------
    print("===== 3. regime 过滤规则：市场动量 > 阈值时强制持有 =====")
    data = build_dataset(FEATURE_DIR, 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")

    train_rows = []
    for symbol, entry in panel.items():
        mask = np.array([d[:4].isdigit() and 2006 <= int(d[:4]) <= 2021
                         for d in entry["dates"]])
        train_rows.append(entry["values"][mask])
    train_stack = np.concatenate(train_rows, axis=0)
    q20 = float(np.quantile(train_stack[:, idx["ma20_relative"]], 0.2))
    q80 = float(np.quantile(train_stack[:, idx["ma20_relative"]], 0.8))
    print(f"    train 段 ma20_relative 分位: q20={q20:+.4f} q80={q80:+.4f}")

    for mom_threshold in (0.0, 0.03, 0.05):
        results = {}
        for split in ("val", "test"):
            for uid, obs, bars in data.strict_pools[split]:
                symbol = uid.split("_")[0]
                entry = panel.get(symbol)
                if entry is None:
                    continue
                pos = {d: i for i, d in enumerate(entry["dates"])}
                start = pos.get(str(bars.dates[0]))
                if start is None:
                    continue
                n = len(bars.dates)
                if start + n > len(entry["dates"]):
                    continue
                values = entry["values"][start:start + n]
                dates = entry["dates"][start:start + n]
                ma20 = values[:, idx["ma20_relative"]]
                # 目标持仓：市场动量达标 → 强制持有；否则按反转带
                target = np.zeros(n)
                holding = False
                held_days = 0
                for k in range(n):
                    mom = market.get(dates[k], 0.0)
                    if mom > mom_threshold:
                        holding = True
                        held_days = 0
                    else:
                        if not holding:
                            if ma20[k] < q20:
                                holding = True
                                held_days = 0
                        else:
                            held_days += 1
                            if held_days >= 10 and ma20[k] > q80:
                                holding = False
                                held_days = 0
                    target[k] = 1.0 if holding else 0.0
                env = make_strict_env(obs, bars, cfg,
                                      execution_version=EXECUTION_CORPORATE_ACTIONS)
                env.reset()
                done = False
                step = 0
                held = False
                days_in = 0
                while not done:
                    want = target[min(step, n - 1)]
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
                env.close()
                results.setdefault(split, []).append(
                    (metrics["excess_return"], metrics["total_turnover_ratio"],
                     days_in / max(step, 1)))
        val = results.get("val", [])
        test = results.get("test", [])
        if not val or not test:
            continue
        print(f"    市场动量阈值 > {mom_threshold:+.0%}:")
        print(f"      val  超额 {np.mean([r[0] for r in val]):+.2%}  "
              f"换手 {np.mean([r[1] for r in val]):.1f}  "
              f"在市 {np.mean([r[2] for r in val]):.1%}")
        print(f"      test 超额 {np.mean([r[0] for r in test]):+.2%}  "
              f"换手 {np.mean([r[1] for r in test]):.1f}  "
              f"在市 {np.mean([r[2] for r in test]):.1%}")
    print()
    print("  对照: PPO(v3) val −1.11% / test −5.04%；naive ma20_rev_q0.2 val +2.35% / test −26.50%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
