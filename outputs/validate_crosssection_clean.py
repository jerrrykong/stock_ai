"""方案 C 的干净度量：不重叠 20 日前瞻收益对比（避免复利与幸存者偏差污染）。

前面的组合回测有两个口径问题：
  ① 基准 +5187%：股票池是幸存者样本（在册 2006 年 60 只 → 2025 年 197 只，
     且都是活下来的赢家），且每日再平衡的等权会系统性高估。
  ② 超额用复利差表示，被两个极端数字淹没。

本脚本改用**最干净的度量**：在每个调仓日 T，
  ① 取信号最好的 top-K 股票，算它们的未来 20 日收益均值 → 组合毛收益
  ② 取当日全部可交易股票的 20 日收益均值 → 市场基准
  ③ 差值 = 该次调仓的超额
不重叠地遍历所有调仓日，累加这些超额，并扣摩擦。

这直接回答：「按信号选股，每 20 日换一次，平均每次能比随机多赚多少？」
不含复利、不含幸存者漂移（每日截面内的股票都是当时真实存在的）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_ai.scale.unified import FEATURES  # noqa: E402

FEATURE_DIR = ROOT / "data" / "raw_hfq_v2_train"
SPLITS = {"train": (2006, 2021), "val": (2022, 2023), "test": (2024, 2025)}
HORIZON = 20
ROUND_TRIP_COST = 0.003
SIGNAL = "ma20_relative"


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
    panel["fwd"] = (panel.groupby("symbol")["close"].shift(-HORIZON)
                    / panel["close"] - 1.0)
    panel["year"] = panel["date"].dt.year
    return panel


def main() -> int:
    print("加载面板 ...")
    panel = build_panel()
    data = panel[panel["fwd"].notna()].copy()
    data = data[data["fwd"].abs() < 1.0]
    print(f"  有效样本 {len(data):,}，{data['symbol'].nunique()} 只")
    print()

    print("===== 不重叠 20 日前瞻：信号选股 vs 当日全市场平均 =====")
    print("    每次调仓的超额 = top-K 的 20 日收益均值 − 当日全部可交易股票的均值")
    print()
    print(f"    {'切分':6s}{'K':>5s}{'调仓次数':>9s}{'每次超额':>11s}"
          f"{'累计超额':>11s}{'年化':>10s}{'胜率':>8s}{'摩擦后/次':>11s}")
    for split in ("train", "val", "test"):
        a, b = SPLITS[split]
        sub = data[(data.year >= a) & (data.year <= b)]
        for top_k in (5, 10, 20, 40):
            excesses = []
            dates = sub["date"].unique()
            for t in range(0, len(dates) - HORIZON, HORIZON):
                day = pd.Timestamp(dates[t])
                today = sub[sub.date == day]
                future = sub[sub.date == pd.Timestamp(dates[t + HORIZON])]
                if len(today) < top_k or len(future) < top_k:
                    continue
                # 信号最负（最超涨）的前 K 只
                picked = today.nsmallest(top_k, SIGNAL)
                picked_fwd = future.set_index("symbol")["fwd"].reindex(picked["symbol"])
                picked_fwd = picked_fwd.dropna()
                if len(picked_fwd) < top_k:
                    continue
                market_fwd = future["fwd"].mean()
                excesses.append(float(picked_fwd.mean() - market_fwd))
            if not excesses:
                continue
            arr = np.array(excesses)
            n_per_year = 252 / HORIZON
            friction = top_k / len(today) * 2 * ROUND_TRIP_COST  # 换掉的比例×往返
            annual = arr.mean() * n_per_year
            print(f"    {split:6s}{top_k:>5d}{len(arr):>9d}{arr.mean():>+11.3%}"
                  f"{arr.sum():>+11.2%}{annual:>+10.2%}"
                  f"{float((arr > 0).mean()):>8.1%}{arr.mean() - friction:>+11.3%}")
        print()

    print("===== 对照：随机选 K 只的期望超额（应为 0）=====")
    rng = np.random.default_rng(42)
    for split in ("val", "test"):
        a, b = SPLITS[split]
        sub = data[(data.year >= a) & (data.year <= b)]
        random_excess = []
        dates = sub["date"].unique()
        for t in range(0, len(dates) - HORIZON, HORIZON):
            day = pd.Timestamp(dates[t])
            today = sub[sub.date == day]
            future = sub[sub.date == pd.Timestamp(dates[t + HORIZON])]
            if len(today) < 20 or len(future) < 20:
                continue
            picked = today.sample(20, random_state=int(t))["symbol"]
            picked_fwd = future.set_index("symbol")["fwd"].reindex(picked).dropna()
            if len(picked_fwd) < 20:
                continue
            random_excess.append(float(picked_fwd.mean() - future["fwd"].mean()))
        if random_excess:
            arr = np.array(random_excess)
            print(f"  {split:6s} 随机 K=20: 每次超额 {arr.mean():+.3%}  "
                  f"（应≈0，作为对照基准）")
    print()
    print("  读法：信号组每次超额若显著 > 随机对照组，且 > 摩擦（0.3%×换手比例），")
    print("  则横截面选股有可执行价值 → 方案 C 可行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
