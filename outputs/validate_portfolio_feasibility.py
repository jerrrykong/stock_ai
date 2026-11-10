"""方案 C 可行性验证：横截面/组合架构下，残差信号能否变成正超额？

诊断已确立（outputs/diagnose_ic_semantics.py）：
  - 稳定的是**个股相对市场残差**的预测力（ma20_relative 的残差 IC
    train −0.0241 / val −0.0347 / test −0.0294）
  - 单股择时赚绝对收益 → 残差信号用不上
  - 残差信号正是横截面架构需要的

本脚本用**等权组合**做忠实模拟（不做 RL，只验证「信号有没有可执行价值」）：
  每个调仓日：
    1. 对全部 197 只按信号排序
    2. 持有信号最好的 top-K，其余空仓（或做空对冲，但 A 股个股难做空，
       故只测多头版本）
    3. 下一交易日按 open 成交（与项目 T+1 开盘成交语义一致）
    4. 扣摩擦：每次调仓换掉的比例 × 往返成本
  基准：等权全持有组合

K 的选择很关键：K 越大暴露越满、越接近基准；K 越小越集中、越依赖信号。
同时测「信号加权」版本（按信号强度加权持仓）。

注意：这里用 hfq 收盘算信号与收益（研究口径），不含涨跌停/停牌/分红。
真实环境会有额外约束，但方向性结论（信号能否产生超额）不受影响。
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
ROUND_TRIP_COST = 0.003
REBALANCE_DAYS = 20
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
    panel["ret"] = panel.groupby("symbol")["close"].pct_change()
    panel["year"] = panel["date"].dt.year
    return panel


def rank_ic(x, y):
    if len(x) < 30:
        return np.nan
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def portfolio_backtest(panel, top_k, rebalance=REBALANCE_DAYS, cost=ROUND_TRIP_COST,
                       weight_by_signal=False):
    """等权组合：每 rebalance 日按信号排序持有 top_k，下一日生效。

    权重在两次调仓之间**持续持有**（不因未调仓而清零）——否则组合只在调仓日
    有仓位，其余日空仓，会得到虚假的巨额亏损。
    """
    wide = panel.pivot(index="date", columns="symbol", values="ret")
    signal_wide = panel.pivot(index="date", columns="symbol", values=SIGNAL)
    dates = wide.index.to_numpy()
    symbols = list(wide.columns)
    n = len(dates)

    # 前值收益：信号在 T 日出，持有人在 T+1 吃到收益
    prev_ret = wide.shift(1)

    # 基准口径修正：等权基准必须**只算当日实际有行情的股票**，
    # 且不能把「当日无行情」当作 0 收益（那是停牌，不是跌停）。
    # 另一个关键：股票池是幸存者样本（在册数 2006 年 60 只 → 2025 年 197 只），
    # 早期只有「后来活下来的赢家」，因此历史等权基准天然虚高，
    # 这一点在解读绝对收益时必须记住（超额才是可比量）。
    alive = prev_ret.notna()
    n_alive = alive.sum(axis=1)
    bench = np.where(n_alive > 0,
                     (prev_ret.fillna(0) * alive).sum(axis=1) / np.maximum(n_alive, 1),
                     np.nan)

    port = np.zeros(n)
    weights = np.zeros((n, len(symbols)))
    current = np.zeros(len(symbols))
    for t in range(1, n):
        if (t - 1) % rebalance == 0:
            sig = signal_wide.iloc[t - 1]
            ok = sig.notna() & prev_ret.iloc[t].notna()
            if ok.sum() >= top_k:
                ranked = sig[ok].sort_values()
                picked = ranked.index[:top_k]
                w = np.zeros(len(symbols))
                if weight_by_signal:
                    vals = sig[picked].to_numpy()
                    z = (vals - vals.mean()) / (vals.std() + 1e-9)
                    raw_w = 1.0 / (1.0 + np.exp(z))
                    raw_w = raw_w / raw_w.sum()
                else:
                    raw_w = np.full(len(picked), 1.0 / len(picked))
                for s, weight in zip(picked, raw_w):
                    w[symbols.index(s)] = weight
                current = w
            # else: 维持原权重（不调仓不等于清仓）
        weights[t] = current
        port[t] = float((current * prev_ret.iloc[t].fillna(0).to_numpy()).sum())

    valid = ~np.isnan(bench)
    bench_c = np.cumprod(1 + bench[valid])
    turnover = np.abs(np.diff(weights[valid], axis=0)).sum(axis=1)
    friction = turnover * cost
    net_daily = (1 + port[valid])[1:] * np.exp(-friction)
    port_net = np.concatenate([[1 + port[valid][0]], np.cumprod(net_daily)])
    return {
        "port": float(port_net[-1] - 1),
        "bench": float(bench_c[-1] - 1),
        "excess": float(port_net[-1] - bench_c[-1]),
        "turnover": float(turnover.mean() * 252 / rebalance),
        "n_days": int(valid.sum()),
    }


def main() -> int:
    print("加载面板 ...")
    panel = build_panel()
    print(f"  {len(panel):,} 行，{panel['symbol'].nunique()} 只")
    print()

    # ---- 先确认横截面 IC 的量级（这是方案 C 的前提）----
    print("===== 横截面 IC（同日跨股，预测未来 20 日收益）=====")
    panel["fwd20"] = (panel.groupby("symbol")["close"].shift(-20)
                      / panel["close"] - 1.0)
    for split in ("train", "val", "test"):
        a, b = SPLITS[split]
        sub = panel[(panel.year >= a) & (panel.year <= b)].dropna(subset=["fwd20"])
        values = []
        for date, group in sub.groupby("date"):
            if len(group) < 30:
                continue
            ic = rank_ic(group[SIGNAL].to_numpy(), group["fwd20"].to_numpy())
            if np.isfinite(ic):
                values.append(ic)
        print(f"  {split:5s} n={len(values):5d} 天  日均 IC {np.mean(values):+.4f}  "
              f"标准误 {np.std(values) / np.sqrt(len(values)):.4f}  "
              f"t ≈ {np.mean(values) / (np.std(values) / np.sqrt(len(values)) + 1e-12):+.1f}")

    # ---- 组合回测 ----
    print("\n===== 等权组合回测：每 20 日调仓，持有信号 top-K =====")
    print(f"    {'切分':6s}{'K':>5s}{'组合收益':>11s}{'基准收益':>11s}"
          f"{'超额':>11s}{'年换手':>9s}{'在市占比':>9s}")
    for split in ("train", "val", "test"):
        a, b = SPLITS[split]
        sub = panel[(panel.year >= a) & (panel.year <= b)].copy()
        for top_k in (5, 10, 20, 40, 80):
            r = portfolio_backtest(sub, top_k)
            exposure = top_k / panel["symbol"].nunique()
            print(f"    {split:6s}{top_k:>5d}{r['port']:>+11.2%}{r['bench']:>+11.2%}"
                  f"{r['excess']:>+11.2%}{r['turnover']:>9.1f}{exposure:>9.1%}")
        print()

    print("===== 信号加权版本（按信号强度分配权重）=====")
    for split in ("val", "test"):
        a, b = SPLITS[split]
        sub = panel[(panel.year >= a) & (panel.year <= b)].copy()
        for top_k in (20, 40):
            r = portfolio_backtest(sub, top_k, weight_by_signal=True)
            print(f"    {split:6s}K={top_k:<4d} 组合 {r['port']:+.2%}  "
                  f"基准 {r['bench']:+.2%}  超额 {r['excess']:+.2%}  "
                  f"年换手 {r['turnover']:.1f}")
    print()
    print("  读法：若某段出现「超额为正且换手可控」→ 方案 C 有可行性，")
    print("  值得把 RL 架构改成组合；若三段全负 → 连组合都救不了，")
    print("  必须先换特征族（残差/基本面/资金流）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
