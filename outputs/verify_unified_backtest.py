"""用统一台架（outputs/timing_backtest.py）重跑关键结论，验证可复现性。

三条已知结论（来自各自的独立脚本）应被复现：
  1. 现状 9 特征 + logreg：val 净 −2.07% / 毛 −0.03%；test 净 −2.73% / 毛 −0.19%
  2. +市场独立列：test 净 −10.24%（比现状更差）
  3. +残差特征（调仓 20 日）：val 净 +1.91% / 毛 +3.00%；test 净 −9.35%

若本脚本能同时复现这三条，说明台架口径统一、可信。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "outputs"))

from stock_ai.scale.unified import FEATURES  # noqa: E402
from timing_backtest import TimingBacktest, split_by_year  # noqa: E402

FEATURE_DIR = ROOT / "data" / "raw_hfq_v2_train"
SPLITS = {"train": (2006, 2021), "val": (2022, 2023), "test": (2024, 2025)}
MARKET_FEATURES = ("market_mom_5", "market_mom_20", "market_mom_60",
                   "market_vol_20", "market_breadth_20")


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
    panel["fwd"] = (panel.groupby("symbol")["close"].shift(-20)
                    / panel["close"] - 1.0)
    panel["year"] = panel["date"].dt.year
    return panel


def build_market(panel):
    rows = []
    for symbol, group in panel.groupby("symbol", sort=False):
        close = group["close"].to_numpy()
        ret = np.zeros(len(close))
        ret[1:] = close[1:] / close[:-1] - 1.0
        rows.append(pd.DataFrame({"date": group["date"].to_numpy(), "ret": ret}))
    frame = pd.concat(rows, ignore_index=True)
    daily = frame.groupby("date")["ret"].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    values = equity.to_numpy()
    dates = equity.index.to_numpy()
    out = {}
    for i, date in enumerate(dates):
        rec = {}
        for k in (5, 20, 60):
            start = max(0, i - k + 1)
            rec[f"market_mom_{k}"] = (values[i] / values[start] - 1.0) if i > 0 else 0.0
        window = daily.iloc[max(0, i - 19):i + 1]
        rec["market_vol_20"] = float(window.std(ddof=0)) if len(window) > 1 else 0.0
        out[date] = rec
    # 广度：当日收盘高于自身 20 日均线的股票占比
    above = {}
    for symbol, group in panel.groupby("symbol", sort=False):
        close = group["close"].to_numpy()
        ma20 = pd.Series(close).rolling(20, min_periods=1).mean().to_numpy()
        for date, flag in zip(group["date"].to_numpy(), close > ma20):
            above[date] = flag
    breadth = pd.Series(above).groupby(level=0).mean()
    for date in out:
        out[date]["market_breadth_20"] = float(breadth.get(date, 0.5))
    return out


def fit_logistic(X, y, epochs=250, lr=0.05, l2=1e-3):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    mw = np.zeros(d); vw = np.zeros(d); mb = 0.0; vb = 0.0
    for t in range(1, epochs + 1):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        mw = 0.9 * mw + 0.1 * grad_w; vw = 0.999 * vw + 0.001 * grad_w ** 2
        mb = 0.9 * mb + 0.1 * grad_b; vb = 0.999 * vb + 0.001 * grad_b ** 2
        w -= lr * (mw / (1 - 0.9 ** t)) / (np.sqrt(vw / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return w, b


def main() -> int:
    print("加载面板 ...")
    panel = build_panel()
    market = build_market(panel)
    mf = pd.DataFrame.from_dict(market, orient="index").reset_index()
    mf = mf.rename(columns={"index": "date"})
    mf["date_key"] = mf["date"].dt.strftime("%Y%m%d")
    panel["date_key"] = panel["date"].dt.strftime("%Y%m%d")
    panel = panel.merge(mf[["date_key", *MARKET_FEATURES]], on="date_key", how="left")
    for c in MARKET_FEATURES:
        panel[c] = panel[c].fillna(0.0)

    data = panel[panel["fwd"].notna()].copy()
    data = data[data["fwd"].abs() < 1.0].reset_index(drop=True)
    # 保留全局行号：切分后的子集靠它与全局概率数组对齐
    data["_row"] = np.arange(len(data))
    y = (data["fwd"].to_numpy() > 0).astype(float)
    parts = split_by_year(data, SPLITS)
    train_mask = ((data.year >= 2006) & (data.year <= 2021)).to_numpy()

    print("\n===== 台架自检（test 段）=====")
    bt_test = TimingBacktest(parts["test"], rebalance=1, global_len=len(data))
    bt_test.report_self_check()

    variants = {
        "现状 9 特征": list(FEATURES),
        "+市场独立列": list(FEATURES) + list(MARKET_FEATURES),
    }
    print("===== 用统一台架重跑 =====")
    print(f"    {'特征集':16s}{'切分':6s}{'净超额':>10s}{'毛超额':>10s}"
          f"{'年换手':>9s}{'在市':>8s}")
    for vname, cols in variants.items():
        X = data[cols].to_numpy(float)
        mean = X[train_mask].mean(0)
        std = X[train_mask].std(0)
        std[std <= 0] = 1.0
        Z = np.clip((X - mean) / std, -10, 10)
        w, b = fit_logistic(Z[train_mask], y[train_mask])
        prob = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        for split in ("val", "test"):
            sub = parts[split]
            # 把整段概率按行号映射到子集
            sub_prob = prob[sub["_row"].to_numpy()] if "_row" in sub else prob
            bt = TimingBacktest(sub, rebalance=1, global_len=len(data))
            r = bt.run_probability(prob)
            print(f"    {vname:16s}{split:6s}{r.net_excess:>+10.2%}"
                  f"{r.gross_excess:>+10.2%}{r.switches:>9.1f}{r.exposure:>8.1%}")
        print()

    print("  期望复现：")
    print("    现状 9 特征   val −2.07%/−0.03%   test −2.73%/−0.19%")
    print("    +市场独立列  test −10.24%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
