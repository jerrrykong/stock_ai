"""残差特征为何 val 有效、test 无效？——过拟合还是 regime 依赖？

现象（§1.14）：+残差特征在 val 段毛超额 +3.00%（优于现状 +0.93%），
但 test 段 −8.28%（现状 −0.19%），train→test 崩塌。

两种解释，处理方式完全不同：
  H_overfit  : 残差特征是噪声，模型在 train 上拟合了它 → 需要更多正则/更少自由度
  H_regime   : 残差信号只在特定市场状态下有效（train/val 恰逢该状态，test 相反）
               → 需要按 regime 分段评估，条件性使用

检验：
  ① 逐年分解：残差特征组与现状组在**每一年**的超额对比。
     若残差组只在个别年份好 → 过拟合（挑中的年份是运气）。
  ② 按市场 regime 分段：分别在牛/熊/震荡下算超额。
     若残差组只在特定 regime 好 → H_regime。
  ③ 训练段内部的时间外推检验（train 前 70% 拟合 → train 后 30% 评估），
     排除「样本量不足导致的不稳定」。
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
from timing_backtest import TimingBacktest  # noqa: E402

FEATURE_DIR = ROOT / "data" / "raw_hfq_v2_train"
SPLITS = {"train": (2006, 2021), "val": (2022, 2023), "test": (2024, 2025)}
HORIZON = 20


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
    panel["fwd"] = (panel.groupby("symbol")["close"].shift(-HORIZON)
                    / panel["close"] - 1.0)
    panel["year"] = panel["date"].dt.year
    return panel


def rolling_beta(panel, window=250, min_periods=60):
    rows = []
    for symbol, group in panel.groupby("symbol", sort=False):
        close = group["close"].to_numpy()
        ret = np.zeros(len(close))
        ret[1:] = close[1:] / close[:-1] - 1.0
        rows.append(pd.DataFrame({"date": group["date"].to_numpy(), "ret": ret}))
    frame = pd.concat(rows, ignore_index=True)
    market = frame.groupby("date")["ret"].mean().sort_index()
    market = market.reindex(panel["date"].unique()).sort_index()
    out = {}
    for symbol, group in panel.groupby("symbol", sort=False):
        dates = group["date"].to_numpy()
        r_i = group["ret"].to_numpy()
        r_m = market.reindex(dates).to_numpy()
        betas = np.full(len(dates), np.nan)
        for t in range(len(dates)):
            if t < min_periods:
                continue
            lo = max(0, t - window)
            x, y = r_m[lo:t], r_i[lo:t]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < min_periods:
                continue
            x, y = x[ok], y[ok]
            var = float(np.var(x))
            if var > 1e-14:
                betas[t] = float(np.cov(x, y)[0, 1] / var)
        for d, b in zip(dates, betas):
            out[(symbol, d)] = b
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
    betas = rolling_beta(panel)
    panel["beta"] = [betas.get((s, d), np.nan) for s, d in
                     zip(panel["symbol"], panel["date"])]
    panel["beta"] = panel["beta"].fillna(1.0)
    rows = []
    for symbol, group in panel.groupby("symbol", sort=False):
        rows.append(pd.DataFrame({"date": group["date"].to_numpy(),
                                   "ret": group["ret"].to_numpy()}))
    frame = pd.concat(rows, ignore_index=True)
    market = frame.groupby("date")["ret"].mean().sort_index()
    panel["mkt_ret"] = panel["date"].map(market)
    panel["resid_ret"] = panel["ret"] - panel["beta"] * panel["mkt_ret"]
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    for k in (5, 20):
        rolled = (panel.groupby("symbol", sort=False)["resid_ret"]
                  .rolling(k, min_periods=1).sum()
                  .reset_index(level=0, drop=True))
        panel[f"resid_cum_{k}"] = np.nan_to_num(rolled.to_numpy(dtype="float64"),
                                                 nan=0.0, posinf=0.0, neginf=0.0)
    panel["beta_level"] = np.nan_to_num((panel["beta"] - 1.0).to_numpy(dtype="float64"),
                                        nan=0.0, posinf=0.0, neginf=0.0)

    data = panel[panel["fwd"].notna()].copy()
    data = data[data["fwd"].abs() < 1.0].reset_index(drop=True)
    data["_row"] = np.arange(len(data))
    y = (data["fwd"].to_numpy() > 0).astype(float)

    # 市场状态序列（对 panel 每行）
    mkt_equity = (1.0 + market).cumprod()
    mom20_map = {}
    vals = mkt_equity.to_numpy()
    dts = mkt_equity.index.to_numpy()
    for i, d in enumerate(dts):
        s = max(0, i - 19)
        mom20_map[d] = vals[i] / vals[s] - 1.0 if i > 0 else 0.0
    data["mkt_mom20"] = data["date"].map(mom20_map).fillna(0.0)
    data["regime"] = np.where(data["mkt_mom20"] > 0.05, "牛",
                              np.where(data["mkt_mom20"] < -0.05, "熊", "震荡"))

    train_mask = ((data.year >= 2006) & (data.year <= 2021)).to_numpy()
    variants = {
        "现状": list(FEATURES),
        "+残差": list(FEATURES) + ["resid_cum_5", "resid_cum_20", "beta_level"],
    }
    probs = {}
    for vname, cols in variants.items():
        X = data[cols].to_numpy(float)
        mean = X[train_mask].mean(0)
        std = X[train_mask].std(0)
        std[std <= 0] = 1.0
        Z = np.clip((X - mean) / std, -10, 10)
        w, b = fit_logistic(Z[train_mask], y[train_mask])
        probs[vname] = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))

    # ---------- ① 逐年分解 ----------
    print("===== ① 逐年毛超额分解（调仓间隔 20）=====")
    print(f"    {'年份':6s}{'现状':>10s}{'+残差':>10s}{'差':>10s}"
          f"{'市场动量':>10s}{'regime':>8s}")
    for year in range(2006, 2026):
        sub = data[data.year == year].copy()
        if len(sub) < 100:
            continue
        line = f"    {year:<6d}"
        values = {}
        for vname in variants:
            bt = TimingBacktest(sub, rebalance=20, global_len=len(data))
            r = bt.run_probability(probs[vname])
            values[vname] = r.gross_excess
            line += f"{r.gross_excess:>+10.2%}"
        line += f"{values['+残差'] - values['现状']:>+10.2%}"
        mom = float(sub["mkt_mom20"].mean())
        reg = "牛" if mom > 0.05 else ("熊" if mom < -0.05 else "震荡")
        line += f"{mom:>+10.2%}{reg:>8s}"
        print(line)

    # ---------- ② 按 regime 分段 ----------
    print("\n===== ② 按市场状态分段的毛超额（调仓间隔 20）=====")
    for vname in variants:
        cells = []
        for regime in ("牛", "熊", "震荡"):
            parts = []
            for split, (a, b) in SPLITS.items():
                sub = data[(data.year >= a) & (data.year <= b) & (data.regime == regime)].copy()
                if len(sub) < 200:
                    continue
                bt = TimingBacktest(sub, rebalance=20, global_len=len(data))
                r = bt.run_probability(probs[vname])
                parts.append((split, r.gross_excess, len(sub)))
            cells.append(f"{regime}[" + ", ".join(f"{s} {g:+.2%}" for s, g, _ in parts) + "]")
        print(f"  {vname:8s} " + "  ".join(cells))

    # ---------- ③ train 内部时间外推 ----------
    print("\n===== ③ train 内部时间外推（拟合 train 前 70% → 评估后 30%）=====")
    train_years = list(range(2006, 2022))
    cut = int(len(train_years) * 0.7)
    early = train_years[:cut]
    late = train_years[cut:]
    print(f"  拟合段 {early[0]}~{early[-1]}  评估段 {late[0]}~{late[-1]}")
    early_mask = data.year.isin(early).to_numpy()
    late_mask = data.year.isin(late).to_numpy()
    for vname, cols in variants.items():
        X = data[cols].to_numpy(float)
        mean = X[early_mask].mean(0)
        std = X[early_mask].std(0)
        std[std <= 0] = 1.0
        Z = np.clip((X - mean) / std, -10, 10)
        w, b = fit_logistic(Z[early_mask], y[early_mask])
        prob = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        sub = data[late_mask].copy()
        bt = TimingBacktest(sub, rebalance=20, global_len=len(data))
        r = bt.run_probability(prob)
        print(f"  {vname:8s} 评估段毛超额 {r.gross_excess:+.2%}  净 {r.net_excess:+.2%}")
    print()
    print("  判读：")
    print("    若 train 内部外推也为负 → 残差特征本身就不稳（不是 test 特有）")
    print("    若 train 内部外推为正、只有 test 为负 → test 期恰逢不利 regime（H_regime）")
    print("    若逐年分解显示残差组只在个别年份好 → 过拟合（H_overfit）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
