"""方案 B 前置验证：联合量 vs 独立列 vs 现状，在监督学习台架上对比。

§1.9 已证明：把市场动量当**独立特征列**喂给模型是有害的
（换手 8.5→26.1，毛超额仍为 0）。

本脚本检验「联合量」路线是否有效：
  price            : 现有 9 个单股特征（= 当前观测）
  price+mkt_col    : 加市场动量等**独立列**（§1.9 已知有害，此处作对照）
  price+joint      : 加**联合量** —— 按 regime 把 ma20_relative 的符号预先校正
                     （熊市取反，牛/震荡取正），并加 regime 编码
  joint_only       : 只用联合量（检验联合量本身是否足够）

判据：联合量组的**毛超额**必须显著 > 0（不只是净超额，因为净超额会被
换手摩擦吃掉）；若毛超额仍 ≈ 0，则方案 B 也不值得做。

台架与 §1.9 相同：逐 (股票, 年) episode、T+1 生效、0.3% 往返摩擦。
"""
from __future__ import annotations

import sys
import time
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
MARKET_FEATURES = ("market_mom_5", "market_mom_20", "market_mom_60",
                   "market_vol_20", "market_breadth_20")
BULL = 0.05
BEAR = -0.05


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


def fit_mlp(X, y, hidden=48, epochs=80, lr=3e-3, l2=1e-4, seed=42):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    n = len(xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad()
            loss_fn(net(xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    return net


def predict(net, X):
    import torch
    with torch.no_grad():
        logits = net(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def backtest(data, prob, cost=ROUND_TRIP_COST):
    net, gross, switches, exposures = [], [], [], []
    for (symbol, year), group in data.groupby(["symbol", "year"], sort=False):
        group = group.sort_values("date")
        close = group["close"].to_numpy()
        if len(close) < 30:
            continue
        ret = np.zeros(len(close))
        ret[1:] = close[1:] / close[:-1] - 1.0
        pos = (prob[group.index][:-1] > 0.5).astype(float)
        days = ret[1:] * pos
        sw = float(np.abs(np.diff(pos)).sum())
        strat = float(np.prod(1 + days) - 1)
        hold = float(np.prod(1 + ret[1:]) - 1)
        net.append(strat - sw * cost - hold)
        gross.append(strat - hold)
        switches.append(sw)
        exposures.append(float(pos.mean()))
    return (float(np.mean(net)), float(np.mean(gross)),
            float(np.mean(switches)), float(np.mean(exposures)))


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

    # ---- 构造联合量 ----
    # regime 由 market_mom_20 决定（只用当日及以前）
    panel["regime"] = np.where(panel["market_mom_20"] > BULL, 1.0,
                                np.where(panel["market_mom_20"] < BEAR, -1.0, 0.0))
    # 联合量：按 regime 预先校正 ma20_relative 的符号
    #   熊市(regime=-1) → 取反（均值回归）；牛市/震荡 → 保持（原符号）
    panel["joint_ma20"] = np.where(panel["regime"] < 0,
                                   -panel["ma20_relative"],
                                   panel["ma20_relative"])
    panel["joint_mom5"] = np.where(panel["regime"] < 0,
                                   -panel["momentum_5"],
                                   panel["momentum_5"])
    panel["regime_code"] = panel["regime"]

    data = panel[panel["fwd"].notna()].copy()
    data = data[data["fwd"].abs() < 1.0].reset_index(drop=True)
    y = (data["fwd"].to_numpy() > 0).astype(float)

    def mask_of(split):
        a, b = SPLITS[split]
        return ((data.year >= a) & (data.year <= b)).to_numpy()

    train = mask_of("train")

    variants = {
        "price(现状)": list(FEATURES),
        "+独立市场列": list(FEATURES) + list(MARKET_FEATURES),
        "+联合量": list(FEATURES) + ["joint_ma20", "joint_mom5", "regime_code"],
        "仅联合量": ["joint_ma20", "joint_mom5", "regime_code",
                     "volatility_20", "volume_relative", "low_relative"],
    }

    print("\n===== 择时迷你回测：逐 (股票, 年) episode，含 0.3% 往返摩擦 =====")
    print(f"    {'特征集':14s}{'模型':8s}{'切分':6s}{'净超额':>10s}{'毛超额':>10s}"
          f"{'换手':>8s}{'在市':>8s}{'摩擦':>9s}")
    table = {}
    for vname, cols in variants.items():
        X = data[cols].to_numpy(float)
        mean = X[train].mean(0)
        std = X[train].std(0)
        std[std <= 0] = 1.0
        Z = np.clip((X - mean) / std, -10, 10)
        for mname in ("logreg", "mlp"):
            if mname == "logreg":
                w, b = fit_logistic(Z[train], y[train])
                prob = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
            else:
                net = fit_mlp(Z[train], y[train])
                prob = predict(net, Z)
            for split in ("val", "test"):
                m = mask_of(split)
                sub = data[m].reset_index(drop=True)
                n_ex, g_ex, sw, expo = backtest(sub, prob[m])
                print(f"    {vname:14s}{mname:8s}{split:6s}{n_ex:>+10.2%}{g_ex:>+10.2%}"
                      f"{sw:>8.1f}{expo:>8.1%}{g_ex - n_ex:>+9.2%}")
                table[(vname, mname, split)] = (n_ex, g_ex, sw)
        print()

    print("===== 判据：联合量 vs 现状 =====")
    for mname in ("logreg", "mlp"):
        base = table.get(("price(现状)", mname, "test"))
        joint = table.get(("+联合量", mname, "test"))
        col = table.get(("+独立市场列", mname, "test"))
        if base and joint:
            print(f"  {mname} test 净超额: 现状 {base[0]:+.2%} → "
                  f"独立列 {col[0]:+.2%} → 联合量 {joint[0]:+.2%}")
            print(f"  {mname} test 毛超额: 现状 {base[1]:+.2%} → "
                  f"独立列 {col[1]:+.2%} → 联合量 {joint[1]:+.2%}")
    print()
    print("  判读：联合量的**毛超额**若显著 > 0 且优于现状 → 方案 B（改观测契约）")
    print("  值得做；若毛超额仍 ≈ 0 → 说明连联合量都榨不出超额，")
    print("  问题在特征族本身，应转向方案 C 或更换数据（基本面/资金流）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
