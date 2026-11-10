"""解释矛盾：市场上下文为什么让超额大幅恶化？

两个看似冲突的事实：
  A. 市场动量的时序 IC 在牛熊之间符号翻转（牛 +0.023 / 熊 −0.080）
     → 暗示「知道 regime 就能选对信号方向」
  B. 加入市场特征后，监督学习在择时回测里超额大幅恶化
     （test 增量 −7.5% ~ −10.9%）

可能的解释（本脚本逐个检验）：
  H1 方向翻转是真的，但**单靠 regime 不足以选对方向**——还需要知道
     「当前这只股票相对它自己的历史处于什么位置」，两者要联合看。
  H2 市场特征带来的额外自由度让模型**过度换手**（换手从 8.5 → 26.1 → 41.0），
     摩擦吃掉了本就没有的毛边际。
  H3 市场特征在 train 上有微弱信号，在 test 上符号翻转，模型学到的
     是**噪声**，于是 test 表现崩塌。

检验方法：把毛超额（扣费前）与换手分开看，并对 train/test 的
市场特征 IC 分别计算。
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


def backtest(data, prob, cost=ROUND_TRIP_COST):
    """返回 (净超额均值, 毛超额均值, 平均换手, 平均在市)。"""
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


def rank_ic(x, y):
    if len(x) < 30:
        return np.nan
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


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
    y = (data["fwd"].to_numpy() > 0).astype(float)

    def mask_of(split):
        a, b = SPLITS[split]
        return ((data.year >= a) & (data.year <= b)).to_numpy()

    train = mask_of("train")

    print("\n===== H3 检验：市场特征在各段的 IC（对个股未来 20 日收益）=====")
    print(f"    {'特征':22s}{'train':>10s}{'val':>10s}{'test':>10s}{'符号一致?':>10s}")
    for name in MARKET_FEATURES:
        ics = []
        for split in ("train", "val", "test"):
            m = mask_of(split)
            ics.append(rank_ic(data.loc[m, name].to_numpy(),
                               data.loc[m, "fwd"].to_numpy()))
        same = "是" if (ics[0] > 0) == (ics[2] > 0) else "★否★"
        print(f"    {name:22s}{ics[0]:>+10.4f}{ics[1]:>+10.4f}{ics[2]:>+10.4f}{same:>10s}")
    print()
    print("    价格特征对照：")
    for name in ("ma20_relative", "momentum_5", "volatility_20"):
        ics = []
        for split in ("train", "val", "test"):
            m = mask_of(split)
            ics.append(rank_ic(data.loc[m, name].to_numpy(),
                               data.loc[m, "fwd"].to_numpy()))
        same = "是" if (ics[0] > 0) == (ics[2] > 0) else "★否★"
        print(f"    {name:22s}{ics[0]:>+10.4f}{ics[1]:>+10.4f}{ics[2]:>+10.4f}{same:>10s}")

    print("\n===== H2 检验：毛超额（扣费前）与换手分解 =====")
    print(f"    {'特征集':14s}{'切分':6s}{'净超额':>11s}{'毛超额':>11s}"
          f"{'换手':>8s}{'在市':>8s}{'摩擦拖累':>11s}")
    for vname, cols in (("price", list(FEATURES)),
                        ("price+mkt", list(FEATURES) + list(MARKET_FEATURES))):
        X = data[cols].to_numpy(float)
        mean = X[train].mean(0)
        std = X[train].std(0)
        std[std <= 0] = 1.0
        Z = np.clip((X - mean) / std, -10, 10)
        w, b = fit_logistic(Z[train], y[train])
        prob = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        for split in ("val", "test"):
            m = mask_of(split)
            sub = data[m].reset_index(drop=True)
            net, gross, sw, expo = backtest(sub, prob[m])
            print(f"    {vname:14s}{split:6s}{net:>+11.2%}{gross:>+11.2%}"
                  f"{sw:>8.1f}{expo:>8.1%}{gross - net:>+11.2%}")
    print()
    print("    读法：若「毛超额」在两套特征下都接近 0，而「净超额」随换手上升")
    print("    而下降 → 摩擦是主因（H2），模型在拟合噪声。")

    print("\n===== H1 检验：方向翻转是否需要「regime × 个股位置」的联合条件？=====")
    # 在牛/熊两态下，分别看 ma20_relative 的 IC；若符号确实翻转，
    # 则说明「知道 regime」是必要但不充分的条件
    data["mkt_regime"] = np.where(data["market_mom_20"] > 0.05, "牛",
                                   np.where(data["market_mom_20"] < -0.05, "熊", "震荡"))
    for split in ("train", "test"):
        print(f"    {split} 段：")
        for regime in ("牛", "熊", "震荡"):
            part = data[(data.year >= SPLITS[split][0]) & (data.year <= SPLITS[split][1])
                        & (data.mkt_regime == regime)]
            if len(part) < 500:
                continue
            ic = rank_ic(part["ma20_relative"].to_numpy(), part["fwd"].to_numpy())
            print(f"      {regime:4s} (n={len(part):6d})  ma20_relative 的 20 日 IC = {ic:+.4f}")
    print()
    print("    若「牛熊 IC 符号翻转」在 train 与 test 都成立 → H1 成立：")
    print("    regime 是必要条件，但必须与个股位置联合使用，")
    print("    单纯把市场动量塞进特征让模型自己学，反而给了它拟合噪声的自由度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
