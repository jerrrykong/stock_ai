"""决定性验证：加入市场上下文后，同一套特征能否变出可执行的择时 alpha？

背景（已有诊断）：
  - ma20_relative 的 20 日 IC 在牛熊之间**符号翻转**（牛 +0.023 / 熊 −0.080）
  - 现有观测 (20, 9) 全是单股相对价格，**不含任何市场信息**
  - 但监督学习在纯价格特征上 test 年化只有 +0.32%（MLP）→ 经济价值极低

本脚本回答：「市场上下文」是 (a) 无用的装饰，还是 (b) 让信号变得可执行的关键？

做法：把「择时」任务做成忠实的迷你回测（而非样本级分类）：
  - 逐 (股票, 年) episode，逐日推进
  - 第 T 日用特征预测，第 T+1 日生效（T+1，无前视）
  - 持仓日吃该股日收益，空仓日收益 0
  - 每次换仓扣一次往返摩擦（买 0.125% + 卖 0.175% ≈ 0.3%）
  - 与同股票同年「一直持有」对比 → 逐 episode 超额

三种特征集对照：
  price        : 现有 9 个单股特征（= 当前观测）
  price+mkt    : 加市场上下文（动量/波动/广度）
  regime_cond  : 市场状态显式条件化（牛用动量信号、熊用反转信号）
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

#: 往返摩擦：买入 佣金0.025%+滑点0.1% = 0.125%；卖出 佣金0.025%+印花税0.05%+滑点0.1% = 0.175%
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
    """等权市场代理 + 广度。全部只用当日及以前数据，无前视。"""
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
        flag = close > ma20
        dates = group["date"].to_numpy()
        for i, date in enumerate(dates):
            above[date] = flag[i]
    breadth = pd.Series(above).groupby(level=0).mean()
    for date in out:
        out[date]["market_breadth_20"] = float(breadth.get(date, 0.5))
    return out


def fit_logistic(X, y, epochs=250, lr=0.05, l2=1e-3, seed=42):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    mw = np.zeros(d); vw = np.zeros(d); mb = 0.0; vb = 0.0
    for t in range(1, epochs + 1):
        z = np.clip(X @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
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


def minibacktest(data, prob, cost=ROUND_TRIP_COST):
    """逐 (股票, 年) 迷你回测：信号 T 日出、T+1 生效。返回逐 episode 超额。"""
    results = []
    for (symbol, year), group in data.groupby(["symbol", "year"], sort=False):
        group = group.sort_values("date")
        close = group["close"].to_numpy()
        if len(close) < 30:
            continue
        ret = np.zeros(len(close))
        ret[1:] = close[1:] / close[:-1] - 1.0
        p = prob[group.index]
        # 第 t 日信号 → 第 t+1 日持仓
        pos = (p[:-1] > 0.5).astype(float)
        days = ret[1:] * pos
        # 换仓摩擦
        switches = np.abs(np.diff(pos)).sum()
        strat = float(np.prod(1 + days) - 1 - switches * cost)
        hold = float(np.prod(1 + ret[1:]) - 1)
        results.append({"symbol": symbol, "year": int(year),
                        "excess": strat - hold, "switch": float(switches),
                        "exposure": float(pos.mean())})
    return results


def main() -> int:
    print("加载面板 ...")
    panel = build_panel()
    market = build_market(panel)
    market_frame = pd.DataFrame.from_dict(market, orient="index")
    market_frame.index.name = "date"
    market_frame = market_frame.reset_index()
    # 键格式必须与面板侧一致（.dt.strftime('%Y%m%d')）。
    # 直接 astype(str) 会得到 '2006-01-04'，与面板的 '20060104' 匹配不上，
    # 导致市场特征被 fillna(0) 全填成 0（曾导致「加市场特征结果完全不变」）。
    market_frame["date_key"] = market_frame["date"].dt.strftime("%Y%m%d")
    panel["date_key"] = panel["date"].dt.strftime("%Y%m%d")
    before = len(panel)
    panel = panel.merge(market_frame[["date_key", *MARKET_FEATURES]],
                        on="date_key", how="left")
    assert len(panel) == before, "merge 改变了行数，说明键不唯一"
    for name in MARKET_FEATURES:
        panel[name] = panel[name].fillna(0.0)
    zero_frac = float((panel[list(MARKET_FEATURES)] == 0).all(axis=1).mean())
    print(f"  面板 {len(panel):,} 行，市场特征 {len(MARKET_FEATURES)} 个")
    print(f"  合并后全零行占比 {zero_frac:.2%}（应接近 0，否则说明键仍不匹配）")
    assert zero_frac < 0.02, f"市场特征合并失败：全零行占比 {zero_frac:.1%}"

    data = panel[panel["fwd"].notna()].copy()
    data = data[data["fwd"].abs() < 1.0]
    data = data.reset_index(drop=True)
    y = (data["fwd"].to_numpy() > 0).astype(np.float64)

    def mask_of(split):
        a, b = SPLITS[split]
        return ((data.year >= a) & (data.year <= b)).to_numpy()

    train = mask_of("train")
    variants = {
        "price": list(FEATURES),
        "price+mkt": list(FEATURES) + list(MARKET_FEATURES),
    }

    print("\n===== 择时迷你回测：逐 (股票, 年) episode，含往返摩擦 =====")
    print(f"    {'特征集':14s}{'模型':8s}{'切分':6s}{'超额均值':>11s}{'中位数':>11s}"
          f"{'为正占比':>9s}{'年换手':>9s}{'在市':>8s}")
    summary = {}
    for vname, cols in variants.items():
        X = data[cols].to_numpy(dtype="float64")
        mean = X[train].mean(axis=0)
        std = X[train].std(axis=0)
        std[std <= 0] = 1.0
        Z = np.clip((X - mean) / std, -10, 10)
        for mname in ("logreg", "mlp"):
            t0 = time.time()
            if mname == "logreg":
                w, b = fit_logistic(Z[train], y[train])
                prob_all = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
            else:
                net = fit_mlp(Z[train], y[train])
                prob_all = predict(net, Z)
            fit_time = time.time() - t0
            for split in ("val", "test"):
                m = mask_of(split)
                sub = data[m].reset_index(drop=True)
                prob = prob_all[m]
                rows = minibacktest(sub, prob)
                excess = np.array([r["excess"] for r in rows])
                switch = np.mean([r["switch"] for r in rows])
                expo = np.mean([r["exposure"] for r in rows])
                print(f"    {vname:14s}{mname:8s}{split:6s}{excess.mean():>+11.2%}"
                      f"{np.median(excess):>+11.2%}{float((excess > 0).mean()):>9.1%}"
                      f"{switch:>9.1f}{expo:>8.1%}")
                summary[(vname, mname, split)] = excess.mean()
        print()

    print("===== 增量：市场上下文带来的提升 =====")
    for mname in ("logreg", "mlp"):
        for split in ("val", "test"):
            a = summary.get(("price", mname, split))
            b = summary.get(("price+mkt", mname, split))
            if a is not None and b is not None:
                print(f"  {mname:7s} {split:5s}: price {a:+.2%} → +mkt {b:+.2%}"
                      f"   增量 {b - a:+.2%}")
    print()
    print("  若增量显著为正 → 市场上下文是把信号变成可执行 alpha 的关键，")
    print("     方案 B（观测契约升级）值得做。")
    print("  若增量≈0 → 现有特征族无论加不加市场信息都榨不出超额，")
    print("     应转向方案 C（组合化/选股）或换特征族。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
