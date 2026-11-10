"""方案 D+E 联合验证：残差特征 + 低换手，能否让毛超额转正？

两条已确立的事实：
  ① 稳定信号在**个股相对市场的残差**里（残差 IC: train −0.0241 /
     val −0.0347 / test −0.0294），而单股择时赚绝对收益 → 架构用不上
  ② 摩擦门槛高：年换手 17 倍需毛边际 ≥5.1% 才能打平

本脚本检验两个最小改动能否打开空间：
  D: 把**残差序列直接作为特征**（让模型不必自己学「减掉 beta」）
     —— 残差 = 个股收益 − beta×市场收益（beta 用 train 段滚动估计）
  E: 把**换手频率**从每日降到 20 日（用阈值 + 滞后带抑制）

度量：逐 (股票, 年) episode 的择时迷你回测，报**毛超额**（扣费前）与净超额。
判据：毛超额显著 > 0 且净超额 > 0。

beta 估计必须无泄漏：滚动回归只用 T 及以前的数据。
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
REBALANCE = 20


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
    """滚动 beta：个股日收益对等权市场日收益回归，只用 T 及以前数据。

    beta_t = cov(r_i, r_m)[窗口内] / var(r_m)[窗口内]，窗口截至 T-1。
    """
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
            # 只用 T-1 及以前的窗口（严格因果）
            x = r_m[lo:t]
            y = r_i[lo:t]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < min_periods:
                continue
            x, y = x[ok], y[ok]
            var = float(np.var(x))
            if var <= 1e-14:
                continue
            betas[t] = float(np.cov(x, y)[0, 1] / var)
        for d, b in zip(dates, betas):
            out[(symbol, d)] = b
    return out


def backtest(data, prob_full, rows, cost=ROUND_TRIP_COST, rebalance=1):
    """逐 (股票, 年) episode 回测。rebalance>1 时只允许每隔 N 日换仓。

    **因果对齐（关键）**：
      信号在第 t 日收盘算出（含 close[t]），只能赚 **t → t+1** 的收益。
      即 days[k] = ret[k+1] × pos[k]，pos[k] 由第 k 日的信号决定。
      若写成 days[k] = ret[k] × pos[k] 就是**前视泄漏**（用含 close[k] 的
      信号去乘已经发生的 k−1→k 收益），会得出虚高的正超额。

    参数：
      data     : 切分子集
      prob_full: 整个 data 上的概率数组
      rows     : data 各行在 prob_full 中的原始行号
    """
    net, gross, switches = [], [], []
    for _key, group in data.groupby(["symbol", "year"], sort=False):
        group = group.sort_values("date")
        close = group["close"].to_numpy()
        if len(close) < 40:
            continue
        ret = np.zeros(len(close))
        ret[1:] = close[1:] / close[:-1] - 1.0
        raw = (prob_full[group["_row"].to_numpy()] > 0.5).astype(float)
        # 第 t 日信号 → 第 t+1 日持仓（严格因果）
        pos = np.zeros(len(raw))
        held = False
        last_change = -10 ** 9
        for t in range(len(raw) - 1):
            want = raw[t]
            if want != held and (t - last_change) >= rebalance:
                held = want
                last_change = t
            pos[t] = held
        days = ret[1:] * pos[:-1]          # ← 因果：信号 t 吃 ret[t+1]
        sw = float(np.abs(np.diff(pos[:-1])).sum())
        strat = float(np.prod(1 + days) - 1)
        hold = float(np.prod(1 + ret[1:]) - 1)
        net.append(strat - sw * cost - hold)
        gross.append(strat - hold)
        switches.append(sw)
    return (float(np.mean(net)), float(np.mean(gross)),
            float(np.mean(switches)))


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
    print("估计滚动 beta（只用 T-1 及以前）...")
    betas = rolling_beta(panel)
    panel["beta"] = [betas.get((s, d), np.nan) for s, d in
                     zip(panel["symbol"], panel["date"])]
    coverage = float(panel["beta"].notna().mean())
    print(f"  beta 覆盖 {coverage:.1%}（其余为窗口不足）")
    panel["beta"] = panel["beta"].fillna(1.0)

    # 残差序列：个股收益 − beta×市场收益
    rows = []
    for symbol, group in panel.groupby("symbol", sort=False):
        rows.append(pd.DataFrame({"date": group["date"].to_numpy(),
                                   "ret": group["ret"].to_numpy()}))
    frame = pd.concat(rows, ignore_index=True)
    market = frame.groupby("date")["ret"].mean().sort_index()
    panel["mkt_ret"] = panel["date"].map(market)
    panel["resid_ret"] = panel["ret"] - panel["beta"] * panel["mkt_ret"]

    # 残差的累计（作为特征）：过去 5/20 日残差累计
    # 注意：groupby.rolling 后按 (symbol, date) 重新对齐，
    # 且必须 fillna(0) —— NaN 会污染整个逻辑回归（权重变 NaN、换手变 0）。
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
    y = (data["fwd"].to_numpy() > 0).astype(float)
    # 保留一个与 prob 完全对齐的行号列：切分子集时用它还原位置
    data["_row"] = np.arange(len(data))
    def mask_of(split):
        a, b = SPLITS[split]
        return ((data.year >= a) & (data.year <= b)).to_numpy()

    train = mask_of("train")

    variants = {
        "现状(9 特征)": list(FEATURES),
        "+残差特征": list(FEATURES) + ["resid_cum_5", "resid_cum_20", "beta_level"],
    }

    print("\n===== 方案 D+E：残差特征 × 换手频率 =====")
    print(f"    {'特征集':16s}{'调仓间隔':>9s}{'切分':6s}{'净超额':>10s}{'毛超额':>10s}"
          f"{'年换手':>9s}")
    for vname, cols in variants.items():
        X = data[cols].to_numpy(float)
        mean = X[train].mean(0)
        std = X[train].std(0)
        std[std <= 0] = 1.0
        Z = np.clip((X - mean) / std, -10, 10)
        w, b = fit_logistic(Z[train], y[train])
        prob = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        for rebalance in (1, 10, 20):
            for split in ("val", "test"):
                m = mask_of(split)
                sub = data[m].copy()
                n_ex, g_ex, sw = backtest(sub, prob, sub["_row"].to_numpy(),
                                          rebalance=rebalance)
                print(f"    {vname:16s}{rebalance:>9d}{split:6s}{n_ex:>+10.2%}"
                      f"{g_ex:>+10.2%}{sw:>9.1f}")
        print()

    print("===== 残差特征本身的预测力检验 =====")
    for name in ("resid_cum_5", "resid_cum_20", "beta_level"):
        cells = []
        for split in ("train", "val", "test"):
            m = mask_of(split)
            x = data.loc[m, name].to_numpy()
            yy = data.loc[m, "fwd"].to_numpy()
            xr = np.argsort(np.argsort(x))
            yr = np.argsort(np.argsort(yy))
            cells.append(f"{split}={np.corrcoef(xr, yr)[0, 1]:+.4f}")
        print(f"    {name:16s} " + "  ".join(cells))
    print()

    print("===== 台架自检：未来知情信号必须显著为正 =====")
    print("  用「明日收益符号」当信号（作弊），若台架对齐正确，毛超额应显著为正；")
    print("  若它也不为正，说明回测 harness 本身有问题，不能用来判断其他结论。")
    m = mask_of("test")
    sub = data[m].copy()
    rows = sub["_row"].to_numpy()
    cheat = np.zeros(len(data))
    for _key, group in data.groupby(["symbol", "year"], sort=False):
        idx = group["_row"].to_numpy()
        r = group["ret"].to_numpy()
        nxt = np.roll(r, -1)      # 组内下一交易日的收益（作弊信号）
        nxt[-1] = 0.0
        cheat[idx] = nxt
    n_ex, g_ex, sw = backtest(sub, (cheat > 0).astype(float), rows, rebalance=1)
    print(f"  作弊信号 test: 净超额 {n_ex:+.2%}  毛超额 {g_ex:+.2%}  换手 {sw:.1f}")
    print(f"  → {'✓ 台架可用（能识别真实优势）' if g_ex > 0.05 else '✗ 台架有问题，不能采信'}")

    print("\n===== 参照：随机信号（因波动拖累必然为负，不是 0）=====")
    rng = np.random.default_rng(7)
    random_prob = rng.random(len(data))
    n_ex, g_ex, sw = backtest(sub, random_prob, rows, rebalance=1)
    print(f"  随机信号 test: 净超额 {n_ex:+.2%}  毛超额 {g_ex:+.2%}  换手 {sw:.1f}")
    print("  说明：随机择时少赚 beta 又承担复利拖累，故必然为负；")
    print("        真实信号应显著优于这个负基准。")
    print()
    print("  判据：若「+残差特征」组的毛超额显著优于随机基准、且优于现状，")
    print("  则方案 D+E 成立 → 可以先改数据管线，再改 RL 环境。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
