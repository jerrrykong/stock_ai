"""量化「缺失市场上下文」这个特征缺口的代价。

现有观测 = (20 窗口, 9 个单股相对价格特征)，**完全不含市场信息**。
而诊断已证明：ma20_relative 的 IC 在牛熊之间符号翻转（牛 +0.023 / 熊 −0.080），
即「该反转还是该跟随」取决于市场状态——但模型看不见市场状态。

本脚本：
  1. 把单股未来 20 日收益拆成 市场部分 + 个股残差
  2. 算各特征对「残差收益」的 IC（去掉市场后还剩多少信号）
  3. 算各特征对「市场收益」的 IC（市场动量本身可预测吗）
  4. 量化：若观测里加入市场动量，可多捕获多少

回答的问题是：加市场特征值不值得改观测契约。
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


def market_series(panel):
    """等权市场代理：日收益横截面均值 → 净值 → 20 日动量。"""
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
    mom = {}
    for i, date in enumerate(dates):
        start = max(0, i - 19)
        mom[date] = values[i] / values[start] - 1.0 if i > 0 else 0.0
    return mom, daily


def rank_ic(x, y):
    """Spearman 秩相关。"""
    if len(x) < 30:
        return np.nan
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> int:
    print("加载面板 ...")
    panel = build_panel()
    market_mom, daily = market_series(panel)

    # 未来 20 日的市场收益（对齐到个股日期）
    market_fwd = {}
    values = (1.0 + daily).cumprod().to_numpy()
    dates = daily.index.to_numpy()
    for i, date in enumerate(dates):
        j = i + HORIZON
        market_fwd[date] = (values[j] / values[i] - 1.0) if j < len(values) else np.nan

    panel["mkt_mom"] = panel["date"].map(market_mom)
    panel["mkt_fwd"] = panel["date"].map(market_fwd)
    data = panel[panel["fwd"].notna() & panel["mkt_fwd"].notna()].copy()
    data = data[data["fwd"].abs() < 1.0]
    # 残差收益 = 个股未来收益 − 市场未来收益（beta≈1 的简化口径）
    data["resid"] = data["fwd"] - data["mkt_fwd"]
    data["split"] = data["year"].map(
        lambda y: next((k for k, (a, b) in SPLITS.items() if a <= y <= b), "out"))
    print(f"  有效样本 {len(data):,}")

    print("\n===== 1. 收益分解：市场 vs 个股残差 =====")
    for split in ("train", "val", "test"):
        sub = data[data.split == split]
        var_total = float(sub["fwd"].var())
        var_resid = float(sub["resid"].var())
        print(f"  {split:5s} 个股收益方差 {var_total:.5f} → 去市场后残差方差 "
              f"{var_resid:.5f}  （市场解释 {1 - var_resid / var_total:5.1%}）")

    print("\n===== 2. 各特征对「市场未来收益」的 IC =====")
    print("    （若这些特征能预测市场动量，则等于免费获得 regime 信息）")
    for split in ("train", "val", "test"):
        sub = data[data.split == split]
        cells = []
        for name in FEATURES:
            ic = rank_ic(sub[name].to_numpy(), sub["mkt_fwd"].to_numpy())
            cells.append(f"{name}={ic:+.3f}" if np.isfinite(ic) else f"{name}=—")
        print(f"    {split:5s} " + "  ".join(cells))

    print("\n===== 3. 各特征对「残差收益」的 IC（去掉市场后还剩多少）=====")
    for split in ("train", "val", "test"):
        sub = data[data.split == split]
        cells = []
        for name in list(FEATURES) + ["mkt_mom"]:
            ic = rank_ic(sub[name].to_numpy(), sub["resid"].to_numpy())
            cells.append(f"{name}={ic:+.3f}" if np.isfinite(ic) else f"{name}=—")
        print(f"    {split:5s} " + "  ".join(cells))

    print("\n===== 4. regime 交互：特征 IC 在牛熊下的差异（残差收益）=====")
    for split in ("train", "test"):
        sub = data[data.split == split].copy()
        sub["regime"] = np.where(sub["mkt_mom"] > 0.05, "牛",
                                  np.where(sub["mkt_mom"] < -0.05, "熊", "震荡"))
        for regime in ("牛", "熊", "震荡"):
            part = sub[sub.regime == regime]
            if len(part) < 500:
                continue
            cells = []
            for name in ("ma20_relative", "momentum_5", "volatility_20"):
                ic = rank_ic(part[name].to_numpy(), part["resid"].to_numpy())
                cells.append(f"{name}={ic:+.3f}" if np.isfinite(ic) else f"{name}=—")
            print(f"    {split:5s} {regime:4s} (n={len(part):6d}) " + "  ".join(cells))

    print("\n===== 5. 市场动量作为单独预测因子的价值 =====")
    for split in ("train", "val", "test"):
        sub = data[data.split == split]
        ic = rank_ic(sub["mkt_mom"].to_numpy(), sub["fwd"].to_numpy())
        ic_res = rank_ic(sub["mkt_mom"].to_numpy(), sub["resid"].to_numpy())
        # 分位组合：市场动量最高 20% vs 最低 20%
        q80 = sub["mkt_mom"].quantile(0.8)
        q20 = sub["mkt_mom"].quantile(0.2)
        high = sub[sub.mkt_mom >= q80]["fwd"].mean()
        low = sub[sub.mkt_mom <= q20]["fwd"].mean()
        print(f"  {split:5s} IC(市场动量→个股收益)={ic:+.4f}  "
              f"IC(市场动量→残差)={ic_res:+.4f}  "
              f"高动量组未来 {high:+.2%} vs 低动量组 {low:+.2%}  差 {high - low:+.2%}")
    print()
    print("  若「高动量组 vs 低动量组」的收益差显著且跨期稳定 → 市场动量本身")
    print("  就是当前观测里缺失的关键特征，且不需要任何外部数据源。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
