"""决定性检验：IC 稳定但毛超额为 0 —— 到底差在哪一环？

观测到的矛盾：
  - ma20_relative 的 20 日时序 IC 跨期稳定：−0.109 / −0.117 / −0.112
  - 但择时迷你回测里毛超额 ≈ 0（price logreg test: −0.19%）
  - 联合量、regime 门控都没有改善

可能的解释（本脚本逐个量化）：
  E1 IC 是**样本级**统计（个股自身时间序列上的秩相关），但择时需要的是
     「在给定日期的横截面上，特征能否区分个股」——两者不是一回事。
     换句话说：IC 衡量的是"过去 20 天涨得多的股票未来 20 天相对弱"
     （**时间序列上的反转**），而不是"今天哪些股票明天会涨"。
  E2 择时的困难在于**决策噪声**：单一信号 + 固定阈值 + 摩擦，
     毛边际 0.2% 级、摩擦 0.3%/次 → 结构性不可行。
  E3 真正的可交易 alpha 需要**横截面**信息（谁比谁强），
     而这需要多股票同期观测 —— 当前架构（单股 episode）根本拿不到。

关键量化：对每个 (日期, 股票) 样本，把「未来 20 日收益」拆成
  市场部分 + 个股残差，
然后算：
  ① 特征对「个股绝对收益」的时序 IC（已知：−0.11，稳定）
  ② 特征对「个股相对市场残差收益」的时序 IC
  ③ 若 ② 仍然稳定 → 说明信号是「个股相对反转」，在单股择时里用不上，
     但在组合/横截面架构里能用 → 支持方案 C
  ④ 若 ② 也不稳定 → 信号本身脆弱，方案 C 也救不了 → 需要换特征族
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


def market_forward(panel, horizon=20):
    """等权市场组合的未来 horizon 日收益，按日期对齐。"""
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
        j = i + horizon
        out[date] = (values[j] / values[i] - 1.0) if j < len(values) else np.nan
    return out


def rank_ic(x, y):
    if len(x) < 30:
        return np.nan
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> int:
    print("加载面板 ...")
    panel = build_panel()
    mkt_fwd = market_forward(panel)
    panel["mkt_fwd"] = panel["date"].map(mkt_fwd)
    data = panel[panel["fwd"].notna() & panel["mkt_fwd"].notna()].copy()
    data = data[data["fwd"].abs() < 1.0]
    data["resid"] = data["fwd"] - data["mkt_fwd"]
    print(f"  有效样本 {len(data):,}")
    print()

    print("===== ① vs ② vs ③：信号到底作用于「绝对收益」还是「相对市场」=====")
    print(f"    {'特征':20s}{'目标':12s}{'train':>10s}{'val':>10s}{'test':>10s}")
    for name in ("ma20_relative", "momentum_5", "volatility_20", "low_relative"):
        for target, label in (("fwd", "绝对收益"), ("resid", "相对市场残差")):
            ics = []
            for split in ("train", "val", "test"):
                a, b = SPLITS[split]
                sub = data[(data.year >= a) & (data.year <= b)]
                ics.append(rank_ic(sub[name].to_numpy(), sub[target].to_numpy()))
            print(f"    {name:20s}{label:12s}{ics[0]:>+10.4f}{ics[1]:>+10.4f}{ics[2]:>+10.4f}")
        print()

    print("===== ④ 时序 IC 的语义分解 =====")
    print("    拆解：IC(特征, 绝对收益) = IC(特征, 市场收益) + IC(特征, 残差收益) 的近似分解")
    print("    若特征主要在预测「市场方向」→ 单股择时无法用（因为它得的是 beta）")
    print("    若特征主要在预测「个股相对强弱」→ 需要横截面架构（方案 C）")
    print()
    for name in ("ma20_relative", "momentum_5", "volatility_20"):
        for split in ("train", "test"):
            a, b = SPLITS[split]
            sub = data[(data.year >= a) & (data.year <= b)]
            ic_abs = rank_ic(sub[name].to_numpy(), sub["fwd"].to_numpy())
            ic_mkt = rank_ic(sub[name].to_numpy(), sub["mkt_fwd"].to_numpy())
            ic_res = rank_ic(sub[name].to_numpy(), sub["resid"].to_numpy())
            print(f"    {name:18s} {split:5s}: 绝对 {ic_abs:+.4f}  "
                  f"市场 {ic_mkt:+.4f}  残差 {ic_res:+.4f}  "
                  f"→ {'残差主导' if abs(ic_res) > abs(ic_mkt) else '市场主导'}")

    print("\n===== ⑤ 横截面视角：同一天，特征能否区分「谁未来 20 日更强」=====")
    print("    （这正是组合/选股架构需要的能力，也是方案 C 的前提）")
    for split in ("train", "val", "test"):
        a, b = SPLITS[split]
        sub = data[(data.year >= a) & (data.year <= b)]
        ics = []
        for name in ("ma20_relative", "momentum_5", "volatility_20", "low_relative"):
            values = []
            for date, group in sub.groupby("date"):
                if len(group) < 20:
                    continue
                ic = rank_ic(group[name].to_numpy(), group["fwd"].to_numpy())
                if np.isfinite(ic):
                    values.append(ic)
            ics.append((name, float(np.mean(values)) if values else np.nan))
        cells = "  ".join(f"{n}={v:+.4f}" for n, v in ics)
        print(f"    {split:5s}: {cells}")

    print("\n===== ⑥ 摩擦门槛测算 =====")
    print("    若单次择时的毛边际为 m、往返摩擦 0.3%，则年换手上限 ≈ m / 0.003")
    for m in (0.001, 0.002, 0.005, 0.01):
        print(f"    毛边际 {m:.1%} → 允许年换手上限 {m / 0.003:.1f} 倍")
    print("    对照：v3 PPO 实际年换手 17 倍 → 需要毛边际 ≥ 5.1% 才可能打平")
    print("    而 §1.9 实测毛超额在 test 上是 −0.19% ~ −4.83%（全为负）")
    print()
    print("    → 结论：当前特征族在单股择时架构下，摩擦门槛高到无法跨越。")
    print("      即使信号方向判断完全正确，也换不回钱。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
