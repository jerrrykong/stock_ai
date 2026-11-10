"""诊断 alpha 潜力：现有 9 个特征到底有没有预测力、能不能支撑择时。

三组分析：
  A. 时序 IC：单个股票自身特征与未来 5/20/60 日收益的 Spearman rank IC（择时能力）
  B. 横截面 IC：同一交易日 197 只股票横排，特征能否区分谁涨谁跌（选股能力）
  C. 分布漂移：test 段特征在 train 归一化器下的 z 值分布（是否 OOD）

只用特征缓存（data/raw_hfq_v2_train），不触碰受保护输入。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_ai.scale.unified import FEATURES  # noqa: E402

FEATURE_DIR = ROOT / "data" / "raw_hfq_v2_train"
SPLITS = {"train": (2006, 2021), "val": (2022, 2023), "test": (2024, 2025)}
HORIZONS = (5, 20, 60)


def build_panel() -> pd.DataFrame:
    """拼出全部股票的 date / close / 9 特征 / 未来收益 的长表。"""
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
    # 未来收益（标签，仅用于分析，绝不进特征）
    for h in HORIZONS:
        panel[f"fwd_{h}"] = (panel.groupby("symbol")["close"].shift(-h)
                             / panel["close"] - 1.0)
    panel["year"] = panel["date"].dt.year
    return panel


def split_of(year: int) -> str:
    for name, (start, end) in SPLITS.items():
        if start <= year <= end:
            return name
    return "out"


def timeseries_ic(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """每个（股票, 年）内部算 IC（时序预测力）。

    按年切分是为了让 IC 能落到 train/val/test 三段上；单个股票-年的样本
    约 240 个交易日，足够估一个秩相关。
    """
    rows = []
    for (symbol, year), group in panel.groupby(["symbol", "year"], sort=False):
        group = group[group[f"fwd_{horizon}"].notna()]
        if len(group) < 60:
            continue
        ranked = group[list(FEATURES)].rank(pct=True)
        target = group[f"fwd_{horizon}"].rank(pct=True)
        for name in FEATURES:
            rows.append({"feature": name, "year": int(year),
                         "ic": float(ranked[name].corr(target))})
    return pd.DataFrame(rows)


def crosssection_ic(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """同一天跨股票的 IC（选股能力）。"""
    rows = []
    sub = panel[panel[f"fwd_{horizon}"].notna()]
    for date, group in sub.groupby("date", sort=False):
        if len(group) < 20:
            continue
        ranked = group[list(FEATURES)].rank(pct=True)
        target = group[f"fwd_{horizon}"].rank(pct=True)
        year = int(group["year"].iloc[0])
        for name in FEATURES:
            rows.append({"feature": name, "year": year,
                         "ic": float(ranked[name].corr(target))})
    return pd.DataFrame(rows)


def summarize(ic: pd.DataFrame) -> pd.DataFrame:
    ic = ic.copy()
    ic["split"] = ic["year"].map(split_of)
    rows = []
    for (feature, name), group in ic.groupby(["feature", "split"]):
        values = group["ic"].dropna()
        if len(values) < 30:
            continue
        mean_ic = float(values.mean())
        std_ic = float(values.std(ddof=1))
        rows.append({"feature": feature, "split": name, "n": len(values),
                     "ic_mean": mean_ic, "ic_std": std_ic,
                     "icir": mean_ic / std_ic if std_ic > 1e-12 else 0.0,
                     "hit_rate": float((values > 0).mean())})
    return pd.DataFrame(rows)


def print_ic_table(ic: pd.DataFrame, title: str) -> None:
    print(f"===== {title} =====")
    for horizon, frame in ic.groupby("horizon"):
        sub = summarize(frame)
        sub = sub[sub.split.isin(SPLITS)]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="feature", columns="split", values="ic_mean")
        order = [s for s in ("train", "val", "test") if s in pivot.columns]
        pivot = pivot[order]
        print(f"\n  未来 {horizon} 日收益的 IC 均值：")
        print("    " + f"{'特征':<20s}" + "".join(f"{s:>10s}" for s in order))
        for feature, row in pivot.iterrows():
            print(f"    {feature:<20s}" + "".join(f"{row[s]:>+10.4f}" for s in order))
    print()


def main() -> int:
    print("加载 197 只股票的特征面板 ...")
    panel = build_panel()
    print(f"  行数 {len(panel):,} | 股票 {panel['symbol'].nunique()} | "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")
    print()

    print("计算时序 IC ...")
    ts_frames = []
    for horizon in HORIZONS:
        frame = timeseries_ic(panel, horizon)
        frame["horizon"] = horizon
        ts_frames.append(frame)
    print("计算横截面 IC ...")
    cs_frames = []
    for horizon in HORIZONS:
        frame = crosssection_ic(panel, horizon)
        frame["horizon"] = horizon
        cs_frames.append(frame)

    print_ic_table(pd.concat(ts_frames, ignore_index=True),
                   "A. 时序 IC（单股自身 → 择时能力）")
    print_ic_table(pd.concat(cs_frames, ignore_index=True),
                   "B. 横截面 IC（同日跨股 → 选股能力）")

    # 特征可用性结论：train 段最强特征在 test 段是否还成立
    for scope, frames, title in (
            ("时序", ts_frames, "C1. 时序 IC 的跨期稳定性（20 日）"),
            ("横截面", cs_frames, "C2. 横截面 IC 的跨期稳定性（20 日）")):
        frame = next(f for f in frames if f["horizon"].iloc[0] == 20)
        table = summarize(frame)
        train = table[table.split == "train"].set_index("feature")["ic_mean"]
        test = table[table.split == "test"].set_index("feature")["ic_mean"]
        joined = pd.DataFrame({"train_ic": train, "test_ic": test}).dropna()
        joined = joined.reindex(joined["train_ic"].abs().sort_values(ascending=False).index)
        print(f"===== {title} =====")
        print("    " + f"{'特征':<20s}{'train IC':>10s}{'test IC':>10s}{'保持过半':>10s}")
        for feature, row in joined.iterrows():
            keep = "是" if (abs(row["train_ic"]) > 1e-4
                            and abs(row["test_ic"]) >= 0.5 * abs(row["train_ic"])) else "否"
            print(f"    {feature:<20s}{row['train_ic']:>+10.4f}{row['test_ic']:>+10.4f}{keep:>10s}")
        print()

    # 分布漂移：train 归一化器下的 z 值
    print("===== D. 分布漂移（train 段均值/标准差归一化后） =====")
    normalizer = json.loads((ROOT / "models" / "unified" / "20261020_105633_578f25aa"
                             / "normalizer.json").read_text(encoding="utf-8"))
    mean = np.asarray(normalizer["mean"], dtype="float64")
    std = np.asarray(normalizer["std"], dtype="float64")
    print(f"    {'特征':<20s}" + "".join(f"{s:>12s}" for s in ("train |z|>3", "val |z|>3", "test |z|>3"))
          + f"{'test |z|>10':>12s}")
    for name, split in (("train", (2006, 2021)), ("val", (2022, 2023)), ("test", (2024, 2025))):
        sub = panel[(panel.year >= split[0]) & (panel.year <= split[1])]
        z = np.abs((sub[list(FEATURES)].to_numpy() - mean) / std)
        if name == "train":
            buckets = {"train": z}
        else:
            buckets[name] = z
    for i, feature in enumerate(FEATURES):
        cells = ""
        for name in ("train", "val", "test"):
            z = buckets[name][:, i]
            cells += f"{float((z > 3).mean()):>12.2%}"
        zt = buckets["test"][:, i]
        print(f"    {feature:<20s}{cells}{float((np.abs(zt) > 10).mean()):>12.2%}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
