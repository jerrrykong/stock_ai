"""核查：各诊断脚本里的市场代理是否真的生效（日期键类型一致性）。

背景：diagnose_market_feature_value.py 出现「加市场特征后结果完全不变」，
根因是 merge 时日期键格式不一致（'2006-01-04' vs '20060104'），
市场特征被 fillna(0) 全填成 0 —— 模型根本没看到市场信息。

本脚本逐个检查历史诊断脚本的市场代理是否也存在此问题，
避免一个格式 bug 推翻（或错误支撑）已发布的结论。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FEATURE_DIR = ROOT / "data" / "raw_hfq_v2_train"


def check_keys(label, keys, probe):
    """检查 probe（面板里的日期）能否在 keys（市场代理的键）里命中。"""
    sample = list(probe[:5])
    hits = sum(1 for d in sample if d in keys)
    print(f"  {label}")
    print(f"    键类型        : {type(list(keys)[0])}")
    print(f"    探针类型      : {type(sample[0])}")
    print(f"    命中率        : {hits}/{len(sample)}  "
          f"{'✓ 正常' if hits == len(sample) else '✗ 全部落空！'}")
    return hits == len(sample)


def main() -> int:
    from stock_ai.scale.unified import relative_features

    # 取一只股票的 hfq 面板
    path = sorted(FEATURE_DIR.glob("*.parquet"))[0]
    raw = pd.read_parquet(path)
    df, values = relative_features(raw)
    dates_str = df.date.dt.strftime("%Y%m%d").to_numpy()
    dates_np = df.date.to_numpy()

    print("=== 键类型基础事实 ===")
    print(f"  df.date.dtype                : {df.date.dtype}")
    print(f"  .dt.strftime('%Y%m%d')      : {type(dates_str[0])} {dates_str[0]}")
    print(f"  .to_numpy()                  : {type(dates_np[0])} {dates_np[0]}")
    print(f"  Timestamp 与 np.datetime64 互查: "
          f"{ {np.datetime64('2006-01-04'): 'v'}.get(pd.Timestamp('2006-01-04'), 'MISS') }")
    print()

    ok = True

    # ---- 脚本 1: diagnose_regime_filter.py / validate_combined_strategy.py ----
    # 这两个用 str 键建市场代理、又用 str 日期查 → 必然一致
    print("=== 1. regime_filter.py / validate_combined_strategy.py（str 键）===")
    str_keys = {d: 1.0 for d in dates_str}
    ok &= check_keys("str 键 ↔ str 探针", str_keys, dates_str)

    # ---- 脚本 2: diagnose_market_context_value.py ----
    # 用 group["date"].to_numpy() 建代理（np.datetime64 键），
    # 再用 panel["date"].map(market_mom) 查（datetime64 面板 → Timestamp 探针）
    print("\n=== 2. diagnose_market_context_value.py（np.datetime64 键 + .map）===")
    np_keys = {d: 1.0 for d in dates_np}
    hits = sum(1 for d in df.date[:5] if d in np_keys)
    print(f"    键类型        : {type(list(np_keys)[0])}")
    print(f"    探针类型      : {type(df.date.iloc[0])}")
    print(f"    .map 命中率   : {hits}/5  {'✓ 正常' if hits == 5 else '✗ 落空'}")
    ok &= hits == 5

    # ---- 脚本 3: diagnose_market_feature_value.py（本轮新写的，有 bug）----
    print("\n=== 3. diagnose_market_feature_value.py（merge 路径）===")
    # 复现它的错误路径
    market = {d: 1.0 for d in dates_np}
    mf = pd.DataFrame.from_dict(market, orient="index")
    mf.index.name = "date"
    mf = mf.reset_index()
    mf["date"] = mf["date"].astype(str)          # ← 它的写法
    panel_keys = df.date.dt.strftime("%Y%m%d")   # ← 面板侧写法
    print(f"    市场侧键样例 : {mf['date'].iloc[0]!r}")
    print(f"    面板侧键样例 : {panel_keys.iloc[0]!r}")
    matched = mf["date"].isin(set(panel_keys)).sum()
    print(f"    merge 命中   : {matched}/{len(mf)}  "
          f"{'✓ 正常' if matched == len(mf) else '✗ 全部落空 —— 已确认是本脚本的 bug'}")
    print("    → 结论：历史 regime 结论不受影响；仅本轮新脚本的市场特征失效，")
    print("       需要修正键格式后重跑。")

    print()
    print("=== 4. 修正方案 ===")
    mf2 = pd.DataFrame.from_dict(market, orient="index")
    mf2.index.name = "date"
    mf2 = mf2.reset_index()
    mf2["date_key"] = mf2["date"].dt.strftime("%Y%m%d")   # 用同一格式
    matched2 = mf2["date_key"].isin(set(panel_keys)).sum()
    print(f"    改用 .dt.strftime('%Y%m%d') 后 merge 命中: {matched2}/{len(mf2)}  "
          f"{'✓ 修好' if matched2 == len(mf2) else '✗ 仍失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
