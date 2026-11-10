"""P0-1 落地前第一道 gate：金额口径数据前提断言独立 CLI 守卫（只读）。

用法（务必使用 ``.venv/Scripts/python.exe``）：
    python outputs/verify_amount_precondition.py
        → data/raw_hfq_v2_train 三条断言全通过，exit 0
    python outputs/verify_amount_precondition.py --dir data/raw
        → 合成 amount 断言①失败，exit != 0 并打印「直接止损」

三条断言（plan §3.5 / PRD v2 §2.1⑦）：
  ① ``std(amount/(volume × close_raw)) > 1e-3``            挡合成值（std=0）
  ② ``占比(vwap=amount/volume ∈ [low_raw, high_raw]) > 0.99`` 挡口径错配
  ③ ``vwap != close_raw`` 占比显著>0、``std(vwap_dev)>0``、
     且与 ``volume_relative`` 的 ``corr`` 显著 <1     挡常数 / 共线假特征

判定口径为**目录级聚合**（PRD §2.1⑦「对实际使用的 amount 目录逐日验算」）：
逐只按 ``%Y%m%d`` 8 位日期键对齐后合并成一个总帧，``volume_relative``
逐股票 rolling（绝不串接股票）。任一断言不通过 → 该目录 amount 不可用、
该特征族不可构造、**直接止损，不进入回测**。

本脚本只读 ``data/``，不写任何文件、不落盘、不缓存、不跑回测、不训练。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "outputs"))

from feature_families import (  # noqa: E402
    PreconditionError, PreconditionReport, align_frames_by_day_key,
    compute_precondition, load_config,
)

#: 对齐后参与断言的最小列集合（其余列不入内存总帧，减少开销）
_KEEP_COLS = ["symbol", "amount", "volume", "close_raw", "high_raw", "low_raw"]


def _resolve(path_str: str) -> Path:
    """把 CLI 路径解析为绝对路径（相对路径以项目根为基准）。"""
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def collect_aligned(amount_dir: Path,
                    raw_dir: Path) -> tuple[pd.DataFrame, int, list[str]]:
    """逐只读两份 parquet、按 8 位日期键对齐，合并成目录级总帧。

    Args:
        amount_dir: 提供 ``amount`` / ``volume`` 的目录（真实或合成）。
        raw_dir: 提供未复权 ``close/high/low`` 的目录（价格分母）。

    Returns:
        ``(总帧, 成功对齐股票数, 跳过股票列表)``。
    """
    parts: list[pd.DataFrame] = []
    skipped: list[str] = []
    for hfq_file in sorted(amount_dir.glob("*.parquet")):
        raw_file = raw_dir / hfq_file.name
        if not raw_file.exists():
            skipped.append(f"{hfq_file.stem}(缺 raw)")
            continue
        frame = pd.read_parquet(hfq_file)
        raw_frame = pd.read_parquet(raw_file)
        if frame.empty or raw_frame.empty:
            skipped.append(f"{hfq_file.stem}(空帧)")
            continue
        aligned = align_frames_by_day_key(frame, raw_frame)
        aligned = aligned.assign(symbol=hfq_file.stem)
        parts.append(aligned[_KEEP_COLS].copy())
    if not parts:
        raise SystemExit(
            f"错误：{amount_dir} 与 {raw_dir} 没有可对齐的 parquet 样本；"
            "直接止损，不进入回测"
        )
    return pd.concat(parts, ignore_index=True), len(parts), skipped


def _print_report(report: PreconditionReport, amount_dir: str, raw_dir: str,
                  n_symbols: int, skipped: list[str],
                  thresholds: dict) -> None:
    """打印可直接贴进报告的三条断言实测值、逐条结论与目录口径。"""
    line = "=" * 78
    print(line)
    print("数据前提断言守卫（P0-1 落地前第一道 gate）")
    print(line)
    print(f"  amount 目录      : {amount_dir}")
    print(f"  价格分母目录     : {raw_dir}  （未复权 OHLC；禁用 close_hfq）")
    print(f"  金额分子说明     : 仅用 amount 目录的真实 amount；禁用 data/raw 合成 amount")
    print(f"  日期键           : %Y%m%d 8 位字符串（严禁与 2006-01-04 混用）")
    print(f"  成功对齐股票数   : {n_symbols} 只")
    print(f"  对齐总行数       : {report.n_rows} 行")
    if skipped:
        print(f"  跳过样本         : {len(skipped)} 只  {skipped[:5]}")
    print("-" * 78)
    print("  三条断言实测值（目录级聚合）")
    print(f"    ① std(amount/(volume×close_raw))      = "
          f"{report.std_amount_ratio:>14.8g}   "
          f"阈值 > {float(thresholds['std_min']):.4g}   "
          f"{'通过' if report.checks['①']['pass'] else '失败'}")
    inr = report.checks["②"]
    print(f"    ② 占比(vwap ∈ [low_raw, high_raw])    = "
          f"{report.vwap_in_range_ratio:>14.8g}   "
          f"阈值 > {float(thresholds['vwap_in_range_min']):.4g}   "
          f"{'通过' if inr['pass'] else '失败'}")
    print(f"       （严格口径，诊断用）               = "
          f"{report.vwap_in_range_ratio_strict:>14.8g}   "
          f"价格精度容差 {float(thresholds['vwap_range_tolerance']):.4g}")
    a = report.checks["③a"]
    b = report.checks["③b"]
    c = report.checks["③c"]
    print(f"    ③a 占比(vwap != close_raw)            = "
          f"{report.vwap_ne_close_ratio:>14.8g}   "
          f"阈值 > {float(thresholds['vwap_ne_close_ratio_min']):.4g}   "
          f"{'通过' if a['pass'] else '失败'}")
    print(f"    ③b std(vwap_dev)                      = "
          f"{report.vwap_dev_std:>14.8g}   "
          f"阈值 > {float(thresholds['vwap_dev_std_min']):.4g}   "
          f"{'通过' if b['pass'] else '失败'}")
    print(f"    ③c corr(vwap_dev, volume_relative)    = "
          f"{report.corr_with_volume_relative:>14.8g}   "
          f"阈值 |·| < {float(thresholds['collinearity_max']):.4g}   "
          f"{'通过' if c['pass'] else '失败'}")
    print("-" * 78)
    print(f"  三条断言结论     : ①{'✓' if report.checks['①']['pass'] else '✗'}  "
          f"②{'✓' if inr['pass'] else '✗'}  "
          f"③{'✓' if (a['pass'] and b['pass'] and c['pass']) else '✗'}"
          f"    passed = {report.passed}")
    print(line)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：全通过 ``exit 0``；任一失败打印「直接止损」并 ``exit 1``。"""
    parser = argparse.ArgumentParser(
        description="金额口径数据前提断言守卫（P0-1 落地前第一道 gate，只读）")
    parser.add_argument("--dir", default="data/raw_hfq_v2_train",
                        help="提供 amount 的目录（默认 data/raw_hfq_v2_train）")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="提供未复权 OHLC 价格分母的目录（默认 data/raw）")
    args = parser.parse_args(argv)

    amount_dir = _resolve(args.dir)
    raw_dir = _resolve(args.raw_dir)
    if not amount_dir.is_dir() or not raw_dir.is_dir():
        print(f"错误：目录不存在 amount={amount_dir} raw={raw_dir}")
        print("直接止损")
        return 2

    cfg = load_config()
    thresholds = cfg["assertions"]
    aligned, n_symbols, skipped = collect_aligned(amount_dir, raw_dir)
    # 目录级聚合判定：与 assert_amount_precondition 共享同一 compute_precondition
    # 实现，避免多只帧按单键 merge 造成笛卡尔积错配。
    report = compute_precondition(aligned, amount_dir=args.dir, cfg=cfg)
    _print_report(report, args.dir, args.raw_dir, n_symbols, skipped, thresholds)

    if report.passed:
        print("  结论：三条数据前提断言全部通过，该 amount 目录可用于 P0-1 金额型特征。")
        print("  exit 0")
        return 0

    # 用权威异常统一失败文案（含「直接止损」），并把实测值一并抛出
    error = PreconditionError(report)
    print(f"  触发条款         : {'、'.join(report.failed_checks)}")
    print(f"  结论：{error}")
    print("  直接止损 —— 该 amount 目录不可用、该特征族不可构造，不进入回测。")
    print("  exit != 0")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
