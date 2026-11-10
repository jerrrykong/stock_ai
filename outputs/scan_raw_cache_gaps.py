"""扫描 v2 训练子集的行情缓存缺口。

两类缺口：
  A) 相对「全市场交易日并集」参考日历的缺失日（缓存不完整）
  B) 账本事件日期在行情缓存中无对应日线（v2 环境会直接拒绝构建）

只读，不修改任何缓存。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "raw_hfq_v2_train"
RAW = ROOT / "data" / "raw"
LEDGERS = ROOT / "data" / "corporate_actions"


def main() -> int:
    symbols = sorted(p.stem for p in TRAIN.glob("*.parquet"))
    # 参考日历：全部 300 只 raw 缓存的交易日并集
    calendar: set[str] = set()
    frames = {}
    for path in sorted(RAW.glob("*.parquet")):
        days = pd.read_parquet(path, columns=["date"]).date.dt.strftime("%Y%m%d")
        calendar.update(days.tolist())
    print(f"参考日历（全市场交易日并集）: {len(calendar)} 天 "
          f"{min(calendar)} ~ {max(calendar)}")

    gap_symbols, event_missing = [], []
    for symbol in symbols:
        raw = pd.read_parquet(RAW / f"{symbol}.parquet")
        days = set(raw.date.dt.strftime("%Y%m%d").tolist())
        missing = calendar - days
        if missing:
            gap_symbols.append((symbol, len(missing), min(missing), max(missing)))

        ledger_path = LEDGERS / f"{symbol}.json"
        if not ledger_path.exists():
            continue
        document = json.loads(ledger_path.read_text(encoding="utf-8"))
        bad = []
        for event in document["events"]:
            for field in ("record_date", "ex_date", "pay_date", "listing_date"):
                value = event.get(field)
                if value and value not in days:
                    bad.append((event["event_id"], field, value))
        if bad:
            event_missing.append((symbol, bad))

    print(f"\n[A] 缓存缺日股票: {len(gap_symbols)} / {len(symbols)}")
    for symbol, count, first, last in gap_symbols[:20]:
        print(f"   {symbol}: 缺 {count} 天（{first} ~ {last}）")

    print(f"\n[B] 事件日期无日线的股票: {len(event_missing)}")
    for symbol, bad in event_missing[:20]:
        print(f"   {symbol}: {bad[:4]}{' ...' if len(bad) > 4 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
