"""从已抓取的原始存档统计：停用公告 PDF 分析后，纯接口方案的字段可得性。

只读 outputs/corporate_action_build/raw/*，不联网、不写生产目录。
回答三个问题：
  Q1 巨潮分红接口本身是否已给出完整的除权要素（登记日/除权日/派息日/比例）？
  Q2 送转事件的 listing_date（上市日）在纯接口下可得多少？
  Q3 配股在回放区间内的规模？
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent / "corporate_action_build" / "raw"
CUTOFF = "20060104"  # 回放区间起点


def norm_date(value) -> str | None:
    if value in (None, "", "NaT"):
        return None
    text = str(value)
    match = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", text)
    return "".join(match.groups()) if match else None


def load(symbol: str, provider: str) -> dict | None:
    path = RAW / f"{symbol}.{provider}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    symbols = sorted({p.name.split(".")[0] for p in RAW.glob("*.json")})
    stats = {
        "symbols": len(symbols),
        "cninfo_rows": 0,
        "symbols_ok": 0,
        "symbols_failed": 0,
        "ex_date_missing": 0,
        "record_date_missing": 0,
        "pay_date_missing": 0,
        "rows_in_range": 0,
        "bonus_transfer_events": 0,
        "arrival_date_filled": 0,
        "sina_listing_filled": 0,
        "union_listing_filled": 0,
        "both_listing_missing": 0,
    }
    per_symbol = []
    for symbol in symbols:
        cn = load(symbol, "cninfo")
        sina = load(symbol, "sina")
        if cn is None or not (cn.get("records") or cn.get("row_count")):
            stats["symbols_failed"] += 1
            continue
        stats["symbols_ok"] += 1
        rows = cn.get("records", [])
        stats["cninfo_rows"] += len(rows)

        # 新浪按除权日建索引
        sina_by_ex: dict[str, dict] = {}
        for row in (sina or {}).get("records", []):
            key = norm_date(row.get("除权除息日"))
            if key:
                sina_by_ex.setdefault(key, row)

        sym_bt = sym_arrival = sym_sina = sym_union = sym_missing = 0
        for row in rows:
            ex = norm_date(row.get("除权日"))
            if not ex:
                stats["ex_date_missing"] += 1
                continue
            cash_leg = row.get("派息比例") or 0
            try:
                has_cash = float(cash_leg) > 0
            except (TypeError, ValueError):
                has_cash = False
            if has_cash and not norm_date(row.get("派息日")):
                stats["pay_date_missing"] += 1
            if not norm_date(row.get("股权登记日")):
                stats["record_date_missing"] += 1
            if ex < CUTOFF:
                continue
            stats["rows_in_range"] += 1

            bonus = row.get("送股比例") or 0
            transfer = row.get("转增比例") or 0
            try:
                is_bt = float(bonus) > 0 or float(transfer) > 0
            except (TypeError, ValueError):
                is_bt = False
            if not is_bt:
                continue
            stats["bonus_transfer_events"] += 1
            sym_bt += 1
            arrival = norm_date(row.get("股份到账日"))
            sina_row = sina_by_ex.get(ex) or {}
            sina_listing = norm_date(sina_row.get("红股上市日"))
            if arrival:
                stats["arrival_date_filled"] += 1
                sym_arrival += 1
            if sina_listing:
                stats["sina_listing_filled"] += 1
                sym_sina += 1
            if arrival or sina_listing:
                stats["union_listing_filled"] += 1
                sym_union += 1
            else:
                stats["both_listing_missing"] += 1
                sym_missing += 1
        per_symbol.append({
            "symbol": symbol, "cninfo_rows": len(rows),
            "bonus_transfer": sym_bt, "arrival": sym_arrival,
            "sina_listing": sym_sina, "listing_ok": sym_union,
            "listing_missing": sym_missing,
        })

    out = Path(__file__).resolve().parent / "cninfo_direct_coverage.json"
    out.write_text(json.dumps({"summary": stats, "per_symbol": per_symbol},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 纯接口方案可得性（基于已抓取原始存档，只读）===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    bt = stats["bonus_transfer_events"] or 1
    print(f"\n  listing_date 可得率(区间内送转事件): {stats['union_listing_filled']}/{bt} "
          f"= {stats['union_listing_filled'] / bt:.1%}")
    print(f"  其中 巨潮股份到账日: {stats['arrival_date_filled']}, 新浪红股上市日: {stats['sina_listing_filled']}")
    print(f"  完全缺失: {stats['both_listing_missing']}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
