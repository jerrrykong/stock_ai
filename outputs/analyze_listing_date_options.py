"""量化 listing_date 三种口径下的可得性，供决策。

现行契约 src/stock_ai/data/corporate_actions.py:164 明确禁止用「股份到账日」替代上市日。
本脚本只读原始存档，不联网，不给结论，只给数字。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent / "corporate_action_build" / "raw"
CUTOFF = "20060104"


def norm_date(value) -> str | None:
    if value in (None, "", "NaT", "NaN"):
        return None
    m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", str(value))
    return "".join(m.groups()) if m else None


def main() -> int:
    symbols = sorted({p.name.split(".")[0] for p in RAW.glob("*.cninfo.json")})
    bt_total = 0
    a_sina_only = 0          # A: 仅新浪红股上市日（现行契约）
    b_arrival_or_sina = 0    # B: 到账日 或 新浪（放宽契约）
    c_arrival_eq_ex = 0      # C: 到账日==除权日 或 新浪（无锁定期争议子集）
    arrival_lt_ex = 0
    arrival_gt_ex = 0
    arrival_eq_ex_count = 0
    for symbol in symbols:
        cn = json.loads((RAW / f"{symbol}.cninfo.json").read_text(encoding="utf-8"))
        sina_path = RAW / f"{symbol}.sina.json"
        sina_by_ex: dict[str, dict] = {}
        if sina_path.exists():
            for row in json.loads(sina_path.read_text(encoding="utf-8")).get("records", []):
                key = norm_date(row.get("除权除息日"))
                if key:
                    sina_by_ex.setdefault(key, row)
        for row in cn.get("records", []):
            ex = norm_date(row.get("除权日"))
            if not ex or ex < CUTOFF:
                continue
            vals = []
            for name in ("送股比例", "转增比例"):
                try:
                    vals.append(float(row.get(name) or 0))
                except (TypeError, ValueError):
                    vals.append(0.0)
            if not any(v > 0 for v in vals):
                continue
            bt_total += 1
            arrival = norm_date(row.get("股份到账日"))
            sina_listing = norm_date((sina_by_ex.get(ex) or {}).get("红股上市日"))
            if sina_listing:
                a_sina_only += 1
            if arrival or sina_listing:
                b_arrival_or_sina += 1
            if arrival and ex:
                if arrival == ex:
                    arrival_eq_ex_count += 1
                elif arrival < ex:
                    arrival_lt_ex += 1
                else:
                    arrival_gt_ex += 1
            if sina_listing or (arrival and arrival == ex):
                c_arrival_eq_ex += 1
    print(f"区间内送转事件总数: {bt_total}")
    print(f"  A 仅新浪红股上市日(现行契约)      : {a_sina_only} ({a_sina_only / bt_total:.1%})")
    print(f"  B 到账日或新浪(放宽契约)          : {b_arrival_or_sina} ({b_arrival_or_sina / bt_total:.1%})")
    print(f"  C 到账日==除权日 或 新浪(无争议)  : {c_arrival_eq_ex} ({c_arrival_eq_ex / bt_total:.1%})")
    print(f"  其中 到账日==除权日: {arrival_eq_ex_count}, 到账日<除权日(违约): {arrival_lt_ex}, "
          f"到账日>除权日(疑似锁定期): {arrival_gt_ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
