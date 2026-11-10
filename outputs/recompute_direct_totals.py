"""独立重算 direct 版全量汇总，与 coverage_report.json 交叉比对。

不读报告里的 totals，只从 review/*.json 与 ledger/*.json 重新数一遍。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else Path(__file__).resolve().parent / "corporate_action_direct")
_SOURCES = {"sina_red_share_listing", "arrival_equals_ex_date"}


def main() -> int:
    reviews = sorted((OUT / "review").glob("*.json"))
    ledgers = {p.stem: p for p in (OUT / "ledger").glob("*.json")}

    status = Counter()
    included_total = 0
    excluded_total = 0
    reasons = Counter()
    merged = 0
    conflicted = 0
    listing_sources = Counter()
    mismatched_events = []
    empty_drafts = []
    bad_listing = []
    for path in reviews:
        review = json.loads(path.read_text(encoding="utf-8"))
        symbol = review["symbol"]
        status[review.get("status", "?")] += 1
        included = review.get("included", [])
        excluded = review.get("excluded", [])
        included_total += len(included)
        excluded_total += len(excluded)
        for item in excluded:
            reasons[item["reason"]] += 1
        for entry in included:
            if entry.get("merged_rows", 1) > 1:
                merged += 1
            if entry.get("conflicts"):
                conflicted += 1
            event = entry["event"]
            listing_sources[event.get("listing_date_source")] += 1
            if event.get("listing_date") and event.get("listing_date_source") not in _SOURCES:
                bad_listing.append(symbol)
            # 上市日口径 C：只允许新浪上市日，或到账日==除权日
            if event.get("listing_date_source") == "arrival_equals_ex_date" \
                    and event.get("listing_date") != event.get("ex_date"):
                bad_listing.append(f"{symbol}!arrival!=ex")
        if review.get("status") == "draft_generated" and not included:
            empty_drafts.append(symbol)
        # 账本事件必须与 review 逐字一致；被阻断的股票按设计不产出账本。
        ledger_path = ledgers.get(symbol)
        if ledger_path is None:
            if included and review.get("status") == "draft_generated":
                mismatched_events.append(f"{symbol}: 未阻断却有事件但无账本")
            continue
        if review.get("status") == "blocked_capital_change":
            mismatched_events.append(f"{symbol}: 已阻断却写了账本")
        document = json.loads(ledger_path.read_text(encoding="utf-8"))
        if [e["event"] for e in included] != document["events"]:
            mismatched_events.append(f"{symbol}: 账本与review不一致")
        if document["coverage"]["status"] != "pending_review":
            mismatched_events.append(f"{symbol}: coverage.status 不是 pending_review")

    recomputed = {
        "symbols": len(reviews),
        "draft_generated": status["draft_generated"],
        "blocked_capital_change": status["blocked_capital_change"],
        "fetch_failed": status["fetch_failed"],
        "error": status["error"],
        "included_total": included_total,
        "excluded_total": excluded_total,
        "merged_events_total": merged,
        "conflicted_events_total": conflicted,
    }
    reported = json.loads((OUT / "coverage_report.json").read_text(encoding="utf-8"))["totals"]

    print("独立重算:", json.dumps(recomputed, ensure_ascii=False))
    print("报告声明:", json.dumps(reported, ensure_ascii=False))
    diff = {k: (recomputed[k], reported[k]) for k in reported if recomputed.get(k) != reported[k]}
    print("差异:", json.dumps(diff, ensure_ascii=False) or "无")

    reported_reasons = json.loads(
        (OUT / "coverage_report.json").read_text(encoding="utf-8"))["excluded_reason_totals"]
    print("排除原因重算:", json.dumps(dict(reasons), ensure_ascii=False))
    print("排除原因报告:", json.dumps(reported_reasons, ensure_ascii=False))
    reason_diff = {k: (reasons.get(k, 0), v)
                   for k, v in reported_reasons.items() if reasons.get(k, 0) != v}
    print("原因差异:", json.dumps(reason_diff, ensure_ascii=False) or "无")

    print("\n上市日来源分布:", json.dumps(dict(listing_sources), ensure_ascii=False))
    print("上市日口径违规:", bad_listing or "无")
    print("有事件但状态为 draft 的空账本:", empty_drafts or "无")
    print("账本/review 不一致:", mismatched_events or "无")

    report = json.loads((OUT / "coverage_report.json").read_text(encoding="utf-8"))
    blocked = [s["symbol"] for s in report["symbols"] if s["status"] == "blocked_capital_change"]
    failed = [s["symbol"] for s in report["symbols"] if s["status"] == "fetch_failed"]
    print(f"\nblocked_capital_change({len(blocked)}): {blocked}")
    print(f"fetch_failed({len(failed)}): {failed}")
    for symbol in failed:
        entry = next(s for s in report["symbols"] if s["symbol"] == symbol)
        print(f"  {symbol}: {entry.get('errors')}")
    kinds = Counter()
    for symbol in blocked:
        entry = next(s for s in report["symbols"] if s["symbol"] == symbol)
        for alert in entry.get("capital_change_alerts", []):
            kinds[alert["kind"]] += 1
    print("阻断类型分布:", dict(kinds))
    ok = not diff and not reason_diff and not bad_listing and not mismatched_events
    print("\n交叉比对:", "一致" if ok else "不一致")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
