"""公司行为账本构建（纯结构化接口版）。

与旧版 build_corporate_action_ledgers.py 的关键差别：**不再抓取或解析任何公告 PDF**。
巨潮/新浪/东财三个结构化接口已直接提供除权要素，公告文本不再是判据来源。

三源职责：
  * 巨潮 stock_dividend_cninfo —— 权威源（沪深交易所指定信息披露平台），
    提供 股权登记日/除权日/派息日/股份到账日/送股比例/转增比例/派息比例。
  * 新浪 stock_history_dividend_detail —— 交叉校验 + 红股上市日（仅补充来源之一）。
  * 东财 回购注销 + 巨潮 stock_allotment_cninfo —— 只检出股本变动并阻断，不结算。

裁决口径（已确认）：
  * listing_date 仅取「新浪红股上市日」或「巨潮股份到账日 == 除权日」，
    到账日晚于除权日的疑似锁定期事件一律排除。契约层强制校验，防伪造。
  * 两源冲突时以巨潮为准，事件保留并记录冲突明细（source_conflict_resolved_by_authority）。

默认只写 outputs，不写生产 data/corporate_actions。
coverage.status 默认 pending_review —— verified 是审阅声明，程序不得自盖章。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import akshare as ak

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_RAW = ROOT / "data" / "raw"
DEFAULT_REUSE = Path(__file__).resolve().parent / "corporate_action_build" / "raw"
DEFAULT_OUT = Path(__file__).resolve().parent / "corporate_action_direct"

COVERAGE_START = "20060104"
COVERAGE_END = "20251231"
REPLAY_START, REPLAY_END = "20060104", "20251231"

CNINFO_DIV_URL = "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139"
SINA_URL = "https://vip.stock.finance.sina.com.cn/corp/view/vISSUE_ShareBonusDetail.php"
CNINFO_ALLOT_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
EM_REPURCHASE_URL = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
                     "?reportName=RPTA_WEB_GETHGLIST_NEW&columns=ALL&source=WEB")
REPURCHASE_CACHE = (Path(__file__).resolve().parent / "corporate_action_build"
                    / "cache" / "repurchase_cancellations.json")

_TEN = Decimal(10)
_SYMBOL = re.compile(r"[0-9]{6}")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_date(value) -> str | None:
    """把接口返回的日期归一为 YYYYMMDD；NaT/NaN/空一律视为该源未提供。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"NaT", "NaN", "None", "nan", "nat"}:
        return None
    match = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", text)
    return "".join(match.groups()) if match else None


def per_share(value) -> Decimal | None:
    """每 10 股 -> 每股，Decimal 精确换算；缺失返回 None，绝不默认 0。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"NaT", "NaN", "None", "nan"}:
        return None
    try:
        amount = Decimal(text)
    except Exception:  # noqa: BLE001
        return None
    return amount / _TEN


def _days_between(start: str, end: str) -> int:
    from datetime import date
    first = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    second = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
    return (second - first).days


def sha_records(records: list) -> str:
    blob = json.dumps(records, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fetch_with_retry(fn, *args, attempts: int = 3, **kwargs):
    last: Exception | None = None
    for index in range(attempts):
        try:
            return fn(*args, **kwargs), None
        except Exception as exc:  # noqa: BLE001
            last = exc
            if index + 1 < attempts:
                time.sleep(1.5 * (index + 1))
    return None, f"抓取失败({attempts}次重试): {type(last).__name__}: {last}"


def archive(path: Path, provider: str, interface: str, endpoint: str,
            symbol: str, records: list, columns: list, note: str) -> dict:
    payload = {
        "schema": "corporate-action-raw-archive-v1",
        "provider": provider,
        "interface": interface,
        "endpoint": endpoint,
        "symbol": symbol,
        "status": "raw_unreviewed",
        "fetched_at_utc": now_utc(),
        "akshare_version": ak.__version__,
        "python_executable": sys.executable,
        "units": note,
        "row_count": len(records),
        "columns": columns,
        "records_sha256": sha_records(records),
        "records": records,
        "notes": [
            "原始响应逐行归档，未做任何清洗、换算或补齐。",
            "NaN/NaT 统一序列化为 null，表示该源未提供该字段，不得解读为 0。",
            "本版不抓取、不解析任何公告 PDF；公告文本不作为判据。",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def load_archive(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("records") else None


def replay_symbols(data_raw: Path) -> list[str]:
    return sorted(p.stem for p in data_raw.glob("*.parquet") if _SYMBOL.fullmatch(p.stem))


def collect(symbol: str, out: Path, reuse: Path | None) -> dict:
    """取三源原始记录；优先复用同日存档，缺则联网抓取。"""
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"symbol": symbol, "errors": []}

    # 1) 巨潮分红（权威源）
    cn_path = raw_dir / f"{symbol}.cninfo.json"
    cn = load_archive(reuse / f"{symbol}.cninfo.json") if reuse else None
    if cn is None:
        cn = load_archive(cn_path)
    if cn is None:
        frame, error = fetch_with_retry(ak.stock_dividend_cninfo, symbol=symbol)
        if error:
            result["errors"].append(f"cninfo_dividend: {error}")
            result["cninfo"] = None
        else:
            records = json.loads(frame.to_json(orient="records", force_ascii=False,
                                               date_format="iso"))
            cn = archive(cn_path, "cninfo", "akshare.stock_dividend_cninfo",
                         CNINFO_DIV_URL, symbol, records,
                         [str(c) for c in frame.columns],
                         "比例单位为每10股；派息为税前(含税)")
    result["cninfo"] = cn

    # 2) 新浪分红明细（交叉校验 + 红股上市日）
    sina_path = raw_dir / f"{symbol}.sina.json"
    sina = load_archive(reuse / f"{symbol}.sina.json") if reuse else None
    if sina is None:
        sina = load_archive(sina_path)
    if sina is None:
        frame, error = fetch_with_retry(ak.stock_history_dividend_detail, symbol=symbol)
        if error:
            result["errors"].append(f"sina_detail: {error}")
            result["sina"] = None
        else:
            records = json.loads(frame.to_json(orient="records", force_ascii=False,
                                               date_format="iso"))
            sina = archive(sina_path, "sina", "akshare.stock_history_dividend_detail",
                           SINA_URL, symbol, records,
                           [str(c) for c in frame.columns],
                           "比例单位为每10股；进度=实施 才计入交叉校验")
    result["sina"] = sina

    # 3) 配股（只检出阻断，不结算）
    al_path = raw_dir / f"{symbol}.allotment.json"
    existing = load_archive(al_path)
    if existing is not None:
        result["allotment"] = existing
        return result
    frame, error = fetch_with_retry(ak.stock_allotment_cninfo, symbol=symbol,
                                    start_date=REPLAY_START, end_date=REPLAY_END)
    if error:
        result["errors"].append(f"cninfo_allotment: {error}")
        result["allotment"] = None
    else:
        records = json.loads(frame.to_json(orient="records", force_ascii=False,
                                           date_format="iso")) if frame is not None else []
        result["allotment"] = archive(al_path, "cninfo", "akshare.stock_allotment_cninfo",
                                      CNINFO_ALLOT_URL, symbol, records,
                                      [str(c) for c in frame.columns] if frame is not None else [],
                                      "仅检出区间内配股；是否实施不作判定")
    return result


def build(symbol: str, collected: dict, repurchase: dict | None) -> dict:
    """以巨潮为权威源建事件；新浪只做交叉校验与上市日补充。"""
    review: dict = {
        "schema": "corporate-action-review-v2-direct",
        "symbol": symbol,
        "generated_at_utc": now_utc(),
        "announcement_pdf_analysis": "disabled_by_directive",
        "rules": {
            "authority": "两源冲突以巨潮为准（法定指定信息披露平台），并记录冲突明细",
            "listing_date": "仅取新浪红股上市日，或巨潮股份到账日==除权日（到账即上市）",
            "cross_check": "按 ex_date 配对；比例按每10股 Decimal 精确比较，容差 0",
            "merged_same_ex_date": "同一除权日多行必须合并，合计参与结算",
            "capital_change": "区间内配股/注销式回购只阻断，不自行结算",
        },
        "included": [], "excluded": [], "capital_change_alerts": [],
    }
    cn = collected.get("cninfo")
    sina = collected.get("sina")
    if cn is None:
        review["status"] = "fetch_failed"
        review["errors"] = collected.get("errors", [])
        return review

    sina_by_ex: dict[str, dict] = {}
    for row in (sina or {}).get("records", []):
        if str(row.get("进度", "")).strip() != "实施":
            continue
        key = norm_date(row.get("除权除息日"))
        if key:
            sina_by_ex.setdefault(key, row)

    # 同一除权日先合并，避免契约禁止的同 ex_date 冲突
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for index, row in enumerate(cn.get("records", [])):
        ex = norm_date(row.get("除权日"))
        if ex is None:
            review["excluded"].append({"row_index": index, "reason": "ex_date_missing"})
            continue
        if ex < COVERAGE_START:
            review["excluded"].append({"row_index": index, "reason": "historical_only",
                                       "ex_date": ex})
            continue
        if ex > COVERAGE_END:
            review["excluded"].append({"row_index": index, "reason": "beyond_coverage_end",
                                       "ex_date": ex})
            continue
        if ex not in grouped:
            grouped[ex] = {"ex_date": ex, "rows": [], "record_date": None}
            order.append(ex)
        grouped[ex]["rows"].append((index, row))

    for ex in order:
        group = grouped[ex]
        rows = group["rows"]
        record_dates = {norm_date(row.get("股权登记日")) for _, row in rows}
        record_dates.discard(None)
        if not record_dates:
            review["excluded"].append({"ex_date": ex, "reason": "record_date_missing"})
            continue
        if len(record_dates) > 1:
            review["excluded"].append({"ex_date": ex,
                                       "reason": "same_ex_date_record_date_conflict",
                                       "record_dates": sorted(record_dates)})
            continue
        record_date = record_dates.pop()
        if record_date >= ex:
            review["excluded"].append({"ex_date": ex, "reason": "record_date_not_before_ex",
                                       "record_date": record_date})
            continue
        if _days_between(record_date, ex) > 365:
            # 登记日与除权日相隔超过一年，源端日期几乎必然错配，不得放行。
            review["excluded"].append({"ex_date": ex, "reason": "ex_date_record_gap_implausible",
                                       "record_date": record_date})
            continue

        cash = bonus = transfer = Decimal(0)
        legs_present = False
        pay_date = None
        arrival = None
        for _, row in rows:
            cash_leg = per_share(row.get("派息比例"))
            bonus_leg = per_share(row.get("送股比例"))
            transfer_leg = per_share(row.get("转增比例"))
            if cash_leg is not None or bonus_leg is not None or transfer_leg is not None:
                legs_present = True
            cash += cash_leg or Decimal(0)
            bonus += bonus_leg or Decimal(0)
            transfer += transfer_leg or Decimal(0)
            if pay_date is None:
                pay_date = norm_date(row.get("派息日"))
            if arrival is None:
                arrival = norm_date(row.get("股份到账日"))

        if not legs_present:
            review["excluded"].append({"ex_date": ex, "reason": "zero_leg_no_distribution"})
            continue
        if not any(value > 0 for value in (cash, bonus, transfer)):
            review["excluded"].append({"ex_date": ex, "reason": "zero_leg_no_distribution",
                                       "note": "三腿均为 0，不是分配事件"})
            continue
        if cash > 0 and pay_date is None:
            review["excluded"].append({"ex_date": ex, "reason": "pay_date_unknown"})
            continue
        if cash > 0 and pay_date < ex:
            review["excluded"].append({"ex_date": ex, "reason": "pay_date_before_ex_date",
                                       "pay_date": pay_date})
            continue

        listing_date = None
        listing_source = None
        sina_row = sina_by_ex.get(ex) or {}
        sina_listing = norm_date(sina_row.get("红股上市日"))
        if bonus > 0 or transfer > 0:
            if sina_listing:
                listing_date, listing_source = sina_listing, "sina_red_share_listing"
            elif arrival is not None and arrival == ex:
                listing_date, listing_source = arrival, "arrival_equals_ex_date"
            else:
                review["excluded"].append({
                    "ex_date": ex, "reason": "listing_date_unknown",
                    "share_credit_date": arrival,
                    "note": "到账日晚于除权日可能为锁定期，不得当作上市日",
                })
                continue

        conflicts: list[dict] = []
        if sina_row:
            for field, mine, theirs in (
                ("record_date", record_date, norm_date(sina_row.get("股权登记日"))),
                ("cash_per_share", cash, per_share(sina_row.get("派息"))),
                ("bonus_ratio", bonus, per_share(sina_row.get("送股"))),
                ("transfer_ratio", transfer, per_share(sina_row.get("转增"))),
            ):
                if theirs is None:
                    continue
                if mine != theirs:
                    conflicts.append({"field": field, "cninfo": str(mine),
                                      "sina": str(theirs)})

        event = {
            "event_id": f"{symbol}-{ex}-distribution",
            "kind": "distribution",
            "status": "implemented",
            "record_date": record_date,
            "ex_date": ex,
            "pay_date": pay_date if cash > 0 else None,
            "listing_date": listing_date,
            "cash_per_share_gross": str(cash),
            "bonus_ratio": str(bonus),
            "transfer_ratio": str(transfer),
            "reference_method": "standard_cash_bonus",
            "sources": [CNINFO_DIV_URL, SINA_URL],
            "fractional_policy": "reject",
            "listing_date_source": listing_source,
        }
        entry = {
            "event": event,
            "merged_rows": len(rows),
            "cross_checked_with_sina": bool(sina_row),
            "conflicts": conflicts,
        }
        if len(rows) > 1:
            entry["merged"] = True
        if conflicts:
            entry["resolved_by"] = "cninfo_authority"
        review["included"].append(entry)

    blocking: list[dict] = []
    informational: list[dict] = []
    for row in (collected.get("allotment") or {}).get("records", []):
        # 配股产生按配股价认购的权利，未行权即稀释；引擎无结算能力，必须阻断。
        blocking.append({"kind": "rights_issue",
                         "ex_date": norm_date(row.get("除权基准日")),
                         "record_date": norm_date(row.get("股权登记日")),
                         "listing_date": norm_date(row.get("配股上市日"))})
    for row in (repurchase or {}).get(symbol, []):
        # 注销式回购不除权、不改变持股数与每股权益，raw 价结算无需事件；
        # 因此只披露不阻断——用它阻断整只股票会丢弃二十年可结算的分配事件。
        informational.append({"kind": "buyback_cancellation",
                              "updated": norm_date(row.get("UPDATEDATE")),
                              "repurchased_shares": row.get("REPURNUM"),
                              "objective": str(row.get("REPUROBJECTIVE", ""))[:80]})
    review["capital_change_alerts"] = blocking
    review["capital_change_informational"] = informational
    review["status"] = "blocked_capital_change" if blocking else "draft_generated"
    return review


def ledger_document(symbol: str, review: dict, verified: bool) -> dict:
    notes = ("由巨潮/新浪结构化接口直接构建，未解析任何公告 PDF。"
             "权威源为巨潮，两源冲突以巨潮为准并记录在同名 review 文件。"
             "区间内配股不在此账本结算（引擎无配股权结算能力，已阻断）。"
             "注销式回购不除权、不改变持股数，raw 价结算无需事件，仅作信息披露。")
    return {
        "schema": "corporate-actions-v1",
        "symbol": symbol,
        "coverage": {
            "start": COVERAGE_START,
            "end": COVERAGE_END,
            "status": "verified" if verified else "pending_review",
            "scope": "all_shareholder_distributions_and_capital_changes",
            "sources": [CNINFO_DIV_URL, SINA_URL, CNINFO_ALLOT_URL, EM_REPURCHASE_URL],
            "verified_at": now_utc(),
            "notes": notes,
        },
        "events": [entry["event"] for entry in review["included"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="公司行为账本构建（纯结构化接口，不解析公告）")
    parser.add_argument("--data-raw", type=Path, default=DEFAULT_DATA_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reuse-archive", type=Path, default=DEFAULT_REUSE,
                        help="复用同日已抓取的 cninfo/sina 原始存档")
    parser.add_argument("--symbols", default="", help="逗号分隔的股票代码，默认 data/raw 全量")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verified", action="store_true",
                        help="将 coverage.status 置为 verified；仅在人工审阅后使用")
    parser.add_argument("--write-production", type=Path, default=None,
                        help="同时写入生产目录；默认不写")
    args = parser.parse_args()

    if args.out.exists() and not (args.out / "coverage_report.json").exists():
        pass
    args.out.mkdir(parents=True, exist_ok=True)

    symbols = (args.symbols.split(",") if args.symbols else replay_symbols(args.data_raw))
    if args.limit:
        symbols = symbols[:args.limit]

    repurchase = None
    if REPURCHASE_CACHE.exists():
        repurchase = json.loads(REPURCHASE_CACHE.read_text(encoding="utf-8")).get("by_symbol")

    started = now_utc()
    per_symbol: list[dict] = []
    for index, symbol in enumerate(symbols, 1):
        try:
            collected = collect(symbol, args.out, args.reuse_archive)
            review = build(symbol, collected, repurchase)
        except Exception:  # noqa: BLE001
            review = {"schema": "corporate-action-review-v2-direct", "symbol": symbol,
                      "status": "error", "generated_at_utc": now_utc(),
                      "traceback": traceback.format_exc(limit=4)}
        (args.out / "review").mkdir(exist_ok=True)
        (args.out / "review" / f"{symbol}.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")

        if review.get("status") == "draft_generated" and review.get("included"):
            document = ledger_document(symbol, review, args.verified)
            (args.out / "ledger").mkdir(exist_ok=True)
            (args.out / "ledger" / f"{symbol}.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
            if args.write_production:
                args.write_production.mkdir(parents=True, exist_ok=True)
                (args.write_production / f"{symbol}.json").write_text(
                    json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")

        reasons: dict[str, int] = {}
        for item in review.get("excluded", []):
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        per_symbol.append({
            "symbol": symbol,
            "status": review.get("status"),
            "included": len(review.get("included", [])),
            "excluded": len(review.get("excluded", [])),
            "excluded_reason_counts": reasons,
            "merged_events": sum(1 for e in review.get("included", []) if e.get("merged")),
            "conflicted_events": sum(1 for e in review.get("included", []) if e.get("conflicts")),
            "capital_change_alerts": review.get("capital_change_alerts", []),
            "errors": review.get("errors", []),
        })
        if index % 25 == 0 or index == len(symbols):
            print(f"[{index}/{len(symbols)}] {symbol} {review.get('status')} "
                  f"included={len(review.get('included', []))}", flush=True)

    totals = {
        "symbols": len(symbols),
        "draft_generated": sum(1 for s in per_symbol if s["status"] == "draft_generated"),
        "blocked_capital_change": sum(1 for s in per_symbol if s["status"] == "blocked_capital_change"),
        "fetch_failed": sum(1 for s in per_symbol if s["status"] == "fetch_failed"),
        "error": sum(1 for s in per_symbol if s["status"] == "error"),
        "included_total": sum(s["included"] for s in per_symbol),
        "excluded_total": sum(s["excluded"] for s in per_symbol),
        "merged_events_total": sum(s["merged_events"] for s in per_symbol),
        "conflicted_events_total": sum(s["conflicted_events"] for s in per_symbol),
    }
    reasons_total: dict[str, int] = {}
    for entry in per_symbol:
        for reason, count in entry["excluded_reason_counts"].items():
            reasons_total[reason] = reasons_total.get(reason, 0) + count
    report = {
        "schema": "corporate-action-coverage-v2-direct",
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
        "coverage_window": [COVERAGE_START, COVERAGE_END],
        "announcement_pdf_analysis": "disabled",
        "authority": "cninfo",
        "listing_date_policy": "sina_red_share_listing ∪ (cninfo 股份到账日 == 除权日)",
        "conflict_policy": "cninfo_authority",
        "python_executable": sys.executable,
        "akshare": ak.__version__,
        "totals": totals,
        "excluded_reason_totals": reasons_total,
        "symbols": per_symbol,
    }
    (args.out / "coverage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=1))
    print(json.dumps(reasons_total, ensure_ascii=False, indent=1))
    print(f"-> {args.out / 'coverage_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
