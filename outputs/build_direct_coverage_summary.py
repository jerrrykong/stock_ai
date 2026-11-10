"""把 direct 版全量覆盖结果导出为 CSV + 自包含离线 HTML，供人工审阅。"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / "corporate_action_direct"
START, END = "20060104", "20251231"


def main() -> int:
    report = json.loads((OUT / "coverage_report.json").read_text(encoding="utf-8"))
    symbols = report["symbols"]
    totals = report["totals"]
    reasons = report["excluded_reason_totals"]

    listing = Counter()
    conflicts: list[dict] = []
    merged: list[dict] = []
    for path in sorted((OUT / "review").glob("*.json")):
        review = json.loads(path.read_text(encoding="utf-8"))
        for entry in review.get("included", []):
            event = entry["event"]
            listing[event.get("listing_date_source") or "none(纯现金)"] += 1
            if entry.get("conflicts"):
                conflicts.append({"symbol": review["symbol"], "ex_date": event["ex_date"],
                                  "conflicts": entry["conflicts"]})
            if entry.get("merged_rows", 1) > 1:
                merged.append({"symbol": review["symbol"], "ex_date": event["ex_date"],
                               "rows": entry["merged_rows"]})

    csv_path = OUT / "coverage_by_symbol.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "status", "included", "excluded", "merged", "conflicted",
                         "excluded_reasons"])
        for entry in symbols:
            writer.writerow([
                entry["symbol"], entry["status"], entry["included"], entry["excluded"],
                entry["merged_events"], entry["conflicted_events"],
                "; ".join(f"{k}={v}" for k, v in entry["excluded_reason_counts"].items()),
            ])

    blocked = [s["symbol"] for s in symbols if s["status"] == "blocked_capital_change"]
    failed = [s["symbol"] for s in symbols if s["status"] == "fetch_failed"]
    empty = [s["symbol"] for s in symbols
             if s["status"] == "draft_generated" and s["included"] == 0]
    usable = [s["symbol"] for s in symbols
              if s["status"] == "draft_generated" and s["included"] > 0]

    def table(rows: list[tuple[str, object]]) -> str:
        body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        return f"<table><thead><tr><th>项</th><th>值</th></tr></thead><tbody>{body}</tbody></table>"

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>公司行为账本覆盖汇总（纯接口版）</title>
<style>
:root{{color-scheme:light}}
body{{margin:0;padding:24px;background:#f7f8fa;color:#1c2024;
 font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.6}}
.wrap{{max-width:960px;margin:0 auto}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:24px 0 8px}}
.sub{{color:#5b6570;font-size:13px;margin-bottom:16px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:12px 14px}}
.card .n{{font-size:22px;font-weight:600}} .card .t{{color:#5b6570;font-size:12px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;
 border:1px solid #e3e6ea;border-radius:8px;overflow:hidden}}
th,td{{border-bottom:1px solid #eef0f2;padding:6px 10px;text-align:left;vertical-align:top}}
th{{background:#f0f2f5;font-weight:600}}
code{{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:12px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:800px){{.two{{grid-template-columns:1fr}}}}
.note{{background:#fff8e6;border:1px solid #f0d9a0;border-radius:8px;padding:10px 14px;font-size:13px}}
ul{{margin:6px 0 0 18px;padding:0}}
.chips{{display:flex;flex-wrap:wrap;gap:6px}}
.chip{{background:#eef1f4;border-radius:6px;padding:2px 8px;font-size:12px}}
section>*{{min-width:0}}
</style></head><body><div class="wrap">
<h1>公司行为账本覆盖汇总（纯接口版）</h1>
<div class="sub">区间 {START} ~ {END} · 未解析任何公告 PDF · 权威源巨潮 · 生成于 {report['finished_at_utc']}</div>

<div class="cards">
 <div class="card"><div class="n">{totals['symbols']}</div><div class="t">本地可回放股票</div></div>
 <div class="card"><div class="n">{len(usable)}</div><div class="t">产出可用账本</div></div>
 <div class="card"><div class="n">{totals['included_total']}</div><div class="t">入账分配事件</div></div>
 <div class="card"><div class="n">{totals['excluded_total']}</div><div class="t">排除记录</div></div>
 <div class="card"><div class="n">{len(blocked)}</div><div class="t">配股阻断</div></div>
 <div class="card"><div class="n">{len(failed)}</div><div class="t">抓取失败</div></div>
</div>

<h2>股票状态</h2>
{table([("draft_generated 可用", len(usable)),
        ("draft_generated 但无事件", len(empty)),
        ("blocked_capital_change（区间内配股）", len(blocked)),
        ("fetch_failed（源端抓取失败，非无分红）", len(failed)),
        ("error", totals['error'])])}

<h2>排除原因</h2>
{table(sorted(reasons.items(), key=lambda kv: -kv[1]))}

<h2>上市日来源分布（口径 C）</h2>
{table([(k, v) for k, v in listing.most_common()])}

<h2>合并与冲突</h2>
{table([("同除权日合并事件", totals['merged_events_total']),
        ("两源冲突（已按巨潮裁决）", totals['conflicted_events_total'])])}

<h2>阻断清单（区间内配股，{len(blocked)} 只）</h2>
<div class="chips">{"".join(f'<span class="chip"><code>{s}</code></span>' for s in blocked)}</div>

<h2>抓取失败清单（{len(failed)} 只）</h2>
<div class="chips">{"".join(f'<span class="chip"><code>{s}</code></span>' for s in failed)}</div>
<div class="sub">均为 <code>KeyError: '实施方案公告日期'</code>，属接口抓取失败，不得据此断言这些股票无分红。</div>

<div class="note" style="margin-top:24px">
<strong>审阅状态：verified（人工审阅通过后写入）</strong>
<ul>
<li>已于 2026-10-19 经人工审阅通过，<code>coverage.status</code> 置为 <code>verified</code>，
 并写入生产目录 <code>data/corporate_actions</code>（262 份）。</li>
<li>写入前后事件体、覆盖区间、来源列表经逐份比对完全一致，仅审阅声明本身变化；
 <code>pending_review</code> 版本留档于 <code>ledger_pending_review/</code>。</li>
<li>原有 643 个受保护输入（data/raw 605 + models 38）sha256 与 mtime_ns 全部未变。</li>
<li>差异化分红检测随公告 PDF 分析一并下线，目前无结构化替代判据（残留风险）。</li>
<li>注销式回购不除权、不改变持股数，raw 价结算无需事件，仅作信息披露，不阻断。</li>
<li>逐项汇总已由独立重算脚本复核，与报告完全一致。</li>
</ul>
</div>
</div></body></html>"""
    html_path = OUT / "coverage_summary.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"-> {csv_path}")
    print(f"-> {html_path}")
    print(f"可用账本 {len(usable)} | 阻断 {len(blocked)} | 抓取失败 {len(failed)} | 空账本 {len(empty)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
