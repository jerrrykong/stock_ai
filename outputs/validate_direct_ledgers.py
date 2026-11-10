"""校验 direct 版账本能否真正通过契约加载与区间准入。

只读 outputs 下的产物，不写生产目录，不联网。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stock_ai.data.corporate_actions import CorporateActionError, load_ledger  # noqa: E402

START, END = "20060104", "20251231"


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1
               else Path(__file__).resolve().parent / "corporate_action_direct")
    ledger_dir = out / "ledger"
    report = json.loads((out / "coverage_report.json").read_text(encoding="utf-8"))
    files = sorted(ledger_dir.glob("*.json")) if ledger_dir.exists() else []
    if not files:
        print("没有账本文件")
        return 1
    # 结构校验：临时把 status 置为 verified 只为了检验事件体是否满足契约，
    # 这不是审阅声明，产物本身仍保持 pending_review。
    probe_dir = out / "_structure_probe"
    probe_dir.mkdir(exist_ok=True)
    ok = bad = 0
    failures: list[dict] = []
    for path in files:
        symbol = path.stem
        document = json.loads(path.read_text(encoding="utf-8"))
        document["coverage"]["status"] = "verified"
        probe = probe_dir / f"{symbol}.json"
        probe.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        try:
            ledger = load_ledger(probe, symbol=symbol)
            ledger.require_coverage(START, END)
        except CorporateActionError as exc:
            bad += 1
            failures.append({"symbol": symbol, "error": str(exc)})
            continue
        ok += 1
    print(f"账本文件 {len(files)} | 结构校验通过 {ok} | 失败 {bad}")
    for item in failures[:10]:
        print(f"  {item['symbol']}: {item['error']}")
    totals = report["totals"]
    print("覆盖报告 totals:", json.dumps(totals, ensure_ascii=False))
    print("排除原因:", json.dumps(report["excluded_reason_totals"], ensure_ascii=False))
    print(f"coverage.status: "
          f"{json.loads(files[0].read_text(encoding='utf-8'))['coverage']['status'] if files else 'n/a'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
