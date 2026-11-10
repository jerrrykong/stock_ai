"""写入生产后的复核：原 643 个受保护输入是否未变 + 生产账本是否契约可用。

只读，不修改任何文件。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stock_ai.data.corporate_actions import CorporateActionError, load_ledger  # noqa: E402

OUT = Path(__file__).resolve().parent
BASELINE = json.loads((OUT / "corporate_action_protected_inputs_final.json").read_text(encoding="utf-8"))
PRODUCTION = ROOT / "data" / "corporate_actions"
START, END = "20060104", "20251231"


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    baseline = {str(item["path"]): item for item in BASELINE["files"]}
    missing, changed, unchanged = [], [], 0
    for relative, previous in baseline.items():
        path = ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue
        if sha(path) != previous["sha256"] or path.stat().st_mtime_ns != previous["mtime_ns"]:
            changed.append(relative)
            continue
        unchanged += 1

    added = sorted(str(p.relative_to(ROOT)) for p in PRODUCTION.rglob("*")
                   if p.is_file()) if PRODUCTION.exists() else []

    print(f"[受保护输入] 基线 {len(baseline)} | 未变 {unchanged} | 被改 {len(changed)} | 丢失 {len(missing)}")
    for item in changed[:10]:
        print("  被改:", item)
    for item in missing[:10]:
        print("  丢失:", item)

    print(f"\n[生产目录] data/corporate_actions 新增文件 {len(added)}")
    ok = bad = 0
    failures: list[dict] = []
    for relative in added:
        try:
            ledger = load_ledger(ROOT / relative, symbol=Path(relative).stem)
            ledger.require_coverage(START, END)
        except CorporateActionError as exc:
            bad += 1
            failures.append({"symbol": relative, "error": str(exc)})
            continue
        ok += 1
    print(f"[生产账本] 契约加载+区间准入通过 {ok} | 失败 {bad}")
    for item in failures[:10]:
        print("  ", item)

    statuses = set()
    for relative in added:
        document = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        statuses.add(document["coverage"]["status"])
    print("[生产账本] coverage.status 取值:", statuses or "无")
    print("[结论]", "通过" if (not changed and not missing and bad == 0 and statuses == {"verified"})
          else "不通过")
    return 0 if (not changed and not missing and bad == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
