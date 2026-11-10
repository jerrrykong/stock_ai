"""只读复核：data/raw 与 data/raw_hfq 是否仍与验收基线逐位一致。

不写任何文件、不修改基线。models/unified 会随新 run 增长，故不在受保护范围。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "corporate_action_protected_inputs_final.json"
DIRECTORIES = ("data/raw", "data/raw_hfq")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for content in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(content)
    return digest.hexdigest()


def main() -> int:
    baseline = json.loads(BASE.read_text(encoding="utf-8"))
    # 基线里的路径用 Windows 反斜杠存储，比较时统一成 POSIX 形式。
    def norm(value: str) -> str:
        return str(Path(value)).replace("\\", "/")

    expected = {norm(item["path"]): item for item in baseline["files"]
                if norm(item["path"]).startswith(DIRECTORIES)}
    observed = {str(p.relative_to(ROOT)).replace("\\", "/"): p for d in DIRECTORIES
                for p in (ROOT / d).rglob("*") if p.is_file()}
    missing = sorted(set(expected) - set(observed))
    added = sorted(set(observed) - set(expected))
    changed = []
    for rel in sorted(set(expected) & set(observed)):
        path, prev = observed[rel], expected[rel]
        stat = path.stat()
        if sha(path) != prev["sha256"] or stat.st_mtime_ns != prev["mtime_ns"]:
            changed.append(rel)
    print(f"基线文件: {BASE.name}（复核于 {baseline.get('generated_at')}，"
          f"原始基线 {baseline.get('baseline_recorded_at')}）")
    print(f"受保护范围: {', '.join(DIRECTORIES)}")
    print(f"  基线文件数 : {len(expected)}")
    print(f"  当前文件数 : {len(observed)}")
    print(f"  缺失       : {len(missing)}")
    print(f"  新增       : {len(added)}")
    print(f"  变更       : {len(changed)}")
    for rel in (missing + added + changed)[:10]:
        print(f"    - {rel}")
    ok = not (missing or added or changed)
    print(f"结论: {'通过 —— 受保护输入逐位未变' if ok else '失败 —— 受保护输入被改动'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
