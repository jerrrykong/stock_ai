"""Read-only comparison of the exact raw/hfq/unified acceptance baseline."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DIRECTORIES = ("data/raw", "data/raw_hfq", "models/unified")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for content in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(content)
    return digest.hexdigest()


def main() -> int:
    baseline_path = OUT / "corporate_action_protected_inputs_before.json"
    target = OUT / "corporate_action_protected_inputs_final.json"
    if target.exists():
        raise FileExistsError("Protected-input evidence already exists; never overwrite it")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected = {str(Path(item["path"])): item for item in baseline["files"]}
    observed = {str(path.relative_to(ROOT)): path
                for directory in DIRECTORIES
                for path in (ROOT / directory).rglob("*") if path.is_file()}
    missing = sorted(set(expected) - set(observed))
    added = sorted(set(observed) - set(expected))
    changes, checked, errors = [], [], []
    for relative in sorted(set(expected) & set(observed)):
        path, previous = observed[relative], expected[relative]
        try:
            stat_before = path.stat()
            fingerprint = sha(path)
            stat_after = path.stat()
            changed_fields = []
            if fingerprint != previous["sha256"]:
                changed_fields.append("sha256")
            if stat_after.st_mtime_ns != previous["mtime_ns"]:
                changed_fields.append("mtime_ns")
            if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
                changed_fields.append("changed_during_read")
            record = {"path": relative, "sha256": fingerprint, "mtime_ns": stat_after.st_mtime_ns}
            checked.append(record)
            if changed_fields:
                changes.append({**record, "changed_fields": changed_fields})
        except OSError as exc:
            errors.append({"path": relative, "error": str(exc)})
    production_dir = ROOT / "data/corporate_actions"
    production_ledgers = sorted(str(p.relative_to(ROOT)) for p in production_dir.glob("*.json"))
    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "baseline_recorded_at": baseline["recorded_at"],
        "baseline_file": str(baseline_path),
        "scope": list(DIRECTORIES),
        "read_only": True,
        "baseline_count": len(expected),
        "checked_count": len(checked),
        "counts_by_directory": dict(Counter("/".join(Path(p).parts[:2]) for p in expected)),
        "missing": missing,
        "added": added,
        "changed": changes,
        "errors": errors,
        "production_corporate_action_ledgers": production_ledgers,
        "passed": not (missing or added or changes or errors),
        "files": checked,
    }
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
