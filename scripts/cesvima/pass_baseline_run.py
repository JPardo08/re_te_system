"""Check a finished baseline run against the frozen technical PASS rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cesvima_common import fail, require_sha


def main() -> int:
    if len(sys.argv) != 4:
        fail("usage: pass_baseline_run.py RUN_DIR EXPECTED_COUNT EXPECTED_REVISION")
    run_dir = Path(sys.argv[1])
    expected_count = int(sys.argv[2])
    expected_revision = require_sha(sys.argv[3])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    statistics = json.loads((run_dir / "statistics.json").read_text(encoding="utf-8"))
    resolved = manifest["model"]["resolved_revision"]
    if resolved != expected_revision:
        fail(f"resolved revision {resolved} != {expected_revision}")
    relation_filter = manifest.get("windowing", {}).get("relation_filter")
    if relation_filter != "none":
        fail(f"relation filter is {relation_filter}; legacy filter is forbidden")
    if statistics.get("inputs_succeeded") != expected_count:
        fail(f"inputs_succeeded {statistics.get('inputs_succeeded')} != {expected_count}")
    if statistics.get("inputs_failed") not in (0, None):
        fail(f"inputs_failed is {statistics.get('inputs_failed')}")
    for name in ("manifest.json", "predictions.jsonl", "failures.jsonl", "statistics.json"):
        if not (run_dir / name).is_file():
            fail(f"missing artifact {name}")
    print(f"PASS baseline run {run_dir} n={expected_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
