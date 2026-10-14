"""J06: native English REBEL smoke is not runnable until a fixture is frozen."""

from __future__ import annotations

import os
from pathlib import Path

from cesvima_common import write_json


REASON = (
    "No versioned official English REBEL source document exists under "
    "re_te_system tests, fixtures, or docs. The parser tests cover model-output "
    "strings, and the mock pipeline uses a Spanish sentence. The Hohfeld Spanish "
    "benchmark is not a native REBEL smoke and is not used here."
)


def main() -> int:
    payload = {
        "documents_used": [],
        "reason": REASON,
        "scientific_stratum": "NATIVE_REPRODUCTION",
        "status": "BLOCKED_FIXTURE",
    }
    destination = Path(os.environ["P3_RUNS_ROOT"]) / "rebel" / "native_smoke_status.json"
    write_json(destination, payload)
    print("FAIL: J06 BLOCKED_FIXTURE", flush=True)
    print(REASON, flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
