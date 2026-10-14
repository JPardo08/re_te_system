"""J04: resolve Babelscape/rebel-large to an immutable commit SHA."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from cesvima_common import fail, require_sha, write_json


REPO = "Babelscape/rebel-large"


def main() -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        fail(f"huggingface_hub is not importable: {exc}")
    info = HfApi().model_info(REPO)
    sha = require_sha(getattr(info, "sha", None) or "", REPO)
    payload = {
        "repository": REPO,
        "resolved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha": sha,
        "source": "huggingface_hub.HfApi.model_info",
        "source_url": f"https://huggingface.co/api/models/{REPO}",
    }
    pointer = Path(os.environ["P3_RUNS_ROOT"]) / "rebel" / "revision.json"
    write_json(pointer, payload)
    print(f"PASS J04 {REPO}@{sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
