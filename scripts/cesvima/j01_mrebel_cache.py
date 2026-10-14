"""J01: reuse or download the pinned mREBEL snapshot into HF_HOME on SCRATCH."""

from __future__ import annotations

import os
from pathlib import Path

from hf_snapshot import cache_or_download


MREBEL_REPO = "Babelscape/mrebel-large"
MREBEL_REVISION = "50e0587ac7ac87d28b9abd069d72333528a5aa09"


def main() -> int:
    pointer = Path(os.environ["P3_RUNS_ROOT"]) / "mrebel" / "snapshot.json"
    cache_or_download(MREBEL_REPO, MREBEL_REVISION, pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
