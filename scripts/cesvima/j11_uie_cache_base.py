"""J11: cache the pinned UIE base checkpoint. This does not fine-tune it."""

from __future__ import annotations

import os
from pathlib import Path

from hf_snapshot import cache_or_download


UIE_REPO = "luyaojie/uie-base-en"
UIE_REVISION = "966f8b1fc4c74e94ab552081605913ad5133cc41"


def main() -> int:
    pointer = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "base_snapshot.json"
    cache_or_download(UIE_REPO, UIE_REVISION, pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
