"""J19: download the GenIE tokenizer repository. This is not a weight file."""

from __future__ import annotations

import os
from pathlib import Path

from hf_snapshot import cache_or_download


TOKENIZER_REPO = "martinjosifoski/genie-rw"
TOKENIZER_REVISION = "81eb2ccca714fbb62a65283b65c9c153469ee409"


def main() -> int:
    cache_or_download(
        TOKENIZER_REPO,
        TOKENIZER_REVISION,
        Path(os.environ["P3_RUNS_ROOT"]) / "genie" / "tokenizer_snapshot.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
