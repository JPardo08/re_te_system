"""J05: download REBEL only after J04 has frozen a 40-character SHA."""

from __future__ import annotations

import os
from pathlib import Path

from cesvima_common import fail, read_json, require_sha
from hf_snapshot import cache_or_download


def main() -> int:
    pointer = Path(os.environ["P3_RUNS_ROOT"]) / "rebel" / "revision.json"
    if not pointer.is_file():
        fail(f"J04 revision artifact is missing: {pointer}")
    record = read_json(pointer)
    if record.get("repository") != "Babelscape/rebel-large":
        fail(f"unexpected repository in revision artifact: {record.get('repository')}")
    sha = require_sha(record.get("sha", ""), "J04 sha")
    if record.get("source") != "huggingface_hub.HfApi.model_info":
        fail("revision artifact source is not the Hugging Face model info API")
    cache_or_download(
        "Babelscape/rebel-large",
        sha,
        Path(os.environ["P3_RUNS_ROOT"]) / "rebel" / "snapshot.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
