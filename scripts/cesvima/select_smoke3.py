"""Write a three-document smoke JSONL from the canonical benchmark."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from cesvima_common import fail, require_under_scratch, write_json


SMOKE_IDS = ("articulo_61", "articulo_8", "articulo_12")


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: select_smoke3.py OUTPUT.jsonl")
    documents = Path(os.environ["CANONICAL_DOCUMENTS"])
    if not documents.is_file():
        fail(f"canonical documents missing: {documents}")
    found: dict[str, str] = {}
    total = 0
    for line in documents.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        document_id = row.get("document_id")
        text = row.get("source_text")
        if document_id in SMOKE_IDS:
            if not isinstance(text, str) or not text:
                fail(f"{document_id} has no source_text")
            found[document_id] = text
    if total != 36:
        fail(f"canonical documents.jsonl has {total} records; expected 36")
    missing = [item for item in SMOKE_IDS if item not in found]
    if missing:
        fail(f"smoke3 ids missing from canonical documents: {missing}")
    output = require_under_scratch(Path(sys.argv[1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"document_id": document_id, "source_text": found[document_id]},
            ensure_ascii=False,
            sort_keys=True,
        )
        for document_id in SMOKE_IDS
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output.with_suffix(".meta.json"),
        {"document_ids": list(SMOKE_IDS), "records": 3, "source": str(documents)},
    )
    print(f"PASS smoke3 {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
