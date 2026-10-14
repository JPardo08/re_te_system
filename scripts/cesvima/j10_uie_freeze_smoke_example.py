"""J10: freeze one positive CoNLL04 test example before any model output exists."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from cesvima_common import fail, sha256_file, write_json


def main() -> int:
    pointer = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "shot_dir.txt"
    if not pointer.is_file():
        fail(f"J09 shot pointer is missing: {pointer}")
    shot_dir = Path(pointer.read_text(encoding="utf-8").strip())
    if shot_dir.parts[-2:] != ("seed1", "10shot"):
        fail(f"shot pointer is not seed1/10shot: {shot_dir}")
    train_texts = set()
    for line in (shot_dir / "train.json").read_text(encoding="utf-8").splitlines():
        if line.strip():
            train_texts.add(json.loads(line)["text"])
    chosen = None
    for index, line in enumerate((shot_dir / "test.json").read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["text"] not in train_texts:
            chosen = (index, line, row)
            break
    if chosen is None:
        fail("every test text also appears in the 10-shot train split")
    index, line, row = chosen
    relations = [item["type"] for item in row.get("relation", [])]
    payload = {
        "document_id": f"conll04-test-{index}",
        "gold_relation_types": relations,
        "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
        "selection_rule": "first test json line whose text is absent from seed1/10shot train",
        "shot_train_sha256": sha256_file(shot_dir / "train.json"),
        "source_text": row["text"],
        "test_index": index,
        "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
    }
    output = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "smoke_example.json"
    write_json(output, payload)
    print(f"PASS J10 froze {payload['document_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
