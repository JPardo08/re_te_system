"""G05 prepares P0 relation and triplet tables. It does not score Hohfeld Gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cesvima_common import fail, write_json
from re_te_system.controlled_experiment.gemma4 import (
    GemmaGoldAccessRefused,
    evaluation_table,
    refuse_gold_reference,
    score_levels,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G05 Gemma evaluation. Scientific Gold scoring stays closed.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args(argv)
    try:
        refuse_gold_reference(args.gold)
        refuse_gold_reference(args.predictions)
    except GemmaGoldAccessRefused as exc:
        fail(str(exc), code=2)
    gold = _jsonl(args.gold)
    predictions = _jsonl(args.predictions)
    levels = score_levels(gold, predictions)
    row = evaluation_table(
        "google/gemma-4-12B-it",
        args.condition,
        levels,
        estimand="conservative",
        completeness="not_a_scientific_campaign",
        coverage={"documents": len({item.get("document_id") for item in predictions})},
    )
    write_json(
        args.output,
        {
            "contrast": {
                "published": False,
                "reason": "a single condition file cannot form the C1-C0 contrast",
            },
            "experimental_f1": False,
            "row": row,
            "stage": "G05",
        },
    )
    print("PASS G05 synthetic-or-local-files-only", flush=True)
    return 0


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
