#!/usr/bin/env python3
"""Mechanical validation for the frozen UIE CoNLL04 reproduction artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


REQUIRED_FIELDS = {
    "text",
    "tokens",
    "record",
    "entity",
    "relation",
    "event",
    "spot",
    "asoc",
    "spot_asoc",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        missing = REQUIRED_FIELDS - set(value)
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
        rows.append(value)
    return rows


def text_hash(rows: list[dict]) -> str:
    content = "\n".join(row["text"] for row in rows)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_split(path: Path, parser) -> tuple[list[dict], dict]:
    rows = load_jsonl(path)
    _, counter = parser.decode(
        gold_list=[],
        pred_list=[row["record"] for row in rows],
        text_list=[row["text"] for row in rows],
    )
    unexpected_repairs = {
        key: value
        for key, value in counter.items()
        if key in {"fixed", "ill-formed"} and value
    }
    if unexpected_repairs:
        raise ValueError(f"{path}: SEL parser repairs detected: {unexpected_repairs}")
    identifiers = [row["id"] for row in rows if "id" in row]
    duplicate_ids = len(identifiers) - len(set(identifiers))
    if duplicate_ids:
        raise ValueError(f"{path}: duplicate IDs: {duplicate_ids}")
    texts = [row["text"] for row in rows]
    return rows, {
        "duplicate_ids": duplicate_ids,
        "duplicate_text_occurrences": len(texts) - len(set(texts)),
        "ids_present": len(identifiers),
        "parser_counter": dict(sorted(counter.items())),
        "rows": len(rows),
        "sha256": sha256(path),
        "text_sha256": text_hash(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uie-repo", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--shot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.uie_repo))
    from uie.extraction.predict_parser import get_predict_parser
    from uie.extraction.record_schema import RecordSchema

    record_schema = RecordSchema.read_from_file(args.full_dir / "record.schema")
    sel_parser = get_predict_parser("spotasoc", record_schema)

    report: dict = {
        "full": {},
        "leakage": {},
        "schema": {},
        "shot": {},
        "status": "validated",
        "warnings": [],
    }
    full_rows: dict[str, list[dict]] = {}
    shot_rows: dict[str, list[dict]] = {}
    for split, filename in (
        ("train", "train.json"),
        ("validation", "val.json"),
        ("test", "test.json"),
    ):
        full_rows[split], report["full"][split] = validate_split(
            args.full_dir / filename,
            sel_parser,
        )
        shot_rows[split], report["shot"][split] = validate_split(
            args.shot_dir / filename,
            sel_parser,
        )

    for filename in ("record.schema", "entity.schema", "relation.schema", "event.schema"):
        schema = RecordSchema.read_from_file(args.shot_dir / filename)
        report["schema"][filename] = {
            "mapping": schema.type_role_dict,
            "roles": schema.role_list,
            "sha256": sha256(args.shot_dir / filename),
            "types": schema.type_list,
        }

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        report["leakage"][f"full_{left}_{right}"] = len(
            {row["text"] for row in full_rows[left]}
            & {row["text"] for row in full_rows[right]}
        )
    report["leakage"]["shot_train_validation"] = len(
        {row["text"] for row in shot_rows["train"]}
        & {row["text"] for row in shot_rows["validation"]}
    )
    report["leakage"]["shot_train_test"] = len(
        {row["text"] for row in shot_rows["train"]}
        & {row["text"] for row in shot_rows["test"]}
    )
    if any(report["leakage"].values()):
        report["status"] = "validated_with_warnings"
        report["warnings"].append(
            {
                "code": "UPSTREAM_TEXT_OVERLAP_ACROSS_SPLITS",
                "details": report["leakage"],
                "policy": "record_without_manual_correction",
            }
        )

    report["shot"]["train"]["association_counts"] = dict(
        sorted(Counter(label for row in shot_rows["train"] for label in row["asoc"]).items())
    )
    report["shot"]["train"]["spot_counts"] = dict(
        sorted(Counter(label for row in shot_rows["train"] for label in row["spot"]).items())
    )
    report["combined_hash"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
