"""Synthetic end-to-end smoke test for the TSV adapter.

The strings, documents, and reference triples are invented. This module does
not read Hohfeld Gold and does not modify the evaluation package. Temporary
files stay in the pytest temporary directory.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import sys
from pathlib import Path

from re_te_system.contracts import triple_to_dict
from re_te_system.normalization.basic import normalize_triples
from re_te_system.parsing.tsv import parse_tsv
from re_te_system.runner.run import export_evaluation
from re_te_system.validation.structural import validate_structural


EVAL_SRC = Path(__file__).resolve().parents[2] / "re_te_evaluation" / "src"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))

from re_te_evaluation import (  # noqa: E402
    ContextMode,
    EligibilityPolicy,
    EvaluationConfig,
    MetricFamily,
    evaluate_triplets,
)


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        MetricFamily.TRIPLET,
        ContextMode.DOCUMENT,
        EligibilityPolicy.ALL,
    )


def _gold(identifier: str, subject: str, relation: str, object_: str) -> dict[str, str]:
    return {
        "id": identifier,
        "document_id": "doc-alpha",
        "subject": subject,
        "relation": relation,
        "object": object_,
    }


def _record(raw: str, *, run_id: str = "run-synthetic") -> dict:
    parsed = parse_tsv(raw, "seg-alpha")
    normalized = normalize_triples(parsed.triples)
    validated = validate_structural(normalized, parsed.issues)
    return {
        "document_id": "doc-alpha",
        "example_id": "ex-alpha",
        "run_id": run_id,
        "raw": {
            "generation_metadata": {"do_sample": False, "output_reached_limit": False},
            "model_output": raw,
        },
        "parsed": [triple_to_dict(triple) for triple in parsed.triples],
        "normalized": [triple_to_dict(triple) for triple in normalized],
        "validated": {
            "status": validated.status,
            "triples": [triple_to_dict(triple) for triple in validated.triples],
            "violations": [asdict(item) for item in validated.violations],
        },
    }


def _export(tmp_path: Path, raw: str) -> tuple[dict, list[dict]]:
    record = _record(raw)
    source = tmp_path / "predictions.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    exported = export_evaluation(source, tmp_path / "evaluation.jsonl")
    rows = [
        json.loads(line)
        for line in exported.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return record, rows


def _counts(gold: list[dict], predictions: list[dict]) -> tuple[int, int, int]:
    report = evaluate_triplets(gold, predictions, _config())
    values = report.summary["counts"]
    assert report.summary["matching"] == "exact"
    assert report.summary["config"]["matching_policy"] == "exact_multiset_v1"
    return values["tp"], values["fp"], values["fn"]


def test_positive_exact_match_and_normalization(tmp_path: Path) -> None:
    raw = "  North   Plant  \tsupplies\tCafe\u0301 widget\n"
    record, rows = _export(tmp_path, raw)
    assert record["raw"]["model_output"] == raw
    assert rows[0]["subject"] == "North Plant"
    assert rows[0]["relation"] == "supplies"
    assert rows[0]["object"] == "Caf\u00e9 widget"
    assert record["parsed"][0]["subject"] == "  North   Plant  "
    gold = [_gold("g1", "North Plant", "supplies", "Caf\u00e9 widget")]
    assert _counts(gold, rows) == (1, 0, 0)


def test_duplicate_predictions_keep_multiplicity(tmp_path: Path) -> None:
    raw = "Acme\tsupplies\tWidget\nAcme\tsupplies\tWidget\n"
    _record_body, rows = _export(tmp_path, raw)
    assert len(rows) == 2
    assert rows[0]["relation"] == rows[1]["relation"] == "supplies"
    gold = [_gold("g1", "Acme", "supplies", "Widget")]
    assert _counts(gold, rows) == (1, 1, 0)


def test_additional_prediction_is_a_false_positive(tmp_path: Path) -> None:
    raw = "Acme\tsupplies\tWidget\nAcme\towns\tDock\n"
    _record_body, rows = _export(tmp_path, raw)
    gold = [_gold("g1", "Acme", "supplies", "Widget")]
    assert _counts(gold, rows) == (1, 1, 0)


def test_missing_prediction_is_a_false_negative(tmp_path: Path) -> None:
    raw = "Acme\tsupplies\tWidget\n"
    _record_body, rows = _export(tmp_path, raw)
    gold = [
        _gold("g1", "Acme", "supplies", "Widget"),
        _gold("g2", "Acme", "owns", "Dock"),
    ]
    assert _counts(gold, rows) == (1, 0, 1)


def test_open_label_stays_open(tmp_path: Path) -> None:
    raw = "Acme\topen-link\tWidget\n"
    record, rows = _export(tmp_path, raw)
    assert rows[0]["relation"] == "open-link"
    assert record["parsed"][0]["relation"] == "open-link"
    same = [_gold("g1", "Acme", "open-link", "Widget")]
    other = [_gold("g1", "Acme", "closed-label", "Widget")]
    assert _counts(same, rows) == (1, 0, 0)
    assert _counts(other, rows) == (0, 1, 1)


def test_parse_errors_are_not_exported_and_evaluation_does_not_abort(tmp_path: Path) -> None:
    raw = "Acme\tsupplies\tWidget\nAcme\t\tWidget\nnot tabs\nAcme\towns\tDock\n"
    record, rows = _export(tmp_path, raw)
    assert record["raw"]["model_output"] == raw
    assert [row["relation"] for row in rows] == ["supplies", "owns"]
    assert all(row["subject"] and row["relation"] and row["object"] for row in rows)
    assert {item["code"] for item in record["validated"]["violations"]} == {
        "PARSE_UNPARSEABLE_CHUNK"
    }
    gold = [
        _gold("g1", "Acme", "supplies", "Widget"),
        _gold("g2", "Acme", "owns", "Dock"),
    ]
    assert _counts(gold, rows) == (2, 0, 0)


def test_whitespace_only_line_does_not_abort_p0(tmp_path: Path) -> None:
    raw = "Acme\tsupplies\tWidget\nAcme\t   \tWidget\nAcme\towns\tDock\n"
    record, rows = _export(tmp_path, raw)
    assert record["raw"]["model_output"] == raw
    assert [item["relation"] for item in record["parsed"]] == ["supplies", "owns"]
    assert [row["relation"] for row in rows] == ["supplies", "owns"]
    assert all(row["subject"] and row["relation"] and row["object"] for row in rows)
    assert {
        item["code"] for item in record["validated"]["violations"]
    } == {"PARSE_UNPARSEABLE_CHUNK"}
    assert any(
        item["message"] == "Line has an empty required field"
        for item in record["validated"]["violations"]
    )
    gold = [
        _gold("g1", "Acme", "supplies", "Widget"),
        _gold("g2", "Acme", "owns", "Dock"),
    ]
    assert _counts(gold, rows) == (2, 0, 0)


def test_raw_remains_available_after_export(tmp_path: Path) -> None:
    raw = "Acme\tsupplies\tWidget\nbroken line\n"
    record, rows = _export(tmp_path, raw)
    assert record["raw"]["model_output"] == raw
    assert "broken line" in record["raw"]["model_output"]
    assert len(rows) == 1
    assert "broken" not in rows[0]["subject"]
    assert "broken" not in rows[0]["relation"]
    assert "broken" not in rows[0]["object"]
