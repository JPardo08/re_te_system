"""Deterministic campaign files and coverage counts.

Coverage uses the cohort size as its denominator. That number is not an F1
denominator. Synthetic evaluation, when requested, calls ``re_te_evaluation``
and labels the score as a contract test, not as an experimental F1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from re_te_system.controlled_experiment.contracts import (
    CAMPAIGN_CONTRACT_VERSION,
    PRINCIPAL_STATES,
    canonical_json,
)
from re_te_system.controlled_experiment.dependencies import ensure_package


def coverage_report(
    cohort: tuple[Any, ...],
    conditions: tuple[str, ...],
    states: list[Mapping[str, Any]],
) -> dict[str, Any]:
    denominator = len(cohort)
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        rows = [row for row in states if row["condition"] == condition]
        if len(rows) != denominator:
            raise RuntimeError("each condition must report every cohort document")
        if {row["document_id"] for row in rows} != {item.document_id for item in cohort}:
            raise RuntimeError("conditions do not share the cohort")
        counts = {name: 0 for name in PRINCIPAL_STATES}
        diagnostics: dict[str, int] = {}
        for row in rows:
            counts[row["principal_state"]] += 1
            for name in row.get("diagnostics", {}):
                diagnostics[name] = diagnostics.get(name, 0) + 1
        if sum(counts.values()) != denominator:
            raise RuntimeError("principal states do not partition the cohort")
        by_condition[condition] = {
            "denominator": denominator,
            "diagnostics": diagnostics,
            "not_attempted": counts["not_attempted"],
            "principal_states": counts,
            "attempted": denominator - counts["not_attempted"],
        }
    return {
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "cohort_document_ids": [item.document_id for item in cohort],
        "conditions": by_condition,
        "denominator_meaning": "cohort_size_not_f1_denominator",
        "same_cohort_across_conditions": True,
    }


def evaluation_summary(
    gold: tuple[Mapping[str, Any], ...] | None,
    exported_rows: list[Mapping[str, Any]],
    contract_abort: bool,
    prediction_records: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score each condition separately.

    ``evaluation_predictions.jsonl`` is the pooled ``export_evaluation`` file.
    Rows from different conditions share ``document_id`` and do not carry the
    condition, so that file is not a condition contrast.
    """
    base = {
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "experimental_f1": False,
        "gold_source": "synthetic_fixture" if gold is not None else "not_requested",
        "pooled_prediction_rows": len(exported_rows),
        "pooled_export_is_not_a_condition_contrast": True,
        "result_role": "synthetic_contract_test_not_experimental_f1",
    }
    if contract_abort:
        return {
            **base,
            "micro": None,
            "publication": "withheld_evaluation_contract_abort",
        }
    if gold is None:
        return {**base, "micro": None, "publication": "not_requested"}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in prediction_records or []:
        grouped.setdefault(str(record["condition"]), []).append(record)
    if not grouped:
        grouped = {"unseparated": []}
        scored = {"unseparated": exported_rows}
    else:
        scored = {condition: _rows_like_export(records) for condition, records in grouped.items()}
    by_condition = {}
    for condition, rows in scored.items():
        by_condition[condition] = _score(gold, rows)
        if by_condition[condition].get("publication") == "withheld_evaluation_contract_abort":
            return {
                **base,
                "by_condition": by_condition,
                "micro": None,
                "publication": "withheld_evaluation_contract_abort",
                "reason": by_condition[condition].get("reason"),
            }
    result = {
        **base,
        "by_condition": by_condition,
        "publication": "synthetic_contract_only",
    }
    if len(by_condition) == 1:
        only = next(iter(by_condition.values()))
        result["counts"] = only["counts"]
        result["micro"] = only["micro"]
        result["prediction_rows"] = only["prediction_rows"]
    return result


def _score(gold: tuple[Mapping[str, Any], ...], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ensure_package("re_te_evaluation", "re_te_evaluation")
    from re_te_evaluation import (  # noqa: WPS433
        ContextMode,
        EligibilityPolicy,
        EvaluationConfig,
        MetricFamily,
        evaluate_triplets,
    )
    from re_te_evaluation.eligibility import InputContractError

    config = EvaluationConfig(
        MetricFamily.TRIPLET,
        ContextMode.DOCUMENT,
        EligibilityPolicy.STRICT_INLINE,
    )
    try:
        report = evaluate_triplets(gold, rows, config)
    except InputContractError as exc:
        return {
            "counts": None,
            "micro": None,
            "prediction_rows": len(rows),
            "publication": "withheld_evaluation_contract_abort",
            "reason": str(exc),
        }
    return {
        "counts": report.summary["counts"],
        "micro": report.summary["micro"],
        "prediction_rows": len(rows),
        "publication": "synthetic_contract_only",
    }


def _rows_like_export(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for index, triple in enumerate(record["validated"]["triples"]):
            identifier = f'{record["run_id"]}:{record["example_id"]}:{index:06d}'
            rows.append(
                {
                    "document_id": record["document_id"],
                    "id": identifier,
                    "object": triple["object"],
                    "prediction_id": identifier,
                    "relation": triple["relation"],
                    "segment_id": triple.get("segment_id"),
                    "subject": triple["subject"],
                }
            )
    return rows


def write_campaign_files(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _text(output_dir / "campaign_manifest.json", result["manifest"])
    _jsonl(output_dir / "attempts.jsonl", result["attempts"])
    _jsonl(output_dir / "document_states.jsonl", result["document_states"])
    _jsonl(output_dir / "predictions.jsonl", result["prediction_records"])
    _text(output_dir / "coverage.json", result["coverage"])


def _text(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
