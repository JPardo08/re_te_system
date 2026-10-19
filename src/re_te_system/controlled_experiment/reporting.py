"""Coverage, campaign status, and the three protocol estimands.

P0 scores are read from per-condition files written by ``export_evaluation``.
This module does not rebuild those rows. A pooled file is not a contrast.
Synthetic results keep ``experimental_f1`` false.

The protocol's conditional estimand includes every generation whose principal
state is ``completed``. A truncated input is not a valid observation of the
original conditioning. When that flag is present this module withholds the
conditional and paired numbers and records the conflict. It does not change
the protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from re_te_system.controlled_experiment.contracts import (
    CAMPAIGN_CONTRACT_VERSION,
    PRINCIPAL_STATES,
    STATUS_ABORTED_CONFIGURATION,
    STATUS_ABORTED_CONTRACT,
    STATUS_INCOMPLETE,
    STATUS_PUBLICATION_BLOCKED,
    STATUS_PUBLISHABLE,
)
from re_te_system.controlled_experiment.dependencies import ensure_package
from re_te_system.controlled_experiment.persistence import atomic_write_json, atomic_write_jsonl


INPUT_TRUNCATION_CONFLICT = (
    "A completed generation with truncated_input is not a valid observation of "
    "the original conditioning. The protocol's conditional estimand includes "
    "every completed generation. This harness does not rewrite that definition. "
    "It withholds the conditional and paired scores while the flag is present."
)
WITHHELD_PUBLICATIONS = frozenset(
    {
        STATUS_INCOMPLETE,
        STATUS_ABORTED_CONTRACT,
        STATUS_ABORTED_CONFIGURATION,
    }
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a complete JSONL file. A missing final newline is not valid."""
    text = path.read_text(encoding="utf-8")
    if text == "":
        return []
    if not text.endswith("\n"):
        raise OSError(f"{path} does not end with a newline, so it is not a complete JSONL file")
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line == "":
            raise OSError(f"{path}:{line_number} is blank")
        rows.append(json.loads(line))
    return rows


def contract_violation(rows: list[Mapping[str, Any]]) -> str | None:
    for index, row in enumerate(rows):
        for field_name in ("subject", "relation", "object"):
            value = row.get(field_name)
            if not isinstance(value, str) or value.strip() == "":
                return f"exported row {index} has an empty {field_name}"
    return None


def coverage_report(
    cohort: tuple[Any, ...],
    conditions: tuple[str, ...],
    states: list[Mapping[str, Any]],
    attempts: list[Mapping[str, Any]] | None = None,
    exported_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    denominator = len(cohort)
    expected_ids = {item.document_id for item in cohort}
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        rows = [row for row in states if row["condition"] == condition]
        if len(rows) != denominator:
            raise RuntimeError("each condition must report every cohort document")
        if {row["document_id"] for row in rows} != expected_ids:
            raise RuntimeError("conditions do not share the cohort")
        counts = {name: 0 for name in PRINCIPAL_STATES}
        diagnostics: dict[str, int] = {}
        harness: dict[str, int] = {}
        also_observed = 0
        parsed_triples = 0
        for row in rows:
            incident = row.get("harness_incident")
            if incident:
                harness[str(incident)] = harness.get(str(incident), 0) + 1
            else:
                counts[str(row["principal_state"])] += 1
            for name in row.get("diagnostics", {}):
                diagnostics[name] = diagnostics.get(name, 0) + 1
            also_observed += len(row.get("also_observed") or [])
            parsed_triples += int(row.get("parsed_triples") or 0)
        if sum(counts.values()) + sum(harness.values()) != denominator:
            raise RuntimeError("principal states and harness incidents do not cover the cohort")
        invalid_lines = 0
        for attempt in attempts or []:
            if attempt.get("condition") != condition:
                continue
            for issue in attempt.get("parse_issues") or []:
                if issue.get("code") == "UNPARSEABLE_CHUNK":
                    invalid_lines += 1
        not_reached = harness.get("not_reached_after_interruption", 0)
        not_started = harness.get("not_started_configuration_abort", 0)
        by_condition[condition] = {
            "also_observed": also_observed,
            "attempted": denominator - counts["not_attempted"] - not_reached - not_started,
            "completed": counts["completed"],
            "contract_aborts": diagnostics.get("evaluation_contract_abort", 0),
            "denominator": denominator,
            "diagnostics": diagnostics,
            "empty_generations": diagnostics.get("completed_empty", 0),
            "fully_malformed": diagnostics.get("fully_malformed", 0),
            "harness_incidents": harness,
            "invalid_lines": invalid_lines,
            "not_attempted": counts["not_attempted"],
            "not_reached": not_reached,
            "not_started": not_started,
            "parsed_triples": parsed_triples,
            "partially_malformed": diagnostics.get("partially_malformed", 0),
            "principal_states": counts,
            "retry_history": _retry_history(rows),
            "triples_exported_to_p0": None if exported_counts is None else exported_counts.get(condition, 0),
            "truncated_inputs": diagnostics.get("truncated_input", 0),
            "truncated_outputs": diagnostics.get("output_truncated", 0),
        }
    return {
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "cohort_document_ids": [item.document_id for item in cohort],
        "conditions": by_condition,
        "denominator_meaning": "cohort_size_not_f1_denominator",
        "pairs": {
            "attempted": sum(item["attempted"] for item in by_condition.values()),
            "completed": sum(item["completed"] for item in by_condition.values()),
            "expected": denominator * len(conditions),
            "not_attempted": sum(item["not_attempted"] for item in by_condition.values()),
            "not_reached": sum(item["not_reached"] for item in by_condition.values()),
        },
        "same_cohort_across_conditions": True,
    }


def decide_campaign_status(states: list[Mapping[str, Any]], *, contract_abort: bool) -> dict[str, Any]:
    incidents = [str(row["harness_incident"]) for row in states if row.get("harness_incident")]
    finals = [str(row["principal_state"]) for row in states if row.get("principal_state")]
    truncated = any((row.get("diagnostics") or {}).get("truncated_input") is True for row in states)
    not_attempted = "not_attempted" in finals
    blockers: list[str] = []
    if "not_started_configuration_abort" in incidents:
        status = STATUS_ABORTED_CONFIGURATION
        blockers.append("configuration_not_assessed")
    elif any(state in {"external_infrastructure_error", "timeout"} for state in finals) or any(
        code != "not_started_configuration_abort" for code in incidents
    ):
        status = STATUS_INCOMPLETE
        if any(state in {"external_infrastructure_error", "timeout"} for state in finals):
            blockers.append("unresolved_external_failure")
        if "not_reached_after_interruption" in incidents:
            blockers.append("interrupted")
        for code in incidents:
            if code != "not_reached_after_interruption":
                blockers.append(code)
    elif contract_abort:
        status = STATUS_ABORTED_CONTRACT
        blockers.append("evaluation_contract_abort")
    elif truncated or not_attempted:
        status = STATUS_PUBLICATION_BLOCKED
        if truncated:
            blockers.append("truncated_input")
        if not_attempted:
            blockers.append("not_attempted")
    else:
        status = STATUS_PUBLISHABLE
    if truncated and "truncated_input" not in blockers and status != STATUS_ABORTED_CONFIGURATION:
        blockers.append("truncated_input")
    if not_attempted and "not_attempted" not in blockers and status != STATUS_ABORTED_CONFIGURATION:
        blockers.append("not_attempted")
    if contract_abort and "evaluation_contract_abort" not in blockers and status != STATUS_ABORTED_CONFIGURATION:
        blockers.append("evaluation_contract_abort")
    return {
        "blockers": blockers,
        "publishable_under_protocol": status == STATUS_PUBLISHABLE,
        "real_execution_authorized": False,
        "status": status,
    }


def estimand_report(
    gold: tuple[Mapping[str, Any], ...] | None,
    states: list[Mapping[str, Any]],
    conditions: tuple[str, ...],
    cohort_ids: list[str],
    exported_rows: Mapping[str, list[Mapping[str, Any]]],
    export_sha256: Mapping[str, str],
    campaign_status: str,
    contract_abort: bool,
    contract_reason: str | None,
) -> dict[str, Any]:
    base = {
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "experimental_f1": False,
        "gold_source": "synthetic_fixture" if gold is not None else "not_requested",
        "result_role": "synthetic_contract_test_not_experimental_f1",
        "scoring_source": "conditions/<condition>/evaluation_predictions.jsonl",
    }
    if gold is None:
        empty = {"experimental_f1": False, "micro": None, "publication": "not_requested"}
        return {**base, "estimands": {name: empty for name in ("conservative", "conditional", "paired")}, "publication": "not_requested"}
    truncated_ids = _ids_with(states, lambda row: (row.get("diagnostics") or {}).get("truncated_input") is True)
    conflict = INPUT_TRUNCATION_CONFLICT if truncated_ids else None
    numbers_blocked = campaign_status in WITHHELD_PUBLICATIONS or contract_abort or bool(truncated_ids)
    conservative_blocked = campaign_status != STATUS_PUBLISHABLE or contract_abort or bool(truncated_ids)
    by_condition_conservative = {}
    by_condition_conditional = {}
    for condition in conditions:
        rows = list(exported_rows.get(condition, []))
        condition_states = [row for row in states if row["condition"] == condition]
        completed_ids = [
            row["document_id"]
            for row in condition_states
            if row.get("principal_state") == "completed"
            and (row.get("diagnostics") or {}).get("truncated_input") is not True
        ]
        by_condition_conservative[condition] = _labeled_score(
            gold,
            rows,
            cohort_ids,
            export_sha256.get(condition),
            publication=_publication(conservative_blocked, contract_abort, campaign_status, conflict, "conservative"),
            not_the_cohort=False,
            allow_numbers=not conservative_blocked,
        )
        by_condition_conditional[condition] = _labeled_score(
            _gold_for(gold, set(completed_ids)),
            _rows_for(rows, set(completed_ids)),
            completed_ids,
            export_sha256.get(condition),
            publication=_publication(numbers_blocked, contract_abort, campaign_status, conflict, "conditional"),
            not_the_cohort=True,
            allow_numbers=not numbers_blocked,
        )
    paired = _paired(
        gold,
        states,
        conditions,
        cohort_ids,
        exported_rows,
        export_sha256,
        allow_numbers=not numbers_blocked,
        publication=_publication(numbers_blocked, contract_abort, campaign_status, conflict, "paired"),
        conflict=conflict,
    )
    return {
        **base,
        "estimands": {
            "conditional": {
                "conditions": by_condition_conditional,
                "conflict_with_protocol": conflict,
                "experimental_f1": False,
                "not_the_cohort": True,
            },
            "conservative": {
                "conditions": by_condition_conservative,
                "experimental_f1": False,
                "not_the_cohort": False,
            },
            "paired": paired,
        },
        "publication": "synthetic_contract_only" if not conservative_blocked else "withheld",
        "withheld_reason": contract_reason,
    }


def evaluation_summary(
    gold: tuple[Mapping[str, Any], ...] | None,
    exported_rows: list[Mapping[str, Any]],
    contract_abort: bool,
    prediction_records: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Abort gate used by the contract test.

    Prediction records are not a scoring source. Passing them is rejected so a
    reconstructed copy cannot silently become the score.
    """
    if prediction_records:
        raise RuntimeError("prediction records are not a P0 scoring source")
    base = {
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "experimental_f1": False,
        "result_role": "synthetic_contract_test_not_experimental_f1",
    }
    if contract_abort:
        return {**base, "micro": None, "publication": "withheld_evaluation_contract_abort"}
    if gold is None:
        return {**base, "micro": None, "publication": "not_requested"}
    reason = contract_violation(exported_rows)
    if reason is not None:
        return {**base, "micro": None, "publication": "withheld_evaluation_contract_abort", "reason": reason}
    scored = _score(gold, exported_rows)
    return {**base, **scored}


def write_campaign_files(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "campaign_manifest.json", result["manifest"])
    atomic_write_jsonl(output_dir / "attempts.jsonl", list(result["attempts"]))
    atomic_write_jsonl(output_dir / "document_states.jsonl", list(result["document_states"]))
    atomic_write_jsonl(output_dir / "predictions.jsonl", list(result["prediction_records"]))
    atomic_write_json(output_dir / "coverage.json", result["coverage"])
    if "campaign_status" in result:
        atomic_write_json(output_dir / "campaign_status.json", result["campaign_status"])


def _retry_history(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    history = []
    for row in rows:
        for earlier in row.get("also_observed") or []:
            history.append(
                {
                    "attempt_index": earlier.get("attempt_index"),
                    "document_id": row["document_id"],
                    "followed_by_attempt_count": row.get("attempt_count"),
                    "principal_state": earlier.get("principal_state"),
                }
            )
    return history


def _publication(
    blocked: bool,
    contract_abort: bool,
    campaign_status: str,
    conflict: str | None,
    estimand: str,
) -> str:
    if not blocked:
        return "synthetic_contract_only"
    if contract_abort:
        return "withheld_evaluation_contract_abort"
    if campaign_status == STATUS_INCOMPLETE:
        return "withheld_campaign_incomplete"
    if campaign_status == STATUS_ABORTED_CONFIGURATION:
        return "withheld_configuration_abort"
    if conflict is not None and estimand != "conservative":
        return "withheld_truncated_input_conflicts_with_conditional_estimand"
    if conflict is not None:
        return "withheld_truncated_input"
    if campaign_status == STATUS_PUBLICATION_BLOCKED and estimand == "conservative":
        return "withheld_campaign_not_publishable"
    return "withheld"


def _labeled_score(
    gold: tuple[Mapping[str, Any], ...],
    rows: list[Mapping[str, Any]],
    document_ids: list[str],
    export_sha: str | None,
    *,
    publication: str,
    not_the_cohort: bool,
    allow_numbers: bool,
) -> dict[str, Any]:
    report = {
        "document_count": len(document_ids),
        "document_ids": list(document_ids),
        "experimental_f1": False,
        "export_sha256": export_sha,
        "not_the_cohort": not_the_cohort,
        "prediction_rows": len(rows),
        "publication": publication,
        "scored_row_ids": [row.get("id") for row in rows],
    }
    if not allow_numbers:
        report["counts"] = None
        report["micro"] = None
        return report
    if not document_ids:
        report["counts"] = None
        report["micro"] = None
        report["publication"] = "withheld_empty_subset"
        return report
    scored = _score(gold, rows)
    if scored.get("publication") == "withheld_evaluation_contract_abort":
        report["counts"] = None
        report["micro"] = None
        report["publication"] = "withheld_evaluation_contract_abort"
        report["reason"] = scored.get("reason")
        return report
    report["counts"] = scored["counts"]
    report["gold_support"] = scored["counts"]["gold_eligible_occurrences"]
    report["micro"] = scored["micro"]
    return report


def _paired(
    gold: tuple[Mapping[str, Any], ...],
    states: list[Mapping[str, Any]],
    conditions: tuple[str, ...],
    cohort_ids: list[str],
    exported_rows: Mapping[str, list[Mapping[str, Any]]],
    export_sha256: Mapping[str, str],
    *,
    allow_numbers: bool,
    publication: str,
    conflict: str | None,
) -> dict[str, Any]:
    by_document: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in states:
        by_document.setdefault(row["document_id"], {})[row["condition"]] = row
    intersection: list[str] = []
    exclusions: list[dict[str, Any]] = []
    for document_id in cohort_ids:
        reasons = []
        for condition in conditions:
            row = by_document.get(document_id, {}).get(condition, {})
            if row.get("principal_state") != "completed":
                reasons.append(f"{condition}:{row.get('principal_state') or row.get('harness_incident')}")
            elif (row.get("diagnostics") or {}).get("truncated_input") is True:
                reasons.append(f"{condition}:truncated_input")
        if reasons:
            exclusions.append({"document_id": document_id, "reasons": reasons})
        else:
            intersection.append(document_id)
    per_condition = {}
    if allow_numbers and intersection:
        selected = set(intersection)
        for condition in conditions:
            per_condition[condition] = _labeled_score(
                _gold_for(gold, selected),
                _rows_for(list(exported_rows.get(condition, [])), selected),
                intersection,
                export_sha256.get(condition),
                publication="synthetic_contract_only",
                not_the_cohort=True,
                allow_numbers=True,
            )
        paired_publication = "synthetic_contract_only"
    else:
        paired_publication = publication if not intersection else publication
        if allow_numbers and not intersection:
            paired_publication = "withheld_empty_intersection"
    return {
        "conditions": per_condition,
        "conflict_with_protocol": conflict,
        "document_count": len(intersection),
        "document_ids": intersection,
        "exclusions": exclusions,
        "experimental_f1": False,
        "not_the_cohort": True,
        "publication": paired_publication,
    }


def _ids_with(states: list[Mapping[str, Any]], predicate: Any) -> list[str]:
    return sorted({row["document_id"] for row in states if predicate(row)})


def _gold_for(gold: tuple[Mapping[str, Any], ...], document_ids: set[str]) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in gold if row.get("document_id") in document_ids)


def _rows_for(rows: list[Mapping[str, Any]], document_ids: set[str]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("document_id") in document_ids]


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
            "publication": "withheld_evaluation_contract_abort",
            "reason": str(exc),
        }
    return {
        "counts": report.summary["counts"],
        "micro": report.summary["micro"],
        "publication": "synthetic_contract_only",
    }
