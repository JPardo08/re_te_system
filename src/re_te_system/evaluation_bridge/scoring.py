"""File-based P0 scoring for the three protocol estimands.

Numbers are produced only by ``read_records``, ``evaluate_triplets``, and
``write_report``. The campaign summary's stored F1 is not a scoring source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from re_te_system.controlled_experiment.contracts import (
    STATUS_PUBLISHABLE,
    SYNTHETIC_CONTRACT_TEST,
    sha256_text,
)
from re_te_system.controlled_experiment.dependencies import ensure_package
from re_te_system.controlled_experiment.persistence import atomic_write_json, atomic_write_jsonl
from re_te_system.controlled_experiment.reporting import (
    INPUT_TRUNCATION_CONFLICT,
    WITHHELD_PUBLICATIONS,
    _publication,
)
from re_te_system.evaluation_bridge.contracts import (
    BRIDGE_CONTRACT_VERSION,
    ESTIMANDS,
    BridgeError,
    FROZEN_POLICY,
)
from re_te_system.evaluation_bridge.integrity import Inspection, gold_document_id


def score_inspection(request: Any, found: Inspection) -> dict[str, Any]:
    """Return the estimand section. Diagnostic P0 output is written for asymmetry."""
    integrity = [item for item in found.blockers if item != "normalization_asymmetry"]
    if integrity or found.gold is None or found.recomputed_status is None:
        _clear_protocol_rows(found)
        return _withheld_estimands(found, "withheld_integrity")
    if "normalization_asymmetry" in found.blockers:
        _write_normalization_diagnostics(request, found)
        return _withheld_estimands(found, "withheld_normalization_asymmetry")
    return _operational_estimands(request, found)


def frozen_config() -> Any:
    ensure_package("re_te_evaluation", "re_te_evaluation")
    from re_te_evaluation import (
        ContextMode,
        EligibilityPolicy,
        EvaluationConfig,
        MetricFamily,
        NormalizationPolicy,
    )

    return EvaluationConfig(
        metric_family=MetricFamily.TRIPLET,
        context_mode=ContextMode.DOCUMENT,
        eligibility_policy=EligibilityPolicy.STRICT_INLINE,
        normalization=NormalizationPolicy(),
        matching_policy=FROZEN_POLICY["matching_policy"],
    )


def model_dirname(found: Inspection) -> str:
    raw = f"{found.model_identity.get('name', 'unidentified')}@{found.model_identity.get('revision', 'unidentified')}"
    cleaned = "".join(character if character.isalnum() or character in "-._@" else "_" for character in raw)
    return cleaned or "unidentified-model"


def _operational_estimands(request: Any, found: Inspection) -> dict[str, Any]:
    status = str(found.recomputed_status["status"]) if found.recomputed_status else ""
    conflict = INPUT_TRUNCATION_CONFLICT if found.truncated_input else None
    numbers_blocked = status in WITHHELD_PUBLICATIONS or found.contract_abort or found.truncated_input
    conservative_blocked = status != STATUS_PUBLISHABLE or found.contract_abort or found.truncated_input
    intersection, exclusions, retained = _paired_partition(found)
    estimands: dict[str, Any] = {}
    estimands["conservative"] = _family(
        request,
        found,
        "conservative",
        blocked=conservative_blocked,
        status=status,
        conflict=conflict,
        subsets={condition: list(found.document_ids) for condition in found.conditions},
        not_the_cohort=False,
    )
    estimands["conditional"] = _family(
        request,
        found,
        "conditional",
        blocked=numbers_blocked,
        status=status,
        conflict=conflict,
        subsets={condition: _completed_ids(found, condition) for condition in found.conditions},
        not_the_cohort=True,
    )
    estimands["paired"] = _family(
        request,
        found,
        "paired",
        blocked=numbers_blocked,
        status=status,
        conflict=conflict,
        subsets={condition: list(intersection) for condition in found.conditions},
        not_the_cohort=True,
        empty_publication="withheld_empty_intersection",
    )
    estimands["paired"]["document_count"] = len(intersection)
    estimands["paired"]["document_ids"] = list(intersection)
    estimands["paired"]["exclusions"] = exclusions
    estimands["paired"]["retained_diagnostics"] = retained
    if not numbers_blocked and not intersection:
        estimands["paired"]["publication"] = "withheld_empty_intersection"
    return estimands


def _family(
    request: Any,
    found: Inspection,
    estimand: str,
    *,
    blocked: bool,
    status: str,
    conflict: str | None,
    subsets: Mapping[str, list[str]],
    not_the_cohort: bool,
    empty_publication: str | None = None,
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    publications: list[str] = []
    for condition, document_ids in subsets.items():
        publication = _label(
            blocked=blocked or not document_ids,
            status=status,
            conflict=conflict,
            estimand=estimand,
            mode=found.mode,
            empty=not document_ids and not blocked,
            empty_publication=empty_publication,
        )
        if blocked or not document_ids:
            conditions[condition] = _empty_score(
                publication,
                document_ids,
                found.export_sha256.get(condition),
                not_the_cohort=not_the_cohort,
            )
        else:
            conditions[condition] = _score_condition(
                request,
                found,
                estimand,
                condition,
                document_ids,
                publication,
                not_the_cohort=not_the_cohort,
            )
        publications.append(conditions[condition]["publication"])
    unique = set(publications)
    parent = publications[0] if len(unique) == 1 and publications else "mixed"
    return {
        "conditions": conditions,
        "conflict_with_protocol": conflict if estimand != "conservative" else None,
        "experimental_f1": False,
        "not_the_cohort": not_the_cohort,
        "publication": parent,
    }


def _label(
    *,
    blocked: bool,
    status: str,
    conflict: str | None,
    estimand: str,
    mode: str | None,
    empty: bool,
    empty_publication: str | None,
) -> str:
    if empty:
        return empty_publication or "withheld_empty_subset"
    if blocked:
        return _publication(True, False, status, conflict, estimand)
    if mode == SYNTHETIC_CONTRACT_TEST:
        return "synthetic_contract_only"
    return "operational_numbers_not_scientific"


def _score_condition(
    request: Any,
    found: Inspection,
    estimand: str,
    condition: str,
    document_ids: list[str],
    publication: str,
    *,
    not_the_cohort: bool,
) -> dict[str, Any]:
    destination = (
        Path(request.output_dir)
        / "models"
        / model_dirname(found)
        / "conditions"
        / condition
        / "estimands"
        / estimand
    )
    gold_rows = _gold_for(found.gold or [], document_ids)
    prediction_rows = _rows_for(found.exports.get(condition, []), document_ids)
    scored = _invoke_p0(gold_rows, prediction_rows, destination, request)
    if scored is None:
        payload = _empty_score(
            "withheld_evaluation_contract_abort",
            document_ids,
            found.export_sha256.get(condition),
            not_the_cohort=not_the_cohort,
        )
        payload["reason"] = "p0_rejected_the_file_contract"
        return payload
    _write_publication(destination, publication, estimand, scientific=False)
    summary = scored["summary"]
    return {
        "config_id": summary["config_id"],
        "counts": summary["counts"],
        "document_count": len(document_ids),
        "document_ids": list(document_ids),
        "experimental_f1": False,
        "export_sha256": found.export_sha256.get(condition),
        "gold_support": summary["counts"]["gold_eligible_occurrences"],
        "micro": summary["micro"],
        "normalization_policy": summary["normalization_policy"],
        "not_the_cohort": not_the_cohort,
        "p0_output": destination.relative_to(Path(request.output_dir)).as_posix(),
        "p0_summary_sha256": sha256_text((destination / "summary.json").read_text(encoding="utf-8")),
        "prediction_support": summary["counts"]["predictions"],
        "publication": publication,
    }


def _invoke_p0(
    gold_rows: list[Mapping[str, Any]],
    prediction_rows: list[Mapping[str, Any]],
    destination: Path,
    request: Any | None = None,
) -> dict[str, Any] | None:
    """Score through the same three calls the official CLI uses.

    In-process invocation is retained. The CLI's provenance helper supplies
    evaluation code identity and, when a benchmark manifest is present,
    dataset and schema identity. Subprocess invocation is not required:
    ``write_report`` emits the same six artifacts, and a contract error
    withholds the estimand instead of exiting the process.
    """
    ensure_package("re_te_evaluation", "re_te_evaluation")
    from re_te_evaluation.cli import _manifest_metadata
    from re_te_evaluation.eligibility import InputContractError, read_records
    from re_te_evaluation.reporting.writers import write_report
    from re_te_evaluation.triplet.metrics import evaluate_triplets

    destination.mkdir(parents=True, exist_ok=True)
    gold_path = destination / "gold_subset.jsonl"
    prediction_path = destination / "predictions_subset.jsonl"
    atomic_write_jsonl(gold_path, list(gold_rows))
    atomic_write_jsonl(prediction_path, list(prediction_rows))
    config = frozen_config()
    manifest_path = None if request is None else getattr(request, "benchmark_manifest_path", None)
    claimed_gold = gold_path if request is None or getattr(request, "gold_path", None) is None else request.gold_path
    provenance = {
        "bridge_contract_version": BRIDGE_CONTRACT_VERSION,
        "scoring_interface": "read_records+evaluate_triplets+write_report",
    }
    try:
        provenance.update(
            _manifest_metadata(
                None if manifest_path is None else str(manifest_path),
                str(claimed_gold),
            )
        )
        report = evaluate_triplets(
            read_records(gold_path),
            read_records(prediction_path),
            config,
            provenance,
        )
    except InputContractError:
        return None
    write_report(report, destination)
    disk = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    if disk["counts"] != report.summary["counts"] or disk["micro"] != report.summary["micro"]:
        raise BridgeError("P0 summary.json does not match evaluate_triplets")
    if disk["normalization_policy"] != FROZEN_POLICY["normalization_policy"]:
        raise BridgeError("P0 did not apply the frozen normalization policy")
    if disk["config_id"] != config.config_id:
        raise BridgeError("P0 config_id does not match the frozen evaluation identity")
    recorded = disk.get("provenance")
    if not isinstance(recorded, dict) or not recorded.get("code_commit"):
        raise BridgeError("P0 summary is missing evaluation code identity")
    if recorded.get("scoring_interface") != provenance["scoring_interface"]:
        raise BridgeError("P0 summary does not record the scoring interface")
    return {"summary": disk}


def _write_normalization_diagnostics(request: Any, found: Inspection) -> None:
    if found.gold is None:
        _clear_protocol_rows(found)
        return
    for condition, payload in list(found.normalization.get("conditions", {}).items()):
        protocol_rows = payload.get("protocol_rows") if isinstance(payload, dict) else None
        if not isinstance(protocol_rows, list):
            continue
        export_dir = (
            Path(request.output_dir) / "diagnostics" / "normalization" / condition / "export_p0"
        )
        protocol_dir = (
            Path(request.output_dir)
            / "diagnostics"
            / "normalization"
            / condition
            / "protocol_normalization_p0"
        )
        export_score = _invoke_p0(
            list(found.gold),
            list(found.exports.get(condition, [])),
            export_dir,
            request,
        )
        protocol_score = _invoke_p0(list(found.gold), protocol_rows, protocol_dir, request)
        _write_publication(export_dir, "non_publishable_normalization_diagnostic", "normalization", scientific=False)
        _write_publication(
            protocol_dir, "non_publishable_normalization_diagnostic", "normalization", scientific=False
        )
        found.normalization["conditions"][condition] = {
            "export_p0": _diagnostic_counts(export_score),
            "protocol_p0": _diagnostic_counts(protocol_score),
        }
    found.normalization.pop("protocol_rows", None)


def _diagnostic_counts(scored: Mapping[str, Any] | None) -> dict[str, Any]:
    if scored is None:
        return {"counts": None, "micro": None, "publication": "withheld_evaluation_contract_abort"}
    summary = scored["summary"]
    return {
        "counts": summary["counts"],
        "gold_support": summary["counts"]["gold_eligible_occurrences"],
        "micro": summary["micro"],
        "prediction_support": summary["counts"]["predictions"],
        "publication": "non_publishable_normalization_diagnostic",
    }


def _clear_protocol_rows(found: Inspection) -> None:
    for payload in found.normalization.get("conditions", {}).values():
        if isinstance(payload, dict):
            payload.pop("protocol_rows", None)


def _withheld_estimands(found: Inspection, publication: str) -> dict[str, Any]:
    estimands = {}
    for estimand in ESTIMANDS:
        conditions = {
            condition: _empty_score(
                publication,
                found.document_ids,
                found.export_sha256.get(condition),
                not_the_cohort=estimand != "conservative",
            )
            for condition in found.conditions
        }
        estimands[estimand] = {
            "conditions": conditions,
            "experimental_f1": False,
            "not_the_cohort": estimand != "conservative",
            "publication": publication,
        }
    if "paired" in estimands:
        estimands["paired"]["document_ids"] = []
        estimands["paired"]["document_count"] = 0
        estimands["paired"]["exclusions"] = []
    return estimands


def _empty_score(
    publication: str,
    document_ids: list[str],
    export_sha: str | None,
    *,
    not_the_cohort: bool,
) -> dict[str, Any]:
    return {
        "counts": None,
        "document_count": len(document_ids),
        "document_ids": list(document_ids),
        "experimental_f1": False,
        "export_sha256": export_sha,
        "gold_support": None,
        "micro": None,
        "not_the_cohort": not_the_cohort,
        "prediction_support": None,
        "publication": publication,
    }


def _write_publication(destination: Path, publication: str, estimand: str, *, scientific: bool) -> None:
    atomic_write_json(
        destination / "publication.json",
        {
            "estimand": estimand,
            "experimental_f1": scientific,
            "publication": publication,
            "scientific_result": scientific,
        },
    )


def _gold_for(gold: list[Mapping[str, Any]], document_ids: list[str]) -> list[Mapping[str, Any]]:
    selected = set(document_ids)
    return [row for row in gold if gold_document_id(row) in selected]


def _rows_for(rows: list[Mapping[str, Any]], document_ids: list[str]) -> list[Mapping[str, Any]]:
    selected = set(document_ids)
    return [row for row in rows if row.get("document_id") in selected]


def _completed_ids(found: Inspection, condition: str) -> list[str]:
    completed = {
        row.get("document_id")
        for row in found.states
        if row.get("condition") == condition
        and row.get("principal_state") == "completed"
        and (row.get("diagnostics") or {}).get("truncated_input") is not True
    }
    return [document_id for document_id in found.document_ids if document_id in completed]


def _paired_partition(
    found: Inspection,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    by_document: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in found.states:
        document_id = row.get("document_id")
        condition = row.get("condition")
        if isinstance(document_id, str) and isinstance(condition, str):
            by_document.setdefault(document_id, {})[condition] = row
    intersection: list[str] = []
    exclusions: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for document_id in found.document_ids:
        reasons = []
        diagnostics = []
        for condition in found.conditions:
            row = by_document.get(document_id, {}).get(condition, {})
            if row.get("principal_state") != "completed":
                reasons.append(f"{condition}:{row.get('principal_state') or row.get('harness_incident')}")
            elif (row.get("diagnostics") or {}).get("truncated_input") is True:
                reasons.append(f"{condition}:truncated_input")
            else:
                diagnostics.append(
                    {"condition": condition, "diagnostics": dict(row.get("diagnostics") or {})}
                )
        if reasons:
            exclusions.append({"document_id": document_id, "reasons": reasons})
        else:
            intersection.append(document_id)
            retained.append({"diagnostics": diagnostics, "document_id": document_id})
    return intersection, exclusions, retained
