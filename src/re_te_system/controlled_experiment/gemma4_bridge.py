"""Project a finished G04 producer log into the evaluation-bridge contract.

``manifest.json`` remains the producer log from B5.1. The bridge reads a
different set of names. This module derives those names from the same attempt
list and the same producer identity. It does not score F1, does not open the
phase lock, and does not read Hohfeld Gold.

The canonical scoring bytes are
``conditions/<condition>/evaluation_predictions.jsonl``, written by
``export_evaluation``. The producer log keeps the smoke label in
``generation_state``. Protocol principal states live only in
``document_states.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from re_te_system.controlled_experiment.campaign import (
    _diagnostics,
    _issue_dict,
    normalize_controlled_triples,
)
from re_te_system.controlled_experiment.contracts import (
    CAMPAIGN_CONTRACT_VERSION,
    SYNTHETIC_CONTRACT_TEST,
    campaign_scientific_id,
    sha256_text,
)
from re_te_system.controlled_experiment.dependencies import (
    ensure_conditioning_runtime,
    repository_root_from_package,
)
from re_te_system.controlled_experiment.gemma4 import (
    DATASET_ID,
    G04_PHASE_LOCK,
    MODEL_ID,
    PAIRED_CONDITIONS,
    TRANSPORT_POLICY_ID,
    GemmaDuplicatePublication,
    GemmaGoldAccessRefused,
    GemmaGoldNotAuthorized,
    GemmaModelIdentityError,
    refuse_gold_reference,
    separate_gemma_parser_input,
    separate_official_channels,
)
from re_te_system.controlled_experiment.persistence import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
)
from re_te_system.controlled_experiment.reporting import decide_campaign_status
from re_te_system.contracts import PREDICTION_CONTRACT_VERSION, ParseIssue, triple_to_dict
from re_te_system.evaluation_bridge.contracts import (
    FROZEN_POLICY,
    RESULT_ROLE_SYNTHETIC,
    BridgeRequest,
)
from re_te_system.manifests.run_manifest import sha256_file
from re_te_system.parsing.tsv import generation_issue, parse_tsv
from re_te_system.runner.run import export_evaluation


_WEIGHT_SHA = re.compile(r"^[0-9a-f]{64}$")
_BRIDGE_FILES = (
    "attempts.jsonl",
    "bridge_documents.jsonl",
    "campaign_manifest.json",
    "campaign_status.json",
    "document_states.jsonl",
    "synthetic_evaluation_summary.json",
)
_PROTOCOL_STATES = frozenset(
    {
        "completed",
        "context_blocked",
        "external_infrastructure_error",
        "model_execution_error",
        "not_attempted",
        "timeout",
    }
)


def weights_are_verified(snapshot_identity: Mapping[str, Any] | None) -> bool:
    """A missing weight digest is not a verified weight identity."""
    if not isinstance(snapshot_identity, Mapping):
        return False
    digest = snapshot_identity.get("model_weights_sha256")
    return isinstance(digest, str) and _WEIGHT_SHA.fullmatch(digest) is not None


def attach_g04_bridge_view(
    directory: Path,
    *,
    attempts: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    decoding: Any,
    documents_source: Path,
    provenance: Mapping[str, Any],
    revision: str,
) -> dict[str, Any]:
    """Write the bridge files beside the producer log. Do not score them here."""
    if G04_PHASE_LOCK is not True:
        raise GemmaGoldNotAuthorized("G04 phase lock must remain closed in this checkout")
    target = Path(directory)
    _assert_bridge_unpublished(target)
    _require_same_producer_identity(target, provenance)
    derived = _derive_documents(target, cohort, documents_source)
    shell = _shell_identity()
    protocol = _protocol_identity()
    projected = _project_attempts(attempts, cohort)
    identity = _bridge_identity(
        provenance=provenance,
        cohort=cohort,
        decoding=decoding,
        derived=derived,
        protocol=protocol,
        revision=revision,
        shell=shell,
    )
    scientific_id = campaign_scientific_id(identity)
    if scientific_id == provenance.get("run_id"):
        raise GemmaModelIdentityError("bridge identity must not reuse the producer run id")
    _stamp(projected, scientific_id)
    _write_attempts_and_states(target, projected)
    export_sha = _write_condition_files(target, projected, scientific_id)
    _align_producer_triples(target, projected)
    _align_producer_coverage(target, projected)
    status = decide_campaign_status(projected["states"], contract_abort=False)
    operational = _operational(provenance, derived, export_sha, protocol, shell)
    manifest = _campaign_manifest(identity, operational, provenance, scientific_id)
    atomic_write_json(target / "campaign_manifest.json", manifest)
    atomic_write_json(target / "campaign_status.json", status)
    atomic_write_json(target / "synthetic_evaluation_summary.json", _summary(export_sha))
    _require_same_producer_identity(target, provenance)
    return {
        "campaign_scientific_id": scientific_id,
        "document_bytes": operational["document_bytes"],
        "export_sha256": export_sha,
        "manifest": manifest,
        "producer_run_id": provenance["run_id"],
        "protocol_path": None if protocol["path"] is None else str(protocol["path"]),
        "states": projected["states"],
        "status": status,
    }


def evaluate_g04_synthetic(
    campaign_dir: Path,
    *,
    gold_path: Path,
    output_dir: Path,
    documents_path: Path,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Score one synthetic projection. Real Gold paths are refused before P0."""
    if G04_PHASE_LOCK is not True:
        raise GemmaGoldNotAuthorized("G04 phase lock must remain closed in this checkout")
    for candidate in (campaign_dir, gold_path, output_dir, documents_path, protocol_path):
        try:
            refuse_gold_reference(None if candidate is None else Path(candidate))
        except GemmaGoldAccessRefused:
            raise
    from re_te_system.evaluation_bridge.bridge import evaluate_campaign

    resolved_protocol = protocol_path
    if resolved_protocol is None:
        recorded = _protocol_identity()["path"]
        resolved_protocol = recorded
    return evaluate_campaign(
        BridgeRequest(
            campaign_dir=Path(campaign_dir),
            documents_path=Path(documents_path),
            gold_path=Path(gold_path),
            output_dir=Path(output_dir),
            protocol_path=None if resolved_protocol is None else Path(resolved_protocol),
        )
    )


def _assert_bridge_unpublished(directory: Path) -> None:
    """Refuse a second bridge publication. Do not delete or replace evidence."""
    if not directory.exists():
        return
    if not directory.is_dir():
        raise GemmaDuplicatePublication(f"refusing to replace {directory}")
    present = [name for name in _BRIDGE_FILES if (directory / name).exists()]
    if (directory / "conditions").exists():
        present.append("conditions")
    if present:
        raise GemmaDuplicatePublication(
            "refusing to overwrite bridge publication "
            + directory.name
            + ": "
            + ", ".join(present)
        )


def _require_same_producer_identity(directory: Path, provenance: Mapping[str, Any]) -> None:
    path = directory / "manifest.json"
    if not path.is_file():
        return
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded.get("identity") != provenance.get("identity") or recorded.get("run_id") != provenance.get("run_id"):
        raise GemmaModelIdentityError("producer manifest does not match the attempt identity")


def _derive_documents(
    directory: Path,
    cohort: Mapping[str, Any],
    documents_source: Path,
) -> dict[str, Any]:
    source = Path(documents_source)
    refuse_gold_reference(source)
    canonical_sha = sha256_file(source)
    if canonical_sha != cohort.get("documents_sha256"):
        raise GemmaModelIdentityError("canonical document hash does not match the loaded cohort")
    observed = _source_rows(source)
    records = list(cohort["records"])
    if [record.document_id for record in records] != list(cohort["document_ids"]):
        raise GemmaModelIdentityError("cohort order does not match its records")
    derived_rows = []
    for record in records:
        source_text = observed.get(record.document_id)
        if source_text is None or source_text != record.text:
            raise GemmaModelIdentityError("derived text does not match source_text")
        derived_rows.append({"document_id": record.document_id, "text": record.text})
    path = directory / "bridge_documents.jsonl"
    atomic_write_jsonl(path, derived_rows)
    derived_sha = sha256_file(path)
    if derived_sha == canonical_sha:
        raise GemmaModelIdentityError("derived document bytes must not replace the canonical file hash")
    reread = _text_rows(path)
    if reread != [(row["document_id"], row["text"]) for row in derived_rows]:
        raise GemmaModelIdentityError("derived document file changed while it was written")
    return {
        "canonical_sha256": canonical_sha,
        "derived_sha256": derived_sha,
        "path": path,
        "rows": derived_rows,
    }


def _source_rows(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        document_id = value.get("document_id")
        text = value.get("source_text")
        if not isinstance(document_id, str) or not isinstance(text, str):
            raise GemmaModelIdentityError("canonical cohort row is missing source_text")
        if document_id in rows:
            raise GemmaModelIdentityError("canonical cohort repeats a document id")
        rows[document_id] = text
    return rows


def _text_rows(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        rows.append((value["document_id"], value["text"]))
    return rows


def _shell_identity() -> dict[str, Any]:
    ensure_conditioning_runtime()
    from re_te_benchmark.conditioning.shell import SHELL_ID, conditioning_for, shell_template
    from re_te_system.controlled_experiment.contracts import SHELL_ID_REQUIRED

    if SHELL_ID != SHELL_ID_REQUIRED:
        raise GemmaModelIdentityError("the loaded shell is not common-instruction-shell-v2")
    return {
        "id": SHELL_ID,
        "sha256": sha256_text(shell_template()),
        "slots": {condition: sha256_text(conditioning_for(condition)) for condition in PAIRED_CONDITIONS},
    }


def _protocol_identity() -> dict[str, Any]:
    root = repository_root_from_package("re_te_benchmark")
    path = None if root is None else root / "docs" / "CONTROLLED_EXPERIMENT_PROTOCOL.md"
    if path is None or not path.is_file():
        return {"path": None, "sha256": None}
    return {"path": path, "sha256": sha256_text(path.read_text(encoding="utf-8"))}


def _project_attempts(
    attempts: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for attempt in attempts:
        condition = str(attempt["condition"])
        document_id = str(attempt["document_id"])
        if condition not in PAIRED_CONDITIONS:
            raise GemmaModelIdentityError("G04 bridge view only accepts C0 and C1")
        by_pair.setdefault((condition, document_id), []).append(dict(attempt))
    if any(len(rows) != 1 for rows in by_pair.values()):
        raise GemmaModelIdentityError("G04 produced more than one final row for a pair")
    texts = {record.document_id: record.text for record in cohort["records"]}
    attempt_rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    for document_id in cohort["document_ids"]:
        for condition in PAIRED_CONDITIONS:
            final = by_pair.get((condition, document_id))
            if final is None:
                attempt_rows.append(_not_attempted(condition, document_id, texts[document_id]))
                states.append(_state(condition, document_id, "not_attempted", 0, None, ()))
                continue
            chain, projection = _expand(final[0], texts[document_id])
            attempt_rows.extend(chain)
            states.append(
                _state(
                    condition,
                    document_id,
                    projection["principal_state"],
                    len(chain),
                    projection.get("diagnostics"),
                    chain[:-1],
                    parsed_triples=projection.get("parsed_triples"),
                    invalid_lines=projection.get("invalid_lines"),
                )
            )
            if projection["principal_state"] == "completed":
                completed[(condition, document_id)] = projection
    return {"attempts": attempt_rows, "completed": completed, "states": states}


def _expand(final: Mapping[str, Any], text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    prior = final.get("prior_failure")
    index = 1
    if isinstance(prior, Mapping):
        earlier = str(prior.get("state") or "")
        if earlier not in {"timeout", "external_infrastructure_error"}:
            raise GemmaModelIdentityError("a retained retry is not a timeout or infrastructure error")
        chain.append(
            _failure_attempt(final, index, earlier, None if prior.get("exception") is None else str(prior.get("exception")))
        )
        index = 2
    projection = _final_projection(final, text)
    chain.append(_protocol_attempt(final, index, projection, text))
    if int(final.get("attempt_count") or index) != len(chain):
        raise GemmaModelIdentityError("attempt count does not match the retained retry chain")
    return chain, projection


def _final_projection(final: Mapping[str, Any], text: str) -> dict[str, Any]:
    state = str(final.get("state") or "")
    if state in {"timeout", "external_infrastructure_error", "model_execution_error"}:
        return {
            "diagnostics": None,
            "invalid_lines": None,
            "model_output": None,
            "parsed_triples": None,
            "principal_state": state,
            "raw": final.get("raw"),
            "triples": [],
        }
    if state != "completed":
        raise GemmaModelIdentityError(f"G04 state {state} is not a protocol principal state")
    raw = final.get("raw")
    if not isinstance(raw, str):
        raise GemmaModelIdentityError("a completed generation has no raw output")
    separated = separate_gemma_parser_input(raw)
    channels = separate_official_channels(separated["parser_input"])
    metadata = {"output_reached_limit": True} if final.get("truncated") is True else {}
    extra = generation_issue(metadata)
    if channels["status"] in {"answer_only", "separated"} and isinstance(channels.get("answer"), str):
        parser_input = channels["answer"]
        parsed = parse_tsv(parser_input, None)
        issues = parsed.issues if extra is None else (*parsed.issues, extra)
        normalized = normalize_controlled_triples(parsed.triples)
        diagnostics = _diagnostics(parsed.triples, issues, metadata)
    else:
        # transport_and_parse does not give a malformed channel to parse_tsv.
        parser_input = ""
        message = channels.get("message")
        channel_issue = ParseIssue(
            "CHANNEL_NOT_SEPARATED",
            message if isinstance(message, str) else "channel markers are not separated",
            "soft",
        )
        issues = (channel_issue,) if extra is None else (channel_issue, extra)
        normalized = ()
        diagnostics = _diagnostics((), issues, metadata)
    return {
        "diagnostics": diagnostics or None,
        "invalid_lines": sum(1 for issue in issues if issue.code == "UNPARSEABLE_CHUNK"),
        "model_output": parser_input,
        "parsed_triples": len(normalized),
        "principal_state": "completed",
        "raw": raw,
        "removed_terminal_marker": separated["removed_terminal_marker"],
        "text": text,
        "transport_policy": separated["transport_policy"],
        "triples": [triple_to_dict(triple) for triple in normalized],
        "parse_issues": [_issue_dict(issue) for issue in issues],
    }


def _protocol_attempt(
    final: Mapping[str, Any],
    index: int,
    projection: Mapping[str, Any],
    text: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "attempt_index": index,
        "condition": final["condition"],
        "document_id": final["document_id"],
        "example_id": final["document_id"],
        "input_sha256": sha256_text(text),
        "principal_state": projection["principal_state"],
        "prompt_sha256": final.get("rendered_prompt_sha256"),
        "submitted": True,
    }
    if projection["principal_state"] == "completed":
        row["diagnostics"] = projection.get("diagnostics") or {}
        row["model_output"] = projection["model_output"]
        row["parse_issues"] = list(projection.get("parse_issues") or [])
        row["raw"] = projection["raw"]
        row["removed_terminal_marker"] = projection.get("removed_terminal_marker")
        row["transport_policy"] = projection.get("transport_policy") or TRANSPORT_POLICY_ID
    else:
        message = final.get("exception")
        if message:
            row["exception_message"] = str(message)
            row["exception_type"] = "generation_failure"
    return {key: value for key, value in row.items() if value is not None}


def _failure_attempt(
    final: Mapping[str, Any],
    index: int,
    state: str,
    message: str | None,
) -> dict[str, Any]:
    row = {
        "attempt_index": index,
        "condition": final["condition"],
        "document_id": final["document_id"],
        "example_id": final["document_id"],
        "principal_state": state,
        "submitted": True,
    }
    if message:
        row["exception_message"] = message
        row["exception_type"] = "generation_failure"
    return row


def _not_attempted(condition: str, document_id: str, text: str) -> dict[str, Any]:
    return {
        "condition": condition,
        "document_id": document_id,
        "example_id": document_id,
        "input_sha256": sha256_text(text),
        "principal_state": "not_attempted",
        "submitted": False,
    }


def _state(
    condition: str,
    document_id: str,
    principal: str,
    attempt_count: int,
    diagnostics: Mapping[str, Any] | None,
    earlier: Sequence[Mapping[str, Any]],
    parsed_triples: int | None = None,
    invalid_lines: int | None = None,
) -> dict[str, Any]:
    if principal not in _PROTOCOL_STATES:
        raise GemmaModelIdentityError(f"refusing to invent principal state {principal}")
    row: dict[str, Any] = {
        "attempt_count": attempt_count,
        "condition": condition,
        "document_id": document_id,
        "example_id": document_id,
        "principal_state": principal,
    }
    if diagnostics:
        row["diagnostics"] = dict(diagnostics)
    if earlier:
        row["also_observed"] = [
            {"attempt_index": item.get("attempt_index"), "principal_state": item.get("principal_state")}
            for item in earlier
        ]
    if parsed_triples is not None:
        row["parsed_triples"] = parsed_triples
    if invalid_lines is not None:
        row["invalid_lines"] = invalid_lines
    return row


def _stamp(projected: dict[str, Any], scientific_id: str) -> None:
    for row in projected["states"]:
        row["campaign_scientific_id"] = scientific_id


def _bridge_identity(
    *,
    provenance: Mapping[str, Any],
    cohort: Mapping[str, Any],
    decoding: Any,
    derived: Mapping[str, Any],
    protocol: Mapping[str, Any],
    revision: str,
    shell: Mapping[str, Any],
) -> dict[str, Any]:
    producer = dict(provenance["identity"])
    producer_decoding = producer.get("decoding")
    if producer_decoding != {"do_sample": decoding.do_sample, "max_new_tokens": decoding.max_new_tokens}:
        raise GemmaModelIdentityError("bridge decoding does not match the producer identity")
    snapshot = producer.get("snapshot_identity")
    verified = weights_are_verified(snapshot if isinstance(snapshot, Mapping) else None)
    documents = []
    for row in derived["rows"]:
        documents.append(
            {
                "document_id": row["document_id"],
                "example_id": row["document_id"],
                "text_sha256": sha256_text(row["text"]),
            }
        )
    return {
        "annotations_opened": False,
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "conditions": list(PAIRED_CONDITIONS),
        "dataset": {
            "annotations_opened": False,
            "declared_id": DATASET_ID,
            "declared_version": "0.1.0",
            "fixture": "synthetic_not_the_canonical_release",
        },
        "decoding": dict(producer_decoding),
        "documents": documents,
        "evaluation_policy": {**FROZEN_POLICY, "result_role": RESULT_ROLE_SYNTHETIC},
        "extractor": {
            "name": MODEL_ID,
            "prediction_source": "injected_session_not_weights",
            "revision": revision,
        },
        "g04_producer_identity": producer,
        "mode": SYNTHETIC_CONTRACT_TEST,
        "model_weights_verified": verified,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "protocol": {
            "id": "CONTROLLED_EXPERIMENT_PROTOCOL",
            "sha256": protocol["sha256"],
        },
        "request_order": producer.get("request_order"),
        "serialization_id": producer.get("serialization_id"),
        "shell": {"id": shell["id"], "sha256": shell["sha256"]},
        "slot_sha256": dict(shell["slots"]),
        "transport_policy": producer.get("transport_policy"),
    }


def _operational(
    provenance: Mapping[str, Any],
    derived: Mapping[str, Any],
    export_sha: Mapping[str, str],
    protocol: Mapping[str, Any],
    shell: Mapping[str, Any],
) -> dict[str, Any]:
    producer = provenance["identity"]
    snapshot = producer.get("snapshot_identity") if isinstance(producer, Mapping) else None
    gaps = []
    if not weights_are_verified(snapshot if isinstance(snapshot, Mapping) else None):
        gaps.append("model_weights_sha256_not_computed")
    repositories = provenance.get("repositories")
    if isinstance(repositories, Mapping):
        for name, record in repositories.items():
            if not isinstance(record, Mapping):
                continue
            if record.get("dirty") is not False or record.get("commit_identifies_tree") is not True:
                gaps.append(f"{name}:commit_does_not_identify_a_clean_tree")
            head = record.get("head")
            if not isinstance(head, str) or len(head) != 40:
                gaps.append(f"{name}:head_not_a_commit")
    if protocol.get("sha256") is None:
        gaps.append("protocol_text_unavailable")
    return {
        "document_bytes": {
            "byte_strings_identical": False,
            "canonical_field": "source_text",
            "canonical_sha256": derived["canonical_sha256"],
            "derived_field": "text",
            "derived_sha256": derived["derived_sha256"],
            "same_algorithm": "sha256_file",
            "texts_identical": True,
        },
        "export_sha256": dict(export_sha),
        "freeze_gaps": gaps,
        "g04_phase_lock": G04_PHASE_LOCK,
        "producer_operational": provenance.get("operational"),
        "producer_run_id": provenance.get("run_id"),
        "repositories": _repository_view(repositories),
        "shell_id": shell["id"],
    }


def _repository_view(repositories: Any) -> dict[str, Any]:
    if not isinstance(repositories, Mapping):
        return {}
    view = {}
    for name, record in repositories.items():
        if isinstance(record, Mapping):
            view[str(name)] = {
                "commit_identifies_tree": record.get("commit_identifies_tree"),
                "dirty": record.get("dirty"),
                "head": record.get("head"),
            }
    return view


def _campaign_manifest(
    identity: Mapping[str, Any],
    operational: Mapping[str, Any],
    provenance: Mapping[str, Any],
    scientific_id: str,
) -> dict[str, Any]:
    return {
        "campaign_scientific_id": scientific_id,
        "experimental_f1": False,
        "g04_authorized": False,
        "g04_phase_lock": G04_PHASE_LOCK,
        "gold_inference_authorized": False,
        "identity": dict(identity),
        "operational": dict(operational),
        "producer_result_role": "gemma4_g04_synthetic_core_not_scientific_f1",
        "producer_run_id": provenance.get("run_id"),
        "scientific_result": False,
    }


def _write_attempts_and_states(directory: Path, projected: Mapping[str, Any]) -> None:
    atomic_write_jsonl(directory / "attempts.jsonl", list(projected["attempts"]))
    atomic_write_jsonl(directory / "document_states.jsonl", list(projected["states"]))


def _write_condition_files(
    directory: Path,
    projected: Mapping[str, Any],
    scientific_id: str,
) -> dict[str, str]:
    digests = {}
    completed = projected["completed"]
    document_ids = []
    for row in projected["states"]:
        if row["condition"] == "C0" and row["document_id"] not in document_ids:
            document_ids.append(row["document_id"])
    for condition in PAIRED_CONDITIONS:
        condition_dir = directory / "conditions" / condition
        records = []
        for document_id in document_ids:
            projection = completed.get((condition, document_id))
            if projection is None:
                continue
            records.append(
                {
                    "condition": condition,
                    "document_id": document_id,
                    "example_id": f"{condition}:{document_id}",
                    "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
                    "run_id": scientific_id,
                    "validated": {"status": "projected", "triples": list(projection["triples"]), "violations": []},
                }
            )
        prediction_path = condition_dir / "predictions.jsonl"
        export_path = condition_dir / "evaluation_predictions.jsonl"
        partial = condition_dir / ".evaluation_predictions.jsonl.partial"
        atomic_write_jsonl(prediction_path, records)
        atomic_write_jsonl(condition_dir / "withheld_predictions.jsonl", [])
        try:
            export_evaluation(prediction_path, partial)
            partial.replace(export_path)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        digests[condition] = sha256_text(export_path.read_text(encoding="utf-8"))
    return digests


def _align_producer_coverage(directory: Path, projected: Mapping[str, Any]) -> None:
    """Count the same normalized triples the export already wrote."""
    path = directory / "coverage.json"
    if not path.is_file():
        return
    coverage = json.loads(path.read_text(encoding="utf-8"))
    counts = {
        (str(state["document_id"]), str(state["condition"])): int(state.get("parsed_triples") or 0)
        for state in projected["states"]
    }
    for document_id, conditions in coverage.items():
        if not isinstance(conditions, dict):
            continue
        for condition, row in conditions.items():
            if isinstance(row, dict) and (str(document_id), str(condition)) in counts:
                row["triple_count"] = counts[(str(document_id), str(condition))]
    atomic_write_text(
        path,
        json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _align_producer_triples(directory: Path, projected: Mapping[str, Any]) -> None:
    path = directory / "predictions.jsonl"
    if not path.is_file():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    completed = projected["completed"]
    for row in rows:
        projection = completed.get((row.get("condition"), row.get("document_id")))
        if projection is None:
            row["protocol_principal_state"] = _principal_for(projected["states"], row)
            continue
        row["parsed_triples"] = [
            {"object": triple["object"], "relation": triple["relation"], "subject": triple["subject"]}
            for triple in projection["triples"]
        ]
        row["protocol_principal_state"] = "completed"
    atomic_write_jsonl(path, rows)


def _principal_for(states: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> str | None:
    for state in states:
        if state.get("condition") == row.get("condition") and state.get("document_id") == row.get("document_id"):
            principal = state.get("principal_state")
            return principal if isinstance(principal, str) else None
    return None


def _summary(export_sha: Mapping[str, str]) -> dict[str, Any]:
    conditions = {condition: {"export_sha256": digest} for condition, digest in export_sha.items()}
    return {
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "estimands": {
            name: {"conditions": conditions}
            for name in ("conditional", "conservative", "paired")
        },
        "experimental_f1": False,
        "publication": "not_a_scoring_source",
        "result_role": RESULT_ROLE_SYNTHETIC,
        "scoring_source": "conditions/<condition>/evaluation_predictions.jsonl",
    }
