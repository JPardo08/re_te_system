"""Fail-closed checks for a campaign directory and its benchmark inputs.

A manifest field is a claim. The corresponding file bytes have to reproduce
it. This module does not call P0 and does not repair a prediction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from re_te_system.controlled_experiment.contracts import (
    PRINCIPAL_STATES,
    SHELL_ID_REQUIRED,
    SYNTHETIC_CONTRACT_TEST,
    sha256_text,
)
from re_te_system.controlled_experiment.dependencies import SYSTEM_REPO, ensure_package
from re_te_system.controlled_experiment.reporting import (
    contract_violation,
    coverage_report,
    decide_campaign_status,
    gold_document_id,
    read_jsonl,
)
from re_te_system.evaluation_bridge.contracts import (
    FROZEN_POLICY,
    REGISTRY_PROTOCOL_SYNTHETIC,
    RESULT_ROLE_SYNTHETIC,
)
from re_te_system.normalization.basic import normalize_text
from re_te_system.parsing.tsv import parse_tsv


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STATUS_FIELDS = ("blockers", "publishable_under_protocol", "real_execution_authorized", "status")


@dataclass
class Inspection:
    blockers: list[str] = field(default_factory=list)
    scientific_reasons: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None
    states: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    exports: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prediction_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    export_sha256: dict[str, str] = field(default_factory=dict)
    final_attempts: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    gold: list[dict[str, Any]] | None = None
    conditions: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    recorded_status: dict[str, Any] | None = None
    recomputed_status: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    hashes: dict[str, Any] = field(default_factory=dict)
    preservation: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)
    comparability: str = "unresolved"
    registry_protocol_id: str | None = None
    technically_scorable: bool | None = None
    contract_abort: bool = False
    truncated_input: bool = False
    model_identity: dict[str, str] = field(default_factory=dict)
    mode: str | None = None

    def finish(self) -> None:
        self.blockers = sorted(set(self.blockers))
        self.scientific_reasons = sorted(set(self.scientific_reasons))


def inspect_campaign(request: Any) -> Inspection:
    """Read the campaign once and record every integrity failure that is visible."""
    found = Inspection()
    campaign = Path(request.campaign_dir)
    if not campaign.is_dir():
        found.blockers.append("missing_campaign_directory")
        found.finish()
        return found
    found.manifest = _object(campaign / "campaign_manifest.json", "campaign_manifest", found)
    found.recorded_status = _object(campaign / "campaign_status.json", "campaign_status", found)
    found.states = _jsonl(campaign / "document_states.jsonl", "document_states", found)
    found.attempts = _jsonl(campaign / "attempts.jsonl", "attempts", found)
    if found.manifest is None or not found.states:
        if found.manifest is None:
            found.blockers.append("unidentifiable_campaign")
        found.finish()
        return found
    identity = found.manifest.get("identity")
    if not isinstance(identity, dict):
        found.blockers.append("unidentifiable_campaign")
        found.finish()
        return found
    found.identity = identity
    found.mode = identity.get("mode") if isinstance(identity.get("mode"), str) else None
    if not isinstance(found.manifest.get("campaign_scientific_id"), str):
        found.blockers.append("unidentifiable_campaign")
    _conditions_and_documents(found, request)
    _load_condition_files(campaign, found)
    _load_gold(request, found)
    _verify_documents(request, found)
    _verify_protocol_and_shell(request, found)
    _verify_benchmark_manifest(request, found)
    _verify_export_hashes(campaign, found)
    _verify_states(found)
    _verify_exports(found)
    _verify_normalization(found)
    _verify_status(found)
    _provenance(found)
    _registry(request, found)
    _coverage(found)
    _preservation(found)
    found.finish()
    return found


def _object(path: Path, label: str, found: Inspection) -> dict[str, Any] | None:
    if not path.is_file():
        found.blockers.append(f"missing_{label}")
        return None
    try:
        text = path.read_text(encoding="utf-8")
        if text == "" or not text.endswith("\n"):
            raise OSError("incomplete json")
        value = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError):
        found.blockers.append(f"corrupt_{label}")
        return None
    if not isinstance(value, dict):
        found.blockers.append(f"corrupt_{label}")
        return None
    found.hashes[f"{label}_sha256"] = sha256_text(text)
    return value


def _jsonl(
    path: Path,
    blocker: str,
    found: Inspection,
    hash_name: str | None = None,
) -> list[dict[str, Any]]:
    if not path.is_file():
        found.blockers.append(f"missing_{blocker}")
        return []
    try:
        rows = read_jsonl(path)
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError):
        found.blockers.append(f"corrupt_{blocker}")
        return []
    if not all(isinstance(row, dict) for row in rows):
        found.blockers.append(f"corrupt_{blocker}")
        return []
    found.hashes[hash_name or f"{blocker}_sha256"] = sha256_text(text)
    return rows


def _conditions_and_documents(found: Inspection, request: Any) -> None:
    identity = found.identity or {}
    conditions = identity.get("conditions")
    documents = identity.get("documents")
    if (
        not isinstance(conditions, list)
        or not conditions
        or not all(isinstance(item, str) and item for item in conditions)
    ):
        found.blockers.append("unidentifiable_campaign")
        return
    if not isinstance(documents, list) or not documents:
        found.blockers.append("unidentifiable_campaign")
        return
    identifiers = []
    for item in documents:
        if not isinstance(item, dict) or not isinstance(item.get("document_id"), str):
            found.blockers.append("unidentifiable_campaign")
            return
        identifiers.append(item["document_id"])
    if len(set(identifiers)) != len(identifiers):
        found.blockers.append("duplicate_document_state")
    found.conditions = list(conditions)
    found.document_ids = identifiers
    expected = request.expected_document_ids
    if expected is not None and list(expected) != identifiers:
        missing = set(expected) - set(identifiers)
        extra = set(identifiers) - set(expected)
        if missing:
            found.blockers.append("missing_document_state")
        if extra:
            found.blockers.append("unexpected_document_id")
    policy = identity.get("evaluation_policy")
    if not isinstance(policy, dict):
        found.blockers.append("protocol_mismatch")
        return
    for key, required in FROZEN_POLICY.items():
        if policy.get(key) != required:
            found.blockers.append("protocol_mismatch")
    if found.mode == SYNTHETIC_CONTRACT_TEST and policy.get("result_role") != RESULT_ROLE_SYNTHETIC:
        found.blockers.append("synthetic_result_role_rewritten")
    extractor = identity.get("extractor")
    if isinstance(extractor, dict):
        name = extractor.get("name")
        revision = extractor.get("revision")
        found.model_identity = {
            "name": name if isinstance(name, str) else "unidentified",
            "revision": revision if isinstance(revision, str) else "unidentified",
        }
    else:
        found.model_identity = {"name": "unidentified", "revision": "unidentified"}
        found.scientific_reasons.append("model_unidentified")


def _load_condition_files(campaign: Path, found: Inspection) -> None:
    if not found.conditions:
        return
    directories = set()
    root = campaign / "conditions"
    if root.is_dir():
        directories = {path.name for path in root.iterdir() if path.is_dir()}
    if directories != set(found.conditions):
        found.blockers.append("condition_mismatch")
    for condition in found.conditions:
        directory = root / condition
        export_path = directory / "evaluation_predictions.jsonl"
        if not export_path.is_file():
            found.blockers.append("missing_export")
            found.exports[condition] = []
        else:
            found.exports[condition] = _jsonl(
                export_path, "export", found, f"export_{condition}_sha256"
            )
            try:
                found.export_sha256[condition] = sha256_text(export_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                found.blockers.append("corrupt_export")
        predictions = directory / "predictions.jsonl"
        withheld = directory / "withheld_predictions.jsonl"
        if not predictions.is_file():
            found.blockers.append("missing_prediction_records")
        else:
            found.prediction_rows[condition] = _jsonl(
                predictions, "prediction_records", found, f"predictions_{condition}_sha256"
            )
        if not withheld.is_file():
            found.blockers.append("missing_withheld_predictions")
        else:
            _jsonl(withheld, "withheld_predictions", found, f"withheld_{condition}_sha256")


def _load_gold(request: Any, found: Inspection) -> None:
    path = Path(request.gold_path)
    if not path.is_file():
        found.blockers.append("missing_gold")
        return
    ensure_package("re_te_evaluation", "re_te_evaluation")
    from re_te_evaluation.eligibility import InputContractError, read_records

    try:
        records = read_records(path)
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, InputContractError):
        found.blockers.append("corrupt_gold")
        return
    found.gold = records
    found.hashes["gold_sha256"] = sha256_text(text)
    outside = False
    unresolved = False
    for record in records:
        document_id = gold_document_id(record)
        if document_id is None:
            unresolved = True
        elif found.document_ids and document_id not in found.document_ids:
            outside = True
    if unresolved:
        found.blockers.append("gold_document_id_unresolved")
    if outside:
        found.blockers.append("unexpected_document_id")


def _verify_documents(request: Any, found: Inspection) -> None:
    identity = found.identity or {}
    claimed = identity.get("documents")
    if not isinstance(claimed, list):
        return
    path = request.documents_path
    if path is None:
        found.scientific_reasons.append("input_bytes_unverified")
        return
    source = Path(path)
    if not source.is_file():
        found.blockers.append("incorrect_input_hash")
        return
    try:
        rows = read_jsonl(source)
        found.hashes["documents_sha256"] = sha256_text(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        found.blockers.append("incorrect_input_hash")
        return
    observed = []
    texts: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("document_id"), str):
            found.blockers.append("incorrect_input_hash")
            return
        if not isinstance(row.get("text"), str):
            found.blockers.append("incorrect_input_hash")
            return
        observed.append(row["document_id"])
        texts[row["document_id"]] = row["text"]
    if observed != found.document_ids:
        if set(observed) - set(found.document_ids):
            found.blockers.append("unexpected_document_id")
        if set(found.document_ids) - set(observed):
            found.blockers.append("missing_document_state")
        if observed != found.document_ids and "unexpected_document_id" not in found.blockers:
            found.blockers.append("incorrect_input_hash")
    for item in claimed:
        if not isinstance(item, dict):
            found.blockers.append("incorrect_input_hash")
            continue
        document_id = item.get("document_id")
        claimed_hash = item.get("text_sha256")
        text = texts.get(document_id)
        if not isinstance(claimed_hash, str) or text is None or sha256_text(text) != claimed_hash:
            found.blockers.append("incorrect_input_hash")


def _verify_protocol_and_shell(request: Any, found: Inspection) -> None:
    identity = found.identity or {}
    protocol = identity.get("protocol")
    claimed = protocol.get("sha256") if isinstance(protocol, dict) else None
    path = request.protocol_path
    if path is None:
        found.scientific_reasons.append("protocol_bytes_unverified")
    else:
        source = Path(path)
        try:
            observed = sha256_text(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            observed = None
        found.hashes["protocol_sha256"] = observed
        if not isinstance(claimed, str) or observed != claimed:
            found.blockers.append("incorrect_protocol_hash")
    shell = identity.get("shell")
    slots = identity.get("slot_sha256")
    try:
        ensure_package("re_te_benchmark", "re_te_benchmark")
        from re_te_benchmark.conditioning.shell import SHELL_ID, conditioning_for, shell_template
    except (ImportError, OSError):
        found.scientific_reasons.append("shell_unverified")
        return
    if not isinstance(shell, dict) or shell.get("id") != SHELL_ID_REQUIRED or SHELL_ID != SHELL_ID_REQUIRED:
        found.blockers.append("shell_mismatch")
    elif shell.get("sha256") != sha256_text(shell_template()):
        found.blockers.append("shell_mismatch")
    if isinstance(slots, dict):
        for condition in found.conditions:
            try:
                current = sha256_text(conditioning_for(condition))
            except (KeyError, ValueError, OSError):
                found.blockers.append("condition_mismatch")
                continue
            if slots.get(condition) != current:
                found.blockers.append("shell_mismatch")
    else:
        found.blockers.append("shell_mismatch")


def _verify_benchmark_manifest(request: Any, found: Inspection) -> None:
    if request.benchmark_manifest_path is None:
        found.scientific_reasons.append("benchmark_manifest_unverified")
        return
    source = Path(request.benchmark_manifest_path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        found.blockers.append("incorrect_documents_hash")
        return
    if not isinstance(manifest, dict):
        found.blockers.append("incorrect_documents_hash")
        return
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        found.blockers.append("incorrect_documents_hash")
        return
    by_role = {
        item.get("role"): item
        for item in artifacts
        if isinstance(item, dict) and isinstance(item.get("role"), str)
    }
    _match_artifact_hash(by_role.get("canonical_documents"), request.documents_path, "incorrect_documents_hash", found)
    _match_artifact_hash(by_role.get("canonical_annotations"), request.gold_path, "incorrect_gold_hash", found)


def _match_artifact_hash(
    artifact: Mapping[str, Any] | None,
    path: Path | None,
    blocker: str,
    found: Inspection,
) -> None:
    """Compare a manifest claim with the artifact bytes, not with a copied hash field."""
    if artifact is None or path is None:
        found.blockers.append(blocker)
        return
    claimed = artifact.get("sha256")
    try:
        observed = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        found.blockers.append(blocker)
        return
    if claimed != observed:
        found.blockers.append(blocker)


def _manifest_export_claims(found: Inspection) -> dict[str, str] | None:
    """Canonical claim written on the campaign manifest, outside scientific identity."""
    operational = (found.manifest or {}).get("operational")
    if not isinstance(operational, dict) or "export_sha256" not in operational:
        return None
    claimed = operational.get("export_sha256")
    if not isinstance(claimed, dict):
        found.blockers.append("incorrect_export_hash")
        return {}
    return {str(key): value for key, value in claimed.items()}


def _check_export_claim(found: Inspection, condition: str, digest: object) -> None:
    observed = found.export_sha256.get(condition)
    if not isinstance(digest, str) or observed is None or digest != observed:
        found.blockers.append("incorrect_export_hash")


def _summary_export_claims(summary: Mapping[str, Any]) -> dict[str, set[str]]:
    claimed: dict[str, set[str]] = {}
    estimands = summary.get("estimands")
    if isinstance(estimands, dict):
        for payload in estimands.values():
            if not isinstance(payload, dict):
                continue
            conditions = payload.get("conditions")
            if not isinstance(conditions, dict):
                continue
            for condition, row in conditions.items():
                if isinstance(row, dict) and isinstance(row.get("export_sha256"), str):
                    claimed.setdefault(str(condition), set()).add(row["export_sha256"])
    return claimed


def _verify_export_hashes(campaign: Path, found: Inspection) -> None:
    """Check recorded export digests against the export bytes.

    Synthetic campaigns keep the claim inside ``synthetic_evaluation_summary.json``.
    The same digest is also stored at ``operational.export_sha256`` in the
    campaign manifest. That manifest field is the claim a future non-synthetic
    producer must write. A missing claim stays blocked. The digest is
    ``sha256_text`` of the export file text. This check does not score F1.
    """
    summary_path = campaign / "synthetic_evaluation_summary.json"
    manifest_claims = _manifest_export_claims(found)
    if found.mode != SYNTHETIC_CONTRACT_TEST:
        if summary_path.is_file():
            found.blockers.append("synthetic_summary_on_nonsynthetic_campaign")
        if not manifest_claims:
            found.blockers.append("export_hash_unrecorded")
            return
        for condition, digest in manifest_claims.items():
            _check_export_claim(found, condition, digest)
        return
    summary = _object(summary_path, "evaluation_summary", found)
    if summary is None:
        return
    claimed = _summary_export_claims(summary)
    if not claimed:
        found.blockers.append("incorrect_export_hash")
        return
    for condition, digests in claimed.items():
        observed = found.export_sha256.get(condition)
        if observed is None or digests != {observed}:
            found.blockers.append("incorrect_export_hash")
    if manifest_claims:
        for condition, digest in manifest_claims.items():
            _check_export_claim(found, condition, digest)
            if condition in claimed and claimed[condition] != {digest}:
                found.blockers.append("incorrect_export_hash")


def _verify_states(found: Inspection) -> None:
    if not found.conditions or not found.document_ids:
        return
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in found.states:
        condition = row.get("condition")
        document_id = row.get("document_id")
        if not isinstance(condition, str) or not isinstance(document_id, str):
            found.blockers.append("corrupt_document_states")
            continue
        if condition not in found.conditions:
            found.blockers.append("condition_mismatch")
        if document_id not in found.document_ids:
            found.blockers.append("unexpected_document_id")
        grouped.setdefault((condition, document_id), []).append(row)
    expected = {(condition, document_id) for condition in found.conditions for document_id in found.document_ids}
    if set(grouped) - expected:
        found.blockers.append("unexpected_document_id")
    if expected - set(grouped):
        found.blockers.append("missing_document_state")
    if any(len(rows) != 1 for rows in grouped.values()):
        found.blockers.append("duplicate_document_state")
    for rows in grouped.values():
        if len(rows) != 1:
            continue
        row = rows[0]
        if row.get("harness_incident"):
            continue
        principal = row.get("principal_state")
        if principal not in PRINCIPAL_STATES:
            found.blockers.append("invalid_principal_state")
        if (row.get("diagnostics") or {}).get("truncated_input") is True:
            found.truncated_input = True
    last: dict[tuple[str, str], dict[str, Any]] = {}
    for attempt in found.attempts:
        condition = attempt.get("condition")
        document_id = attempt.get("document_id")
        if isinstance(condition, str) and isinstance(document_id, str):
            last[(condition, document_id)] = attempt
    found.final_attempts = last
    if "duplicate_document_state" in found.blockers or "invalid_principal_state" in found.blockers:
        return
    for key, rows in grouped.items():
        if len(rows) != 1 or rows[0].get("harness_incident"):
            continue
        attempt = last.get(key)
        if attempt is None:
            found.blockers.append("coverage_inconsistency")
            continue
        if attempt.get("principal_state") != rows[0].get("principal_state"):
            found.blockers.append("coverage_inconsistency")
        if rows[0].get("principal_state") == "completed" and not isinstance(attempt.get("model_output"), str):
            found.blockers.append("raw_output_missing")


def _verify_exports(found: Inspection) -> None:
    structural = {
        "duplicate_document_state",
        "missing_document_state",
        "unexpected_document_id",
        "invalid_principal_state",
        "corrupt_document_states",
        "corrupt_export",
    }
    if structural & set(found.blockers):
        return
    allowed: dict[str, set[str]] = {}
    for row in found.states:
        if row.get("principal_state") != "completed":
            continue
        if (row.get("diagnostics") or {}).get("truncated_input") is True:
            continue
        allowed.setdefault(str(row["condition"]), set()).add(str(row["document_id"]))
    for condition, rows in found.exports.items():
        reason = contract_violation(rows)
        if reason is not None:
            found.blockers.append("evaluation_contract_abort")
            found.contract_abort = True
        for row in rows:
            document_id = row.get("document_id")
            if not isinstance(document_id, str) or document_id == "":
                found.blockers.append("evaluation_contract_abort")
                found.contract_abort = True
                continue
            if document_id not in allowed.get(condition, set()):
                found.blockers.append("coverage_inconsistency")
        prediction_path_rows = _prediction_keys(found, condition)
        export_keys = [
            (row.get("document_id"), row.get("subject"), row.get("relation"), row.get("object"))
            for row in rows
        ]
        if prediction_path_rows is not None and prediction_path_rows != export_keys:
            found.blockers.append("exporter_contract_violation")
        raw_keys = _raw_system_keys(found, condition)
        if raw_keys is not None and raw_keys != export_keys:
            found.blockers.append("exporter_contract_violation")


def _prediction_keys(found: Inspection, condition: str) -> list[tuple[Any, Any, Any, Any]] | None:
    records = found.prediction_rows.get(condition)
    if records is None:
        return None
    keys: list[tuple[Any, Any, Any, Any]] = []
    for record in records:
        validated = record.get("validated")
        triples = validated.get("triples") if isinstance(validated, dict) else None
        if not isinstance(triples, list):
            return None
        for triple in triples:
            if not isinstance(triple, dict):
                return None
            keys.append(
                (
                    record.get("document_id"),
                    triple.get("subject"),
                    triple.get("relation"),
                    triple.get("object"),
                )
            )
    return keys


def _raw_system_keys(found: Inspection, condition: str) -> list[tuple[Any, Any, Any, Any]] | None:
    from re_te_system.controlled_experiment.campaign import normalize_controlled_triples

    finals = found.final_attempts
    keys: list[tuple[Any, Any, Any, Any]] = []
    for document_id in found.document_ids:
        state = _state(found, condition, document_id)
        if state is None or state.get("principal_state") != "completed":
            continue
        if (state.get("diagnostics") or {}).get("truncated_input") is True:
            continue
        attempt = finals.get((condition, document_id))
        if not isinstance(attempt, dict) or not isinstance(attempt.get("model_output"), str):
            return None
        parsed = parse_tsv(attempt["model_output"], None)
        for triple in normalize_controlled_triples(parsed.triples):
            keys.append((document_id, triple.subject, triple.relation, triple.object))
    return keys


def _state(found: Inspection, condition: str, document_id: str) -> dict[str, Any] | None:
    matches = [
        row
        for row in found.states
        if row.get("condition") == condition and row.get("document_id") == document_id
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _verify_normalization(found: Inspection) -> None:
    found.normalization = {
        "asymmetry_observed": False,
        "conditions": {},
        "controlled_export_policy": FROZEN_POLICY["normalization_policy"],
        "p0_policy": FROZEN_POLICY["normalization_policy"],
        "policies_changed": False,
        "sota_baseline_policy": "NFC+internal_whitespace_collapse+outer_strip",
        "system_policy": "NFC+internal_whitespace_collapse+outer_strip",
    }
    if found.contract_abort or "exporter_contract_violation" in found.blockers:
        return
    if "corrupt_export" in found.blockers or "raw_output_missing" in found.blockers:
        return
    ensure_package("re_te_evaluation", "re_te_evaluation")
    from re_te_evaluation.contracts import NormalizationPolicy

    policy = NormalizationPolicy()
    finals = found.final_attempts
    examples: list[dict[str, str]] = []
    for condition in found.conditions:
        protocol_keys = []
        export_keys = []
        for row in found.exports.get(condition, []):
            export_keys.append(
                (
                    row.get("document_id"),
                    policy.normalize(row.get("subject")) if isinstance(row.get("subject"), str) else None,
                    policy.normalize(row.get("relation")) if isinstance(row.get("relation"), str) else None,
                    policy.normalize(row.get("object")) if isinstance(row.get("object"), str) else None,
                )
            )
        for document_id in found.document_ids:
            state = _state(found, condition, document_id)
            if state is None or state.get("principal_state") != "completed":
                continue
            if (state.get("diagnostics") or {}).get("truncated_input") is True:
                continue
            attempt = finals.get((condition, document_id))
            if not isinstance(attempt, dict) or not isinstance(attempt.get("model_output"), str):
                continue
            parsed = parse_tsv(attempt["model_output"], None)
            for index, triple in enumerate(parsed.triples):
                protocol_value = (
                    document_id,
                    policy.normalize(triple.subject),
                    policy.normalize(triple.relation),
                    policy.normalize(triple.object),
                )
                protocol_keys.append(protocol_value)
                system_value = (
                    document_id,
                    normalize_text(triple.subject),
                    normalize_text(triple.relation),
                    normalize_text(triple.object),
                )
                if system_value != protocol_value and len(examples) < 5:
                    examples.append(
                        {
                            "condition": condition,
                            "document_id": document_id,
                            "occurrence_index": str(index),
                            "protocol_subject": protocol_value[1] or "",
                            "system_subject": system_value[1],
                            "protocol_relation": protocol_value[2] or "",
                            "system_relation": system_value[2],
                            "protocol_object": protocol_value[3] or "",
                            "system_object": system_value[3],
                        }
                    )
        if Counter(protocol_keys) != Counter(export_keys):
            found.normalization["asymmetry_observed"] = True
            found.normalization["conditions"][condition] = {
                "protocol_rows": [
                    {
                        "document_id": item[0],
                        "id": f"protocol-normalization:{condition}:{item[0]}:{index:06d}",
                        "object": item[3],
                        "relation": item[2],
                        "subject": item[1],
                    }
                    for index, item in enumerate(protocol_keys)
                ]
            }
    if found.normalization["asymmetry_observed"]:
        found.blockers.append("normalization_asymmetry")
        found.normalization["examples"] = examples


def _verify_status(found: Inspection) -> None:
    structural = {
        "duplicate_document_state",
        "missing_document_state",
        "unexpected_document_id",
        "invalid_principal_state",
        "corrupt_document_states",
        "unidentifiable_campaign",
        "condition_mismatch",
    }
    if structural & set(found.blockers) or not found.states:
        return
    recomputed = decide_campaign_status(found.states, contract_abort=found.contract_abort)
    found.recomputed_status = recomputed
    recorded = found.recorded_status
    if recorded is None:
        return
    for key in _STATUS_FIELDS:
        if recorded.get(key) != recomputed.get(key):
            found.blockers.append("campaign_status_mismatch")
            return


def _provenance(found: Inspection) -> None:
    operational = (found.manifest or {}).get("operational")
    repositories = operational.get("repositories") if isinstance(operational, dict) else None
    if not isinstance(repositories, dict):
        found.scientific_reasons.append("provenance_unidentifiable")
        return
    for name in ("re_te_benchmark", "re_te_evaluation", "re_te_system"):
        record = repositories.get(name)
        if not isinstance(record, dict):
            found.scientific_reasons.append(f"provenance_{name}_unidentifiable")
            continue
        head = record.get("head")
        if record.get("dirty") is not False or record.get("commit_identifies_tree") is not True:
            found.scientific_reasons.append(f"provenance_{name}_not_a_clean_tree")
        if not isinstance(head, str) or _COMMIT.fullmatch(head) is None:
            found.scientific_reasons.append(f"provenance_{name}_head_not_a_commit")
    gaps = operational.get("freeze_gaps") if isinstance(operational, dict) else None
    if isinstance(gaps, list) and gaps:
        found.scientific_reasons.append("provenance_freeze_incomplete")


def _registry(request: Any, found: Inspection) -> None:
    if found.mode == SYNTHETIC_CONTRACT_TEST:
        found.registry_protocol_id = REGISTRY_PROTOCOL_SYNTHETIC
    else:
        explicit = (found.identity or {}).get("registry_protocol_id")
        found.registry_protocol_id = explicit if isinstance(explicit, str) else None
        if found.registry_protocol_id is None:
            found.scientific_reasons.append("registry_protocol_unidentified")
    path = (
        Path(request.registry_protocols_path)
        if request.registry_protocols_path is not None
        else SYSTEM_REPO / "campaigns" / "exp3c" / "catalog" / "protocols.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        found.scientific_reasons.append("registry_unreadable")
        found.comparability = "unresolved"
        return
    protocols = payload.get("protocols") if isinstance(payload, dict) else None
    record = None
    if isinstance(protocols, list):
        for item in protocols:
            if isinstance(item, dict) and item.get("id") == found.registry_protocol_id:
                record = item
                break
    if record is None:
        found.scientific_reasons.append("registry_protocol_missing")
        found.comparability = "unresolved"
        return
    declared = record.get("comparability")
    found.technically_scorable = record.get("technically_scorable") is True
    synthetic = found.mode == SYNTHETIC_CONTRACT_TEST or found.registry_protocol_id == REGISTRY_PROTOCOL_SYNTHETIC
    if synthetic:
        found.comparability = "not_comparable"
        found.scientific_reasons.append("synthetic_contract")
        if declared == "scientific_comparable":
            found.scientific_reasons.append("registry_rule_rejects_synthetic_promotion")
    elif declared in {"scientific_comparable", "diagnostic_only", "not_comparable"}:
        found.comparability = str(declared)
    else:
        found.comparability = "unresolved"
        found.scientific_reasons.append("registry_comparability_unresolved")
    if found.comparability != "scientific_comparable":
        found.scientific_reasons.append("registry_not_comparable")
    if found.technically_scorable is not True:
        found.scientific_reasons.append("registry_not_technically_scorable")


def _coverage(found: Inspection) -> None:
    structural = {
        "duplicate_document_state",
        "missing_document_state",
        "unexpected_document_id",
        "invalid_principal_state",
        "corrupt_document_states",
        "condition_mismatch",
        "unidentifiable_campaign",
    }
    if structural & set(found.blockers) or not found.document_ids:
        return

    class _Document:
        def __init__(self, document_id: str) -> None:
            self.document_id = document_id

    try:
        found.coverage = coverage_report(
            tuple(_Document(document_id) for document_id in found.document_ids),
            tuple(found.conditions),
            found.states,
            found.attempts,
            {condition: len(rows) for condition, rows in found.exports.items()},
        )
    except (RuntimeError, KeyError, TypeError):
        found.blockers.append("coverage_inconsistency")


def _preservation(found: Inspection) -> None:
    issue_counts: Counter[str] = Counter()
    raw_outputs = 0
    for attempt in found.attempts:
        if isinstance(attempt.get("model_output"), str):
            raw_outputs += 1
        for issue in attempt.get("parse_issues") or []:
            if isinstance(issue, dict) and isinstance(issue.get("code"), str):
                issue_counts[issue["code"]] += 1
    found.preservation = {
        "attempts_sha256": found.hashes.get("attempts_sha256"),
        "document_states_sha256": found.hashes.get("document_states_sha256"),
        "export_sha256": dict(found.export_sha256),
        "parse_issue_counts": dict(sorted(issue_counts.items())),
        "raw_output_count": raw_outputs,
        "raw_outputs_preserved": "raw_output_missing" not in found.blockers,
        "scoring_source": "conditions/<condition>/evaluation_predictions.jsonl",
        "summary_micro_not_a_scoring_source": True,
    }
