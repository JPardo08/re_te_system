"""Campaign layer for the controlled C0–C3 protocol.

The baseline runner remains responsible for baseline experiments. This module
does not call it. The reused pieces are ``parse_tsv``, ``generation_issue``,
``normalize_triples``, ``validate_structural``, and ``export_evaluation``.

EXP.3b.1 still accepts only ``SYNTHETIC_CONTRACT_TEST``. Scores are read from
per-condition files produced by ``export_evaluation``. A truncated input is
kept as evidence and is not a publishable observation. A harness failure is
not recorded as a model failure. Real process timeouts and real tokenizers
are not implemented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from re_te_system.contracts import (
    PREDICTION_CONTRACT_VERSION,
    ExtractionContext,
    ParseIssue,
    triple_to_dict,
)
from re_te_system.controlled_experiment.contracts import (
    CAMPAIGN_CONTRACT_VERSION,
    PREVIOUS_CAMPAIGN_CONTRACT_VERSION,
    REAL_C3_CONDITIONS,
    REQUIRED_CONDITIONS,
    RETRYABLE_STATES,
    SHELL_ID_REQUIRED,
    STATUS_ABORTED_CONFIGURATION,
    STATUS_ABORTED_CONTRACT,
    STATUS_IN_PROGRESS,
    STATUS_INCOMPLETE,
    SYNTHETIC_CONTRACT_TEST,
    AttemptRecord,
    CampaignError,
    ExternalInfrastructureError,
    HarnessPairError,
    PreflightNotAssessed,
    RealC3CampaignRejected,
    ScriptMissing,
    campaign_scientific_id,
    omit_absent,
    sha256_payload,
    sha256_text,
)
from re_te_system.controlled_experiment.dependencies import (
    SYSTEM_REPO,
    ensure_package,
    git_revision,
    repository_root_from_package,
)
from re_te_system.controlled_experiment.fingerprint import (
    audit_inventory,
    collect_implementation_parts,
    implementation_fingerprint,
)
from re_te_system.controlled_experiment.persistence import atomic_write_json, atomic_write_jsonl
from re_te_system.controlled_experiment.preflight import PreflightRequest, assess_preflight
from re_te_system.controlled_experiment.reporting import (
    contract_violation,
    coverage_report,
    decide_campaign_status,
    estimand_report,
    read_jsonl,
    write_campaign_files,
)
from re_te_system.controlled_experiment.synthetic import ScriptedTsvExtractor
from re_te_system.normalization.basic import normalize_triples
from re_te_system.parsing.tsv import generation_issue, parse_tsv
from re_te_system.runner.run import export_evaluation
from re_te_system.validation.structural import validate_structural


@dataclass(frozen=True)
class SyntheticDocument:
    document_id: str
    text: str
    example_id: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id or not self.text.strip():
            raise ValueError("a synthetic document needs an id and non-empty text")

    @property
    def resolved_example_id(self) -> str:
        return self.example_id or self.document_id


@dataclass(frozen=True)
class CampaignConfig:
    conditions: tuple[str, ...]
    cohort: tuple[SyntheticDocument, ...]
    extractor: ScriptedTsvExtractor
    count_tokens: Any
    counter_id: str | None
    tokenizer_id: str | None
    effective_context_limit: int | None
    reserved_output_tokens: int
    mode: str = SYNTHETIC_CONTRACT_TEST
    submitted_ids: frozenset[str] | None = None
    tokenizer_revision: str | None = None
    serialization_id: str | None = None
    special_token_allowance: int = 0
    seed: int | None = 0
    do_sample: bool = False
    synthetic_gold: tuple[Mapping[str, Any], ...] | None = None
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.do_sample:
            raise CampaignError("synthetic campaigns do not sample")
        if not self.cohort:
            raise CampaignError("a campaign needs a cohort")
        if len({item.document_id for item in self.cohort}) != len(self.cohort):
            raise CampaignError("document ids must be unique in the cohort")


def render_controlled_prompt(condition: str, text: str) -> str:
    """Render shell v2. The condition changes only the slot selected by the benchmark."""
    shell = _shell()
    return shell.render_condition(condition, text)


def run_controlled_campaign(config: CampaignConfig) -> dict[str, Any]:
    if config.output_dir is None:
        with TemporaryDirectory() as temporary:
            return _execute(config, Path(temporary))
    return _execute(config, config.output_dir)


def authorize_campaign(mode: str, conditions: tuple[str, ...] | list[str]) -> None:
    if any(condition not in REQUIRED_CONDITIONS for condition in conditions):
        raise CampaignError(f"unknown condition in {conditions!r}")
    if mode == SYNTHETIC_CONTRACT_TEST:
        return
    if any(condition in REAL_C3_CONDITIONS for condition in conditions):
        raise RealC3CampaignRejected(
            "This synthetic authorization cannot run a real C3 or C3-NL campaign. "
            "C3 remains NOT_READY and NOT_EXECUTABLE."
        )
    raise CampaignError("only SYNTHETIC_CONTRACT_TEST is authorized")


def _execute(config: CampaignConfig, output_dir: Path) -> dict[str, Any]:
    authorize_campaign(config.mode, config.conditions)
    shell = _shell()
    if shell.SHELL_ID != SHELL_ID_REQUIRED:
        raise CampaignError(f"shell identity is {shell.SHELL_ID!r}, not {SHELL_ID_REQUIRED}")
    if tuple(shell.CONDITIONS) != REQUIRED_CONDITIONS:
        raise CampaignError(f"shell conditions are {shell.CONDITIONS!r}")
    locations = {
        "re_te_benchmark": ensure_package("re_te_benchmark", "re_te_benchmark"),
        "re_te_evaluation": ensure_package("re_te_evaluation", "re_te_evaluation"),
    }
    benchmark_root = repository_root_from_package("re_te_benchmark")
    evaluation_root = repository_root_from_package("re_te_evaluation")
    readiness = _observe_readiness(benchmark_root)
    shell_text = shell.shell_template()
    shell_sha = sha256_text(shell_text)
    slots = {condition: shell.conditioning_for(condition) for condition in config.conditions}
    slot_hashes = {condition: sha256_text(value) for condition, value in slots.items()}
    revisions = {
        "re_te_benchmark": git_revision(benchmark_root),
        "re_te_evaluation": git_revision(evaluation_root),
        "re_te_system": git_revision(SYSTEM_REPO),
    }
    protocol = _protocol_identity(benchmark_root)
    protocol_text = _protocol_text(benchmark_root)
    parts = collect_implementation_parts(
        system_root=SYSTEM_REPO,
        benchmark_root=benchmark_root,
        evaluation_root=evaluation_root,
        shell_text=shell_text,
        slots=slots,
        protocol_text=protocol_text,
    )
    fingerprint = implementation_fingerprint(parts, revisions)
    freeze_gaps = _freeze_gaps(revisions, protocol, readiness)
    identity = _scientific_identity(config, shell, shell_sha, slot_hashes, protocol)
    scientific_id = campaign_scientific_id(identity)
    manifest = _manifest(
        scientific_id,
        identity,
        fingerprint,
        sorted(parts),
        revisions,
        locations,
        readiness,
        freeze_gaps,
        output_dir,
        config.output_dir,
    )
    pairs = [(condition, document) for condition in config.conditions for document in config.cohort]
    attempts: list[AttemptRecord] = []
    states: list[dict[str, Any]] = []
    publishable: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    _checkpoint(output_dir, manifest, attempts, states, _status(STATUS_IN_PROGRESS, scientific_id, []))
    problem = _configuration_problem(config)
    if problem is not None:
        exception_type, message = problem
        for condition, document in pairs:
            states.append(
                _harness_state(
                    scientific_id,
                    condition,
                    document,
                    "not_started_configuration_abort",
                    message,
                )
            )
        status = decide_campaign_status(states, contract_abort=False)
        _checkpoint(output_dir, manifest, attempts, states, status)
        _write_summary(
            output_dir,
            _empty_summary(status["status"], message),
        )
        raise exception_type(message)
    stopped = False
    for index, (condition, document) in enumerate(pairs):
        if config.submitted_ids is not None and document.document_id not in config.submitted_ids:
            attempts.append(
                AttemptRecord(
                    condition=condition,
                    document_id=document.document_id,
                    example_id=document.resolved_example_id,
                    attempt_index=None,
                    principal_state="not_attempted",
                    input_sha256=sha256_text(document.text),
                    shell_sha256=shell_sha,
                    slot_sha256=slot_hashes[condition],
                    submitted=False,
                    seed=config.seed,
                )
            )
            states.append(_final_state(scientific_id, condition, document, "not_attempted", (), 0))
            _checkpoint(output_dir, manifest, attempts, states, _status(STATUS_IN_PROGRESS, scientific_id, []))
            continue
        try:
            pair_attempts, final_state, prediction, keep = _run_pair(
                config,
                condition,
                document,
                shell_sha,
                slot_hashes[condition],
                scientific_id,
            )
        except HarnessPairError as exc:
            attempts.append(_incident_attempt(condition, document, exc))
            states.append(_harness_state(scientific_id, condition, document, exc.code, str(exc)))
            _mark_not_reached(pairs[index + 1 :], scientific_id, attempts, states)
            stopped = True
            _checkpoint(output_dir, manifest, attempts, states, _status(STATUS_IN_PROGRESS, scientific_id, [exc.code]))
            break
        except OSError:
            raise
        except Exception as exc:
            incident = HarnessPairError("unexpected_harness_error", f"{type(exc).__name__}: {exc}")
            attempts.append(_incident_attempt(condition, document, incident))
            states.append(
                _harness_state(scientific_id, condition, document, incident.code, str(incident))
            )
            _mark_not_reached(pairs[index + 1 :], scientific_id, attempts, states)
            stopped = True
            _checkpoint(
                output_dir,
                manifest,
                attempts,
                states,
                _status(STATUS_IN_PROGRESS, scientific_id, [incident.code]),
            )
            break
        else:
            attempts.extend(pair_attempts)
            states.append(final_state)
            if prediction is not None:
                if keep:
                    publishable.append(prediction)
                else:
                    withheld.append(prediction)
            _checkpoint(output_dir, manifest, attempts, states, _status(STATUS_IN_PROGRESS, scientific_id, []))
    return _finish(
        config,
        output_dir,
        manifest,
        scientific_id,
        attempts,
        states,
        publishable,
        withheld,
        fingerprint,
        stopped,
    )


def _finish(
    config: CampaignConfig,
    output_dir: Path,
    manifest: dict[str, Any],
    scientific_id: str,
    attempts: list[AttemptRecord],
    states: list[dict[str, Any]],
    publishable: list[dict[str, Any]],
    withheld: list[dict[str, Any]],
    fingerprint: str,
    stopped: bool,
) -> dict[str, Any]:
    exported_rows: dict[str, list[dict[str, Any]]] = {}
    export_sha: dict[str, str] = {}
    export_counts: dict[str, int] = {}
    contract_reason = None
    try:
        for condition in config.conditions:
            rows, digest = _export_condition(
                output_dir,
                condition,
                [row for row in publishable if row["condition"] == condition],
                [row for row in withheld if row["condition"] == condition],
            )
            exported_rows[condition] = rows
            export_sha[condition] = digest
            export_counts[condition] = len(rows)
            reason = contract_violation(rows)
            if reason is not None and contract_reason is None:
                contract_reason = reason
    except OSError as exc:
        try:
            atomic_write_json(
                output_dir / "campaign_status.json",
                _status(STATUS_ABORTED_CONTRACT, scientific_id, ["export_failed", str(exc)]),
            )
        except OSError as status_error:
            raise status_error from exc
        raise
    except Exception as exc:
        contract_reason = f"{type(exc).__name__}: {exc}"
    if contract_reason is not None:
        for state in states:
            if state.get("principal_state") == "completed":
                diagnostics = dict(state.get("diagnostics") or {})
                diagnostics["evaluation_contract_abort"] = True
                state["diagnostics"] = diagnostics
    status = decide_campaign_status(states, contract_abort=contract_reason is not None)
    if stopped and status["status"] != STATUS_ABORTED_CONTRACT:
        status["status"] = STATUS_INCOMPLETE
        if "interrupted" not in status["blockers"]:
            status["blockers"].append("interrupted")
        status["publishable_under_protocol"] = False
    coverage = coverage_report(
        config.cohort,
        config.conditions,
        states,
        [item.to_dict() for item in attempts],
        export_counts,
    )
    summary = estimand_report(
        config.synthetic_gold,
        states,
        config.conditions,
        [item.document_id for item in config.cohort],
        exported_rows,
        export_sha,
        status["status"],
        contract_reason is not None,
        contract_reason,
    )
    result = {
        "attempts": [item.to_dict() for item in attempts],
        "campaign_status": status,
        "coverage": coverage,
        "document_states": states,
        "evaluation_summary": summary,
        "implementation_fingerprint": fingerprint,
        "manifest": manifest,
        "prediction_records": publishable,
        "scientific_id": scientific_id,
        "withheld_prediction_records": withheld,
    }
    write_campaign_files(output_dir, result)
    _write_summary(output_dir, summary)
    return result


def _export_condition(
    output_dir: Path,
    condition: str,
    publishable: list[dict[str, Any]],
    withheld: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    directory = output_dir / "conditions" / condition
    directory.mkdir(parents=True, exist_ok=True)
    predictions = directory / "predictions.jsonl"
    exported = directory / "evaluation_predictions.jsonl"
    partial = directory / ".evaluation_predictions.jsonl.partial"
    atomic_write_jsonl(predictions, publishable)
    atomic_write_jsonl(directory / "withheld_predictions.jsonl", withheld)
    try:
        export_evaluation(predictions, partial)
        os.replace(partial, exported)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    rows = read_jsonl(exported)
    return rows, sha256_text(exported.read_text(encoding="utf-8"))


def _run_pair(
    config: CampaignConfig,
    condition: str,
    document: SyntheticDocument,
    shell_sha: str,
    slot_sha: str,
    scientific_id: str,
) -> tuple[list[AttemptRecord], dict[str, Any], dict[str, Any] | None, bool]:
    prompt = render_controlled_prompt(condition, document.text)
    prompt_sha = sha256_text(prompt)
    input_sha = sha256_text(document.text)
    attempts: list[AttemptRecord] = []
    completed_validation = None
    for attempt_index in (1, 2):
        try:
            preflight = assess_preflight(
                PreflightRequest(
                    prompt=prompt,
                    count_tokens=config.count_tokens,
                    effective_context_limit=config.effective_context_limit,
                    reserved_output_tokens=config.reserved_output_tokens,
                    tokenizer_id=config.tokenizer_id,
                    tokenizer_revision=config.tokenizer_revision,
                    serialization_id=config.serialization_id,
                    special_token_allowance=config.special_token_allowance,
                    counter_id=config.counter_id,
                )
            )
        except ValueError as exc:
            raise HarnessPairError(
                "token_counter_error",
                str(exc),
                prompt_sha256=prompt_sha,
                input_sha256=input_sha,
                slot_sha256=slot_sha,
                shell_sha256=shell_sha,
            ) from exc
        if preflight["assurance"] == "not_assessed":
            raise HarnessPairError(
                "preflight_not_assessed",
                str(preflight["reason"]),
                prompt_sha256=prompt_sha,
                input_sha256=input_sha,
                slot_sha256=slot_sha,
                shell_sha256=shell_sha,
            )
        common = dict(
            condition=condition,
            document_id=document.document_id,
            example_id=document.resolved_example_id,
            attempt_index=attempt_index,
            prompt_sha256=prompt_sha,
            slot_sha256=slot_sha,
            shell_sha256=shell_sha,
            input_sha256=input_sha,
            preflight=preflight,
            seed=config.seed,
            reserved_output_tokens=config.reserved_output_tokens,
            effective_context_limit=config.effective_context_limit,
        )
        if preflight["fits"] is not True:
            attempts.append(AttemptRecord(principal_state="context_blocked", **common))
            break
        context = ExtractionContext(str(attempt_index), document.document_id, None, condition)
        try:
            raw = config.extractor.extract(prompt, context)
        except ScriptMissing as exc:
            raise HarnessPairError(
                "script_missing",
                str(exc),
                prompt_sha256=prompt_sha,
                input_sha256=input_sha,
                slot_sha256=slot_sha,
                shell_sha256=shell_sha,
            ) from exc
        except ExternalInfrastructureError as exc:
            attempts.append(_failure_attempt("external_infrastructure_error", exc, common))
        except TimeoutError as exc:
            attempts.append(_failure_attempt("timeout", exc, common))
        except Exception as exc:
            attempts.append(_failure_attempt("model_execution_error", exc, common))
        else:
            if config.extractor.calls[-1][3] != prompt:
                raise HarnessPairError(
                    "prompt_contract_violation",
                    "extractor received a prompt that is not the rendered shell",
                    prompt_sha256=prompt_sha,
                    input_sha256=input_sha,
                    slot_sha256=slot_sha,
                    shell_sha256=shell_sha,
                )
            attempt, validated = _completed_attempt(raw.model_output, raw.generation_metadata, common)
            attempts.append(attempt)
            if attempt.principal_state == "completed":
                completed_validation = validated
        if attempts[-1].principal_state not in RETRYABLE_STATES or attempt_index == 2:
            break
    final = attempts[-1]
    earlier = [
        {"attempt_index": item.attempt_index, "principal_state": item.principal_state}
        for item in attempts[:-1]
    ]
    prediction = None
    keep = False
    parsed_triples = None
    invalid_lines = None
    if final.principal_state == "completed":
        if completed_validation is None:
            raise HarnessPairError("unexpected_harness_error", "completed attempt has no validated triples")
        invalid_lines = sum(1 for issue in final.parse_issues if issue.get("code") == "UNPARSEABLE_CHUNK")
        parsed_triples = len(completed_validation.triples)
        prediction = _prediction_record(scientific_id, condition, document, completed_validation)
        keep = final.diagnostics.get("truncated_input") is not True
        if not keep:
            prediction["withheld_reason"] = "truncated_input"
    return (
        attempts,
        _final_state(
            scientific_id,
            condition,
            document,
            final.principal_state,
            earlier,
            len(attempts),
            final.diagnostics,
            parsed_triples,
            invalid_lines,
        ),
        prediction,
        keep,
    )


def _configuration_problem(config: CampaignConfig) -> tuple[type[Exception], str] | None:
    missing = []
    if not config.tokenizer_id:
        missing.append("tokenizer_id")
    if config.count_tokens is None or not (config.counter_id or "").strip():
        missing.append("token_counter")
    if config.effective_context_limit is None:
        missing.append("effective_context_limit")
    if missing:
        return PreflightNotAssessed, "preflight_not_assessed:" + ",".join(missing)
    if config.reserved_output_tokens < 0 or config.special_token_allowance < 0:
        return CampaignError, "token reserves must be non-negative"
    if config.effective_context_limit is not None and config.effective_context_limit < 0:
        return CampaignError, "effective_context_limit must be non-negative"
    return None


def _scientific_identity(
    config: CampaignConfig,
    shell: Any,
    shell_sha: str,
    slot_hashes: Mapping[str, str],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "conditions": list(config.conditions),
        "context_budget": {
            "effective_context_limit": config.effective_context_limit,
            "reserved_output_tokens": config.reserved_output_tokens,
            "special_token_allowance": config.special_token_allowance,
        },
        "counter_id": config.counter_id,
        "decoding": {"do_sample": False, "seed": config.seed},
        "documents": [
            {
                "document_id": item.document_id,
                "example_id": item.resolved_example_id,
                "text_sha256": sha256_text(item.text),
            }
            for item in config.cohort
        ],
        "evaluation_policy": {
            "context_mode": "document",
            "eligibility_policy": "strict_inline",
            "matching_policy": "exact_multiset_v1",
            "metric_family": "triplet",
            "normalization_policy": "unicode_nfc+outer_strip+lowercase_false",
            "result_role": "synthetic_contract_test_not_experimental_f1",
        },
        "extractor": {
            "name": config.extractor.model_name,
            "prediction_source": "synthetic_script_not_model",
            "revision": config.extractor.model_revision,
            "script_sha256": sha256_payload(_script_identity(config.extractor)),
        },
        "input_truncation_policy": "preflight_block_and_truncated_input_not_publishable",
        "mode": SYNTHETIC_CONTRACT_TEST,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "protocol": protocol,
        "serialization_id": config.serialization_id,
        "shell": {"id": shell.SHELL_ID, "sha256": shell_sha},
        "slot_sha256": dict(slot_hashes),
        "tokenizer": {"id": config.tokenizer_id, "revision": config.tokenizer_revision},
    }


def _manifest(
    scientific_id: str,
    identity: Mapping[str, Any],
    fingerprint: str,
    component_labels: list[str],
    revisions: Mapping[str, Mapping[str, Any]],
    locations: Mapping[str, Mapping[str, str]],
    readiness: Mapping[str, Any],
    freeze_gaps: list[str],
    output_dir: Path,
    configured_output: Path | None,
) -> dict[str, Any]:
    return {
        "audit": {"relevant_tests": audit_inventory(SYSTEM_REPO)},
        "campaign_scientific_id": scientific_id,
        "contract_compatibility": {
            "previous": PREVIOUS_CAMPAIGN_CONTRACT_VERSION,
            "scoring": "conditions/<condition>/evaluation_predictions.jsonl",
            "version": CAMPAIGN_CONTRACT_VERSION,
            "exp3a_pooled_file": "not a scoring source and not written",
        },
        "identity": dict(identity),
        "identity_excludes": [
            "duration",
            "implementation_fingerprint",
            "local_paths",
            "repository_heads",
            "timestamps",
        ],
        "implementation_components": component_labels,
        "implementation_fingerprint": fingerprint,
        "observed_readiness": dict(readiness),
        "operational": {
            "created_at": None,
            "dependency_resolution": {name: value["method"] for name, value in locations.items()},
            "freeze_gaps": freeze_gaps,
            "freeze_status": "incomplete" if freeze_gaps else "content_addressed",
            "input_truncation_attestation": "not_verified_no_backbone_adapter",
            "output_dir": None if configured_output is None else str(output_dir),
            "repositories": {
                name: {
                    "commit_identifies_tree": record["commit_identifies_tree"],
                    "dirty": record["dirty"],
                    "head": record["head"],
                }
                for name, record in revisions.items()
            },
            "timeout_enforcement": "simulated_exception_only_pending_exp_3b",
        },
        "readiness_mutated": False,
    }


def _checkpoint(
    output_dir: Path,
    manifest: Mapping[str, Any],
    attempts: list[AttemptRecord],
    states: list[Mapping[str, Any]],
    status: Mapping[str, Any],
) -> None:
    atomic_write_json(output_dir / "campaign_manifest.json", manifest)
    atomic_write_jsonl(output_dir / "attempts.jsonl", [item.to_dict() for item in attempts])
    atomic_write_jsonl(output_dir / "document_states.jsonl", list(states))
    atomic_write_json(output_dir / "campaign_status.json", status)


def _status(name: str, scientific_id: str, blockers: list[str]) -> dict[str, Any]:
    return {
        "blockers": blockers,
        "campaign_scientific_id": scientific_id,
        "publishable_under_protocol": False,
        "real_execution_authorized": False,
        "status": name,
    }


def _mark_not_reached(
    remaining: list[tuple[str, SyntheticDocument]],
    scientific_id: str,
    attempts: list[AttemptRecord],
    states: list[dict[str, Any]],
) -> None:
    for condition, document in remaining:
        states.append(
            _harness_state(
                scientific_id,
                condition,
                document,
                "not_reached_after_interruption",
                "not reached after an earlier pair failed",
            )
        )


def _incident_attempt(
    condition: str,
    document: SyntheticDocument,
    exc: HarnessPairError,
) -> AttemptRecord:
    return AttemptRecord(
        condition=condition,
        document_id=document.document_id,
        example_id=document.resolved_example_id,
        attempt_index=None,
        principal_state=None,
        harness_incident=exc.code,
        harness_message=str(exc),
        prompt_sha256=exc.prompt_sha256,
        input_sha256=exc.input_sha256,
        slot_sha256=exc.slot_sha256,
        shell_sha256=exc.shell_sha256,
        submitted=True,
    )


def _harness_state(
    scientific_id: str,
    condition: str,
    document: SyntheticDocument,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "attempt_count": 0,
        "campaign_scientific_id": scientific_id,
        "condition": condition,
        "document_id": document.document_id,
        "example_id": document.resolved_example_id,
        "harness_incident": code,
        "harness_message": message,
    }


def _failure_attempt(state: str, exc: BaseException, common: dict[str, Any]) -> AttemptRecord:
    return AttemptRecord(
        principal_state=state,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        **common,
    )


def _completed_attempt(
    output: str,
    metadata: Mapping[str, Any],
    common: dict[str, Any],
) -> tuple[AttemptRecord, Any]:
    parsed = parse_tsv(output, None)
    extra = generation_issue(metadata)
    issues = parsed.issues if extra is None else (*parsed.issues, extra)
    normalized = normalize_triples(parsed.triples)
    validated = validate_structural(normalized, issues)
    attempt = AttemptRecord(
        principal_state="completed",
        model_output=output,
        diagnostics=_diagnostics(parsed.triples, parsed.issues, metadata),
        parse_issues=tuple(_issue_dict(item) for item in issues),
        **common,
    )
    return attempt, validated


def _prediction_record(
    scientific_id: str,
    condition: str,
    document: SyntheticDocument,
    validated: Any,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "document_id": document.document_id,
        "example_id": f"{condition}:{document.resolved_example_id}",
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "run_id": scientific_id,
        "validated": {
            "status": validated.status,
            "triples": [triple_to_dict(triple) for triple in validated.triples],
            "violations": [asdict(item) for item in validated.violations],
        },
    }


def _diagnostics(
    triples: tuple[Any, ...],
    issues: tuple[ParseIssue, ...],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if metadata.get("output_reached_limit") is True:
        diagnostics["output_truncated"] = True
    if metadata.get("truncated_input") is True:
        diagnostics["truncated_input"] = True
        diagnostics["silent_input_truncation"] = "operational_incident"
        diagnostics["conditioning_violation"] = "truncated_input"
    unparseable = any(item.code == "UNPARSEABLE_CHUNK" for item in issues)
    empty = any(item.code == "EMPTY_MODEL_OUTPUT" for item in issues)
    if empty and not triples:
        diagnostics["completed_empty"] = True
    elif not triples and unparseable:
        diagnostics["fully_malformed"] = True
    elif triples and unparseable:
        diagnostics["partially_malformed"] = True
    return diagnostics


def _issue_dict(issue: ParseIssue) -> dict[str, Any]:
    return omit_absent(
        {
            "chunk_index": issue.chunk_index,
            "code": issue.code,
            "message": issue.message,
            "severity": issue.severity,
        }
    )


def _final_state(
    scientific_id: str,
    condition: str,
    document: SyntheticDocument,
    principal_state: str | None,
    earlier: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    attempt_count: int,
    diagnostics: Mapping[str, Any] | None = None,
    parsed_triples: int | None = None,
    invalid_lines: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_count": attempt_count,
        "campaign_scientific_id": scientific_id,
        "condition": condition,
        "document_id": document.document_id,
        "example_id": document.resolved_example_id,
    }
    if principal_state is not None:
        record["principal_state"] = principal_state
    if diagnostics:
        record["diagnostics"] = dict(diagnostics)
    if earlier:
        record["also_observed"] = list(earlier)
    if parsed_triples is not None:
        record["parsed_triples"] = parsed_triples
    if invalid_lines is not None:
        record["invalid_lines"] = invalid_lines
    return record


def _script_identity(extractor: ScriptedTsvExtractor) -> list[dict[str, Any]]:
    rows = []
    for (condition, document_id, attempt), turn in sorted(
        extractor.scripts.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        rows.append(
            omit_absent(
                {
                    "attempt": attempt,
                    "condition": condition,
                    "document_id": document_id,
                    "kind": turn.kind,
                    "message": turn.message if turn.kind != "output" else None,
                    "output_sha256": None if turn.output is None else sha256_text(turn.output),
                    "output_reached_limit": True if turn.output_reached_limit is True else None,
                    "truncated_input": True if turn.truncated_input is True else None,
                }
            )
        )
    return rows


def _shell() -> Any:
    ensure_package("re_te_benchmark", "re_te_benchmark")
    from re_te_benchmark.conditioning import shell

    return shell


def _observe_readiness(benchmark_root: Path | None) -> dict[str, Any]:
    if benchmark_root is None:
        return {"status": "unresolved"}
    conditions_path = benchmark_root / "schemas" / "conditioning_conditions.json"
    package_path = benchmark_root / "schemas" / "c3" / "c3_ontology_v1.json"
    observed: dict[str, Any] = {"status": "observed", "source": "benchmark_files_read_only"}
    if conditions_path.is_file():
        payload = json.loads(conditions_path.read_text(encoding="utf-8"))
        conditions = payload.get("conditions", {})
        observed["C2"] = conditions.get("C2", {}).get("readiness")
        observed["C3"] = conditions.get("C3", {}).get("readiness")
        observed["conditions_file_sha256"] = sha256_text(conditions_path.read_text(encoding="utf-8"))
    else:
        observed["C2"] = None
        observed["C3"] = None
    if package_path.is_file():
        package = json.loads(package_path.read_text(encoding="utf-8"))
        observed["C3_package_experimental_condition_status"] = package.get(
            "experimental_condition_status"
        )
        observed["C3-NL"] = (
            "no separate conditions.json entry; package status is "
            + str(package.get("experimental_condition_status"))
        )
    return observed


def _protocol_identity(benchmark_root: Path | None) -> dict[str, Any]:
    if benchmark_root is None:
        return {"id": "CONTROLLED_EXPERIMENT_PROTOCOL", "sha256": None, "available": False}
    path = benchmark_root / "docs" / "CONTROLLED_EXPERIMENT_PROTOCOL.md"
    if not path.is_file():
        return {"id": "CONTROLLED_EXPERIMENT_PROTOCOL", "sha256": None, "available": False}
    return {
        "available": True,
        "id": "CONTROLLED_EXPERIMENT_PROTOCOL",
        "sha256": sha256_text(path.read_text(encoding="utf-8")),
    }


def _protocol_text(benchmark_root: Path | None) -> str | None:
    if benchmark_root is None:
        return None
    path = benchmark_root / "docs" / "CONTROLLED_EXPERIMENT_PROTOCOL.md"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _freeze_gaps(
    revisions: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> list[str]:
    gaps = []
    for name, record in revisions.items():
        if record.get("head") == "UNRESOLVED" or record.get("dirty") is not False:
            gaps.append(f"{name}:commit_does_not_identify_a_clean_tree")
    if protocol.get("available") is not True:
        gaps.append("protocol_text_unavailable")
    if readiness.get("status") != "observed":
        gaps.append("readiness_unresolved")
    return gaps


def _empty_summary(status: str, reason: str) -> dict[str, Any]:
    withheld = {"experimental_f1": False, "micro": None, "publication": "withheld_configuration_abort"}
    return {
        "campaign_contract_version": CAMPAIGN_CONTRACT_VERSION,
        "campaign_status": status,
        "estimands": {name: withheld for name in ("conservative", "conditional", "paired")},
        "experimental_f1": False,
        "publication": "withheld_configuration_abort",
        "reason": reason,
        "result_role": "synthetic_contract_test_not_experimental_f1",
    }


def _write_summary(output_dir: Path, summary: Mapping[str, Any]) -> None:
    atomic_write_json(output_dir / "synthetic_evaluation_summary.json", summary)
