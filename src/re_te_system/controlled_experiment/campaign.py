"""Campaign layer for the controlled C0–C3 protocol.

``run_pipeline`` remains the baseline runner. It windows inputs, turns an
extractor exception into a prediction record, and has no preflight, so it
cannot represent the protocol states without changing baseline behavior.
This module does not call it.

The reused pieces are ``parse_tsv``, ``generation_issue``, ``normalize_triples``,
``validate_structural``, and ``export_evaluation``. The rendered prompt is
never split and never shortened. A context block does not call the extractor.
A model exception is not stored as an empty generation.

EXP.3a accepts only ``SYNTHETIC_CONTRACT_TEST``. That mode may render C3 and
C3-NL. Any other mode that includes those conditions is rejected. Benchmark
readiness is observed and not modified. A scripted timeout does not enforce
a real process limit; that enforcement is pending for EXP.3b.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
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
    REAL_C3_CONDITIONS,
    REQUIRED_CONDITIONS,
    RETRYABLE_STATES,
    SHELL_ID_REQUIRED,
    SYNTHETIC_CONTRACT_TEST,
    AttemptRecord,
    CampaignError,
    PreflightNotAssessed,
    RealC3CampaignRejected,
    ScriptMissing,
    SimulatedInfrastructureError,
    campaign_scientific_id,
    canonical_json,
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
from re_te_system.controlled_experiment.preflight import PreflightRequest, assess_preflight
from re_te_system.controlled_experiment.reporting import (
    coverage_report,
    evaluation_summary,
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
            raise CampaignError("EXP.3a synthetic campaigns do not sample")
        if not self.cohort:
            raise CampaignError("a campaign needs a cohort")
        if len({item.document_id for item in self.cohort}) != len(self.cohort):
            raise CampaignError("document ids must be unique in the cohort")


def render_controlled_prompt(condition: str, text: str) -> str:
    """Render shell v2. The condition changes only the slot selected by the benchmark."""
    shell = _shell()
    return shell.render_condition(condition, text)


def run_controlled_campaign(config: CampaignConfig) -> dict[str, Any]:
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
    fingerprint = _relevant_fingerprint(shell_text, protocol.get("sha256"))
    freeze_gaps = _freeze_gaps(revisions, protocol, readiness)
    identity = {
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
        "freeze_status": "incomplete" if freeze_gaps else "content_addressed",
        "input_truncation_policy": "preflight_block_no_extractor_truncation",
        "mode": SYNTHETIC_CONTRACT_TEST,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "protocol": protocol,
        "relevant_artifact_fingerprint": fingerprint,
        "repositories": {
            name: {
                "commit_identifies_tree": record["commit_identifies_tree"],
                "dirty": record["dirty"],
                "head": record["head"],
            }
            for name, record in revisions.items()
        },
        "serialization_id": config.serialization_id,
        "shell": {"id": shell.SHELL_ID, "sha256": shell_sha},
        "slot_sha256": slot_hashes,
        "timeout_enforcement": "simulated_exception_only_pending_exp_3b",
        "tokenizer": {
            "id": config.tokenizer_id,
            "revision": config.tokenizer_revision,
        },
    }
    scientific_id = campaign_scientific_id(identity)
    attempts: list[AttemptRecord] = []
    states: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    submitted = config.submitted_ids
    for condition in config.conditions:
        for document in config.cohort:
            if submitted is not None and document.document_id not in submitted:
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
                states.append(
                    _final_state(
                        scientific_id,
                        condition,
                        document,
                        "not_attempted",
                        (),
                        0,
                    )
                )
                continue
            pair_attempts, final_state, prediction = _run_pair(
                config,
                condition,
                document,
                shell_sha,
                slot_hashes[condition],
                scientific_id,
            )
            attempts.extend(pair_attempts)
            states.append(final_state)
            if prediction is not None:
                prediction_records.append(prediction)
    contract_abort = _export_would_abort(prediction_records)
    if contract_abort:
        for state in states:
            if state["principal_state"] == "completed":
                diagnostics = dict(state.get("diagnostics") or {})
                diagnostics["evaluation_contract_abort"] = True
                state["diagnostics"] = diagnostics
    coverage = coverage_report(config.cohort, config.conditions, states)
    result = {
        "attempts": [item.to_dict() for item in attempts],
        "coverage": coverage,
        "document_states": states,
        "freeze_gaps": freeze_gaps,
        "manifest": {
            "campaign_scientific_id": scientific_id,
            "identity": identity,
            "identity_excludes": ["duration", "local_paths", "timestamps"],
            "observed_readiness": readiness,
            "operational": {
                "created_at": None,
                "dependency_resolution": {
                    name: value["method"] for name, value in locations.items()
                },
                "output_dir": None if config.output_dir is None else str(config.output_dir),
            },
            "readiness_mutated": False,
        },
        "prediction_records": prediction_records,
        "scientific_id": scientific_id,
    }
    if config.output_dir is None:
        with TemporaryDirectory() as temporary:
            result["evaluation_summary"] = _publish(
                Path(temporary),
                result,
                config.synthetic_gold,
                contract_abort,
            )
    else:
        result["evaluation_summary"] = _publish(
            config.output_dir,
            result,
            config.synthetic_gold,
            contract_abort,
        )
    return result


def _publish(
    output_dir: Path,
    result: dict[str, Any],
    gold: tuple[Mapping[str, Any], ...] | None,
    contract_abort: bool,
) -> dict[str, Any]:
    write_campaign_files(output_dir, result)
    export_evaluation(
        output_dir / "predictions.jsonl",
        output_dir / "evaluation_predictions.jsonl",
    )
    exported = [
        json.loads(line)
        for line in (output_dir / "evaluation_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = evaluation_summary(
        gold,
        exported,
        contract_abort,
        result["prediction_records"],
    )
    (output_dir / "synthetic_evaluation_summary.json").write_text(
        canonical_json(summary) + "\n",
        encoding="utf-8",
    )
    return summary


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
    raise CampaignError("EXP.3a only authorizes SYNTHETIC_CONTRACT_TEST")


def _run_pair(
    config: CampaignConfig,
    condition: str,
    document: SyntheticDocument,
    shell_sha: str,
    slot_sha: str,
    scientific_id: str,
) -> tuple[list[AttemptRecord], dict[str, Any], dict[str, Any] | None]:
    prompt = render_controlled_prompt(condition, document.text)
    prompt_sha = sha256_text(prompt)
    input_sha = sha256_text(document.text)
    attempts: list[AttemptRecord] = []
    completed_validation = None
    for attempt_index in (1, 2):
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
        if preflight["assurance"] == "not_assessed":
            raise PreflightNotAssessed(str(preflight["reason"]))
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
        context = ExtractionContext(
            str(attempt_index),
            document.document_id,
            None,
            condition,
        )
        try:
            raw = config.extractor.extract(prompt, context)
        except ScriptMissing:
            raise
        except SimulatedInfrastructureError as exc:
            attempts.append(
                _failure_attempt("external_infrastructure_error", exc, common)
            )
        except TimeoutError as exc:
            attempts.append(_failure_attempt("timeout", exc, common))
        except Exception as exc:
            attempts.append(_failure_attempt("model_execution_error", exc, common))
        else:
            if config.extractor.calls[-1][3] != prompt:
                raise CampaignError("extractor received a prompt that is not the rendered shell")
            attempt, validated = _completed_attempt(
                raw.model_output,
                raw.generation_metadata,
                common,
            )
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
    if final.principal_state == "completed":
        if completed_validation is None:
            raise CampaignError("completed attempt has no validated triples")
        prediction = _prediction_record(
            scientific_id,
            condition,
            document,
            completed_validation,
        )
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
        ),
        prediction,
    )


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
    principal_state: str,
    earlier: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    attempt_count: int,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_count": attempt_count,
        "campaign_scientific_id": scientific_id,
        "condition": condition,
        "document_id": document.document_id,
        "example_id": document.resolved_example_id,
        "principal_state": principal_state,
    }
    if diagnostics:
        record["diagnostics"] = dict(diagnostics)
    if earlier:
        record["also_observed"] = list(earlier)
    return record


def _export_would_abort(records: list[dict[str, Any]]) -> bool:
    for record in records:
        for triple in record["validated"]["triples"]:
            for field_name in ("subject", "relation", "object"):
                value = triple.get(field_name)
                if not isinstance(value, str) or value.strip() == "":
                    return True
    return False


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


def _relevant_fingerprint(shell_text: str, protocol_sha: str | None) -> str:
    digest_source = [shell_text, protocol_sha or ""]
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        digest_source.append(path.read_text(encoding="utf-8"))
    return sha256_text("\n".join(digest_source))


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
