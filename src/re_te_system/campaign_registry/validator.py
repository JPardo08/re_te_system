"""Cross-catalog checks. No model loading and no job submission."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from re_te_system.campaign_registry.contracts import (
    AUTHORIZATIONS,
    CLAIMING_STATES,
    COMPARABILITIES,
    EXECUTOR_KINDS,
    INVALID,
    LOCKED_CONDITIONS,
    NOT_READY,
    PLAN_STATES,
    PROGRAMS,
    REGISTRY_CONTRACT_VERSION,
    STRUCTURALLY_VALID,
    schema_errors,
)
from re_te_system.campaign_registry.loader import load_catalog


def validate_catalog(catalog_root: Path) -> dict[str, Any]:
    """Validate the on-disk registry and check that cited batch files exist."""
    documents = load_catalog(catalog_root)
    system_root = Path(catalog_root).resolve().parents[1]
    return validate_documents(documents, path_root=system_root)


def validate_documents(
    documents: Mapping[str, Any],
    path_root: Path | None = None,
) -> dict[str, Any]:
    """Validate an in-memory catalog. ``path_root`` enables batch-file checks."""
    errors: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    systems = _index(documents.get("systems", []), "system", errors)
    datasets = _index(documents.get("datasets", []), "dataset", errors)
    protocols = _index(documents.get("protocols", []), "protocol", errors)
    profiles = _index(documents.get("profiles", []), "profile", errors)
    tasks = _task_index(documents.get("tasks", []), errors)
    _validate_tasks(
        tasks,
        systems,
        datasets,
        protocols,
        profiles,
        documents.get("condition_policy", {}),
        documents.get("task_schema", {}),
        path_root,
        errors,
        gaps,
    )
    _validate_entries(
        documents.get("leaderboard_entries", []),
        protocols,
        datasets,
        documents.get("leaderboard_schema", {}),
        errors,
    )
    return _report(documents, tasks, errors, gaps)


def comparison_key(entry: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Build the only key that can put two rows in one comparison group."""
    fields = (
        "program",
        "protocol_id",
        "dataset_id",
        "split",
        "language",
        "task",
        "reference_ontology",
        "prediction_ontology",
        "ontology_alignment",
        "matching",
        "normalization",
        "eligibility_policy",
        "estimand",
    )
    return tuple((field, "" if entry.get(field) is None else str(entry.get(field))) for field in fields)


def entries_rankable(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two rows may be read as the same comparison.

    Equal keys are required. Diagnostic rows and non-comparable rows never
    join a scientific group. The result does not order either row by score.
    """
    if comparison_key(left) != comparison_key(right):
        return False
    if left.get("comparability") != "scientific_comparable":
        return False
    if right.get("comparability") != "scientific_comparable":
        return False
    return True


def _validate_tasks(
    tasks: dict[str, dict[str, Any]],
    systems: dict[str, dict[str, Any]],
    datasets: dict[str, dict[str, Any]],
    protocols: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    condition_policy: Any,
    task_schema: Any,
    path_root: Path | None,
    errors: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> None:
    policy = condition_policy if isinstance(condition_policy, dict) else {}
    graph = {
        task_id: [item for item in task.get("dependencies", []) if isinstance(item, str)]
        for task_id, task in tasks.items()
    }
    cycle = _cycle(graph)
    if cycle:
        _error(errors, "dependency_cycle", " -> ".join(cycle), None)
    for task_id, task in tasks.items():
        if isinstance(task_schema, dict) and task_schema:
            for detail in schema_errors(task, task_schema, task_id):
                _error(errors, "schema", detail, task_id)
        _validate_task(
            task_id,
            task,
            tasks,
            systems,
            datasets,
            protocols,
            profiles,
            policy,
            path_root,
            errors,
            gaps,
        )


def _validate_task(
    task_id: str,
    task: Mapping[str, Any],
    tasks: Mapping[str, Mapping[str, Any]],
    systems: Mapping[str, Mapping[str, Any]],
    datasets: Mapping[str, Mapping[str, Any]],
    protocols: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    path_root: Path | None,
    errors: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> None:
    status = task.get("preparation_status")
    if status not in PLAN_STATES:
        _error(errors, "unknown_plan_state", str(status), task_id)
    if task.get("program") not in PROGRAMS:
        _error(errors, "unknown_program", str(task.get("program")), task_id)
    if task.get("authorization") not in AUTHORIZATIONS:
        _error(errors, "unknown_authorization", str(task.get("authorization")), task_id)
    if task.get("execution_permitted") is not False:
        _error(
            errors,
            "execution_not_grantable_by_registry",
            "this registry cannot set execution_permitted",
            task_id,
        )
    criteria = task.get("acceptance_criteria")
    if not isinstance(criteria, list) or not criteria or not all(
        isinstance(item, str) and item.strip() for item in criteria
    ):
        _error(errors, "missing_acceptance_criteria", "acceptance_criteria is empty", task_id)
    leaderboard_class = task.get("leaderboard_class")
    if leaderboard_class is not None and leaderboard_class not in COMPARABILITIES:
        _error(errors, "unknown_comparability", str(leaderboard_class), task_id)
    reasons = task.get("block_reasons", [])
    if not isinstance(reasons, list):
        _error(errors, "schema", "block_reasons must be an array", task_id)
        reasons = []
    if status == "BLOCKED" and not reasons:
        _error(errors, "blocked_without_evidence", "BLOCKED requires block_reasons", task_id)
    if status == "PREPARED" and reasons:
        _error(errors, "prepared_while_blocked", "PREPARED cannot carry block_reasons", task_id)
    for dependency in task.get("dependencies", []):
        if dependency not in tasks:
            _error(errors, "missing_dependency", str(dependency), task_id)
    _check_reference(task.get("system_id"), systems, "missing_system", task_id, errors)
    dataset_id = task.get("dataset_id")
    dataset = _check_reference(dataset_id, datasets, "missing_dataset", task_id, errors)
    protocol = _check_reference(
        task.get("protocol_id"), protocols, "missing_protocol", task_id, errors
    )
    profile = _check_reference(
        task.get("resource_profile_id"), profiles, "missing_profile", task_id, errors
    )
    if dataset is not None:
        _check_dataset_use(task_id, task, dataset, errors)
    if protocol is not None and leaderboard_class == "scientific_comparable":
        if protocol.get("comparability") != "scientific_comparable":
            _error(
                errors,
                "incompatible_leaderboard_class",
                "task claims a scientific row for a non-scientific protocol",
                task_id,
            )
    system = systems.get(task.get("system_id")) if isinstance(task.get("system_id"), str) else None
    if system is not None:
        _check_system_link(task_id, task, system, protocol, errors, gaps)
    if isinstance(profile, dict):
        _check_profile(task_id, task, profile, errors, gaps)
    condition = task.get("condition")
    if isinstance(condition, str):
        _check_condition(task_id, status, condition, policy, errors)
    _check_executor(task_id, task, path_root, errors)
    if status in CLAIMING_STATES:
        _claim_gaps_are_errors(task_id, task, system, profile, errors)
    else:
        _record_operational_gaps(task_id, task, system, profile, gaps)


def _check_dataset_use(
    task_id: str,
    task: Mapping[str, Any],
    dataset: Mapping[str, Any],
    errors: list[dict[str, Any]],
) -> None:
    split = task.get("split")
    splits = dataset.get("splits", [])
    split_ids = {
        item.get("id")
        for item in splits
        if isinstance(item, dict)
    }
    if split is not None and split_ids and split not in split_ids:
        _error(errors, "unknown_split", str(split), task_id)
    if task.get("training") is True:
        prohibited = dataset.get("prohibited_uses", [])
        if dataset.get("evaluation_only") is True or "training" in prohibited:
            _error(
                errors,
                "gold_training_prohibited",
                f"{dataset.get('id')} cannot be used for training",
                task_id,
            )
    if task.get("program") == "CONDITIONING" and dataset.get("independent_cohort") is False:
        _error(errors, "projection_used_as_cohort", str(dataset.get("id")), task_id)
    if task.get("program") == "CONDITIONING" and dataset.get("synthetic") is True:
        _error(errors, "synthetic_conditioning_cohort", str(dataset.get("id")), task_id)


def _check_system_link(
    task_id: str,
    task: Mapping[str, Any],
    system: Mapping[str, Any],
    protocol: Mapping[str, Any] | None,
    errors: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> None:
    programs = system.get("programs", [])
    if task.get("program") not in programs:
        _error(errors, "system_program_mismatch", str(system.get("id")), task_id)
    parser = system.get("parser")
    if task.get("evaluation") is not None and not parser:
        _error(errors, "missing_parser", "scored task has no parser", task_id)
    if protocol is not None:
        expected_parser = protocol.get("parser")
        if expected_parser and parser and expected_parser != parser:
            _error(
                errors,
                "parser_protocol_mismatch",
                f"{parser} != {expected_parser}",
                task_id,
            )
    identity = system.get("checkpoint_identity", {})
    if task.get("requires_checkpoint_identity") is True and not _identity_present(identity):
        _gap(gaps, "checkpoint_identity_unresolved", task_id, str(system.get("id")))
    if task.get("requires_tokenizer") is True and not system.get("tokenizer_revision"):
        _gap(gaps, "tokenizer_unverified", task_id, str(system.get("id")))
    if task.get("requires_real_preflight") is True and task.get("real_preflight_done") is not True:
        _gap(gaps, "real_preflight_missing", task_id, str(system.get("id")))
    if task.get("requires_no_truncation") is True and system.get("truncates_input") is not False:
        _gap(gaps, "truncation_guard_absent", task_id, str(system.get("id")))
    if (
        task.get("requires_frozen_output_budget") is True
        and task.get("output_budget_frozen") is not True
    ):
        _gap(gaps, "output_budget_unfrozen", task_id, str(system.get("id")))
    if (
        task.get("requires_channel_separator") is True
        and system.get("channel_separator_verified") is not True
    ):
        _gap(gaps, "channel_separator_unverified", task_id, str(system.get("id")))


def _check_profile(
    task_id: str,
    task: Mapping[str, Any],
    profile: Mapping[str, Any],
    errors: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> None:
    if profile.get("enabled") is False:
        _error(errors, "profile_disabled", str(profile.get("id")), task_id)
    gpu_count = profile.get("gpu_count") or 0
    if gpu_count and profile.get("vram") is None:
        _gap(gaps, "vram_unverified", task_id, str(profile.get("id")))
    if profile.get("id") == "unassigned":
        _gap(gaps, "resource_profile_unassigned", task_id, "no CESVIMA profile selected")
    resources = task.get("resources", {})
    if isinstance(resources, dict) and resources.get("vram") is not None:
        _error(errors, "invented_vram", "task VRAM must stay null until measured", task_id)


def _check_condition(
    task_id: str,
    status: Any,
    condition: str,
    policy: Mapping[str, Any],
    errors: list[dict[str, Any]],
) -> None:
    record = policy.get(condition)
    if not isinstance(record, dict):
        _error(errors, "unknown_condition", condition, task_id)
        return
    executable = record.get("scientifically_executable") is True
    if condition in LOCKED_CONDITIONS and status != "BLOCKED":
        _error(errors, "condition_must_remain_blocked", condition, task_id)
    if not executable and status in CLAIMING_STATES:
        _error(errors, "condition_not_authorized", condition, task_id)


def _check_executor(
    task_id: str,
    task: Mapping[str, Any],
    path_root: Path | None,
    errors: list[dict[str, Any]],
) -> None:
    executor = task.get("executor")
    if not isinstance(executor, dict):
        _error(errors, "schema", "executor must be an object", task_id)
        return
    kind = executor.get("kind")
    if kind not in EXECUTOR_KINDS:
        _error(errors, "unknown_executor", str(kind), task_id)
        return
    path = executor.get("path")
    if kind == "existing_sbatch":
        if not isinstance(path, str) or not path:
            _error(errors, "missing_executor_path", "existing batch path is empty", task_id)
        elif path_root is not None and not (path_root / path).is_file():
            _error(errors, "missing_executor_path", path, task_id)
    elif path not in (None, ""):
        _error(errors, "unexpected_executor_path", str(path), task_id)


def _claim_gaps_are_errors(
    task_id: str,
    task: Mapping[str, Any],
    system: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
    errors: list[dict[str, Any]],
) -> None:
    if system is not None and task.get("requires_checkpoint_identity") is True:
        if not _identity_present(system.get("checkpoint_identity", {})):
            _error(errors, "checkpoint_sha_unknown", str(system.get("id")), task_id)
    if system is not None and task.get("requires_tokenizer") is True:
        if not system.get("tokenizer_revision"):
            _error(errors, "tokenizer_unverified", str(system.get("id")), task_id)
    if task.get("requires_real_preflight") is True and task.get("real_preflight_done") is not True:
        _error(errors, "real_preflight_missing", task_id, task_id)
    if (
        system is not None
        and task.get("requires_channel_separator") is True
        and system.get("channel_separator_verified") is not True
    ):
        _error(errors, "channel_separator_unverified", str(system.get("id")), task_id)
    if profile is not None and (profile.get("gpu_count") or 0) and profile.get("vram") is None:
        _error(errors, "vram_unverified", str(profile.get("id")), task_id)
    if task.get("authorization") != "authorized":
        _error(errors, "authorization_missing", str(task.get("authorization")), task_id)


def _record_operational_gaps(
    task_id: str,
    task: Mapping[str, Any],
    system: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
    gaps: list[dict[str, Any]],
) -> None:
    del system, profile
    if task.get("authorization") != "authorized":
        _gap(gaps, "authorization_missing", task_id, str(task.get("authorization")))


def _identity_present(identity: Any) -> bool:
    if not isinstance(identity, dict):
        return False
    kind = identity.get("kind")
    value = identity.get("value")
    if kind not in {"git_sha", "content_md5"}:
        return False
    return isinstance(value, str) and bool(value.strip())


def _validate_entries(
    entries: Any,
    protocols: Mapping[str, Mapping[str, Any]],
    datasets: Mapping[str, Mapping[str, Any]],
    schema: Any,
    errors: list[dict[str, Any]],
) -> None:
    if not isinstance(entries, list):
        _error(errors, "schema", "leaderboard entries must be an array", None)
        return
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            _error(errors, "schema", "leaderboard entry must be an object", None)
            continue
        entry_id = entry.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            _error(errors, "schema", "entry_id is required", None)
            continue
        if entry_id in seen:
            _error(errors, "duplicate_entry_id", entry_id, None)
        seen.add(entry_id)
        if isinstance(schema, dict) and schema:
            for detail in schema_errors(entry, schema, entry_id):
                _error(errors, "schema", detail, entry_id)
        comparability = entry.get("comparability")
        if comparability not in COMPARABILITIES:
            _error(errors, "unknown_comparability", str(comparability), entry_id)
        protocol = protocols.get(entry.get("protocol_id"))
        dataset = datasets.get(entry.get("dataset_id"))
        synthetic = entry.get("protocol_id") == "SYNTHETIC_CONTRACT" or (
            isinstance(dataset, dict) and dataset.get("synthetic") is True
        )
        if synthetic and comparability == "scientific_comparable":
            _error(errors, "synthetic_in_scientific_ranking", entry_id, entry_id)
        if isinstance(protocol, dict):
            declared = protocol.get("comparability")
            if declared == "diagnostic_only" and comparability == "scientific_comparable":
                _error(errors, "diagnostic_promoted", entry_id, entry_id)
            if declared == "not_comparable" and comparability == "scientific_comparable":
                _error(errors, "not_comparable_promoted", entry_id, entry_id)
            allowed = protocol.get("estimands")
            if isinstance(allowed, list) and entry.get("estimand") not in allowed:
                _error(errors, "unknown_estimand", str(entry.get("estimand")), entry_id)
        if (
            comparability == "scientific_comparable"
            and _ontologies_disagree(entry)
            and not (isinstance(protocol, dict) and protocol.get("schema_mismatch") is True)
        ):
            _error(errors, "unmapped_ontology_ranked", entry_id, entry_id)


def _ontologies_disagree(entry: Mapping[str, Any]) -> bool:
    reference = entry.get("reference_ontology")
    prediction = entry.get("prediction_ontology")
    return reference is not None and prediction is not None and reference != prediction


def _report(
    documents: Mapping[str, Any],
    tasks: Mapping[str, Mapping[str, Any]],
    errors: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    structural = INVALID if errors else STRUCTURALLY_VALID
    invalid_tasks = {item["task_id"] for item in errors if item.get("task_id") in tasks}
    by_program = {"CONDITIONING": 0, "SOTA": 0}
    by_status: dict[str, int] = {state: 0 for state in PLAN_STATES}
    blocked: list[str] = []
    prepared = 0
    for task_id in sorted(tasks):
        task = tasks[task_id]
        program = task.get("program")
        if program in by_program:
            by_program[program] += 1
        status = task.get("preparation_status")
        if status in by_status:
            by_status[status] += 1
        if status == "BLOCKED":
            blocked.append(task_id)
        if status == "PREPARED":
            prepared += 1
    reason_codes = sorted(
        {
            item["code"]
            for item in gaps
        }
        | {
            reason.get("code")
            for task in tasks.values()
            for reason in task.get("block_reasons", [])
            if isinstance(reason, dict) and isinstance(reason.get("code"), str)
        }
        | {"execution_permitted_is_false"}
    )
    readiness = NOT_READY
    execution_granted = bool(tasks) and all(
        task.get("execution_permitted") is True for task in tasks.values()
    )
    if (
        structural == STRUCTURALLY_VALID
        and not gaps
        and not blocked
        and tasks
        and prepared == len(tasks)
        and execution_granted
    ):
        readiness = "OPERATIONAL_GAPS_NONE"
    overall = INVALID if structural == INVALID else NOT_READY if readiness == NOT_READY else STRUCTURALLY_VALID
    entries = documents.get("leaderboard_entries", [])
    groups: dict[str, list[str]] = {}
    if isinstance(entries, list):
        ordered = sorted(
            (entry for entry in entries if isinstance(entry, dict) and entry.get("entry_id")),
            key=lambda entry: str(entry.get("entry_id")),
        )
        for entry in ordered:
            key = "|".join(value for _, value in comparison_key(entry))
            groups.setdefault(key, []).append(str(entry.get("entry_id")))
    report = {
        "blocked_tasks": blocked,
        "catalog_contract": REGISTRY_CONTRACT_VERSION,
        "comparison_groups": [
            {"comparison_key": key, "entry_ids": groups[key]} for key in sorted(groups)
        ],
        "counts": {
            "blocked": len(blocked),
            "by_preparation_status": by_status,
            "by_program": by_program,
            "errors": len(errors),
            "gaps": len(gaps),
            "leaderboard_entries": len(entries) if isinstance(entries, list) else 0,
            "prepared": prepared,
            "structurally_valid_tasks": len(tasks) - len(invalid_tasks),
            "tasks": len(tasks),
        },
        "dependencies": [
            {
                "depends_on": sorted(
                    item for item in tasks[task_id].get("dependencies", []) if isinstance(item, str)
                ),
                "id": task_id,
            }
            for task_id in sorted(tasks)
        ],
        "errors": _sorted_findings(errors),
        "executed_external_commands": [],
        "gaps": _sorted_findings(gaps),
        "non_execution_reasons": reason_codes,
        "notes": sorted(item for item in documents.get("notes", []) if isinstance(item, str)),
        "overall_status": overall,
        "readiness_status": readiness if structural == STRUCTURALLY_VALID else INVALID,
        "structural_status": structural,
        "submit_all_is_not_authorization": documents.get("submit_all_is_not_authorization") is True,
    }
    return report


def _index(
    items: Any,
    label: str,
    errors: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        _error(errors, "schema", f"{label} catalog must be an array", None)
        return indexed
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            _error(errors, "schema", f"{label} needs a string id", None)
            continue
        item_id = item["id"]
        if item_id in indexed:
            _error(errors, "duplicate_id", f"{label}:{item_id}", None)
        indexed[item_id] = item
    return indexed


def _task_index(items: Any, errors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _index(items, "task", errors)


def _check_reference(
    identifier: Any,
    indexed: Mapping[str, Mapping[str, Any]],
    code: str,
    task_id: str,
    errors: list[dict[str, Any]],
) -> Mapping[str, Any] | None:
    if identifier is None:
        return None
    if not isinstance(identifier, str) or identifier not in indexed:
        _error(errors, code, str(identifier), task_id)
        return None
    return indexed[identifier]


def _cycle(graph: Mapping[str, list[str]]) -> list[str] | None:
    color: dict[str, int] = {}
    stack: list[str] = []

    def walk(node: str) -> list[str] | None:
        color[node] = 1
        stack.append(node)
        for nxt in graph.get(node, []):
            if nxt not in graph:
                continue
            state = color.get(nxt, 0)
            if state == 0:
                found = walk(nxt)
                if found:
                    return found
            elif state == 1:
                start = stack.index(nxt)
                return stack[start:] + [nxt]
        stack.pop()
        color[node] = 2
        return None

    for node in sorted(graph):
        if color.get(node, 0) == 0:
            found = walk(node)
            if found:
                return found
    return None


def _sorted_findings(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (item["code"], item["task_id"] or "", item["detail"]))


def _error(errors: list[dict[str, Any]], code: str, detail: str, task_id: str | None) -> None:
    errors.append({"code": code, "detail": detail, "task_id": task_id})


def _gap(gaps: list[dict[str, Any]], code: str, task_id: str, detail: str) -> None:
    gaps.append({"code": code, "detail": detail, "task_id": task_id})
