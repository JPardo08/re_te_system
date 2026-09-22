"""Stable scientific run identity and manifest construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from re_te_system.contracts import (
    PACKAGE_VERSION,
    PREDICTION_CONTRACT_VERSION,
    RUN_MANIFEST_VERSION,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            return "UNCOMMITTED"
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNCOMMITTED"


def scientific_identity(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"run-{digest[:20]}"


def build_manifest(
    *,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    condition: str,
    generation: Mapping[str, Any],
    windowing: Mapping[str, Any],
    benchmark_source: Mapping[str, Any],
    run_role: str = "baseline",
    model_family: str = "UNKNOWN",
    target_schema_knowledge: str = "none",
    native_schema: Mapping[str, Any] | None = None,
    input_language: str = "es",
    language_status: str = "UNKNOWN",
    task_class: str | None = None,
    controlled_experiment_condition: str | None = None,
    code_commit: str | None = None,
) -> dict[str, Any]:
    software = {
        "commit": code_commit or software_commit(),
        "package": "re_te_system",
        "version": PACKAGE_VERSION,
    }
    identity = {
        "benchmark_source": dict(benchmark_source),
        "condition": condition,
        "conditioning": "none",
        "dataset": dict(dataset),
        "generation": dict(generation),
        "input_language": input_language,
        "language_status": language_status,
        "model": dict(model),
        "model_family": model_family,
        "model_name": model.get("name", "UNKNOWN"),
        "model_revision": model.get("resolved_revision", "UNKNOWN"),
        "native_schema": dict(
            native_schema
            or {"id": "UNKNOWN", "inherited_from_model": True}
        ),
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "run_role": run_role,
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "software": software,
        "target_condition": {
            "id": condition,
            "target_schema_knowledge": target_schema_knowledge,
        },
        "target_schema_knowledge": target_schema_knowledge,
        "windowing": dict(windowing),
    }
    if task_class is not None:
        identity["task_class"] = task_class
    if controlled_experiment_condition is not None:
        identity["controlled_experiment_condition"] = controlled_experiment_condition
    return {
        "run_id": scientific_identity(identity),
        **identity,
        "operational_metadata": {"created_at": None},
    }
