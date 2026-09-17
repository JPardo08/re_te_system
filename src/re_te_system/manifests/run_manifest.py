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
        "model": dict(model),
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "run_manifest_version": RUN_MANIFEST_VERSION,
        "software": software,
        "windowing": dict(windowing),
    }
    return {
        "run_id": scientific_identity(identity),
        **identity,
        "operational_metadata": {"created_at": None},
    }
