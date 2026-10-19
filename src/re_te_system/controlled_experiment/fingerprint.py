"""Content fingerprint of the harness and the components it executes.

Keys are logical labels. Absolute paths are not part of the hashed payload.
The scientific configuration identity is separate and is not computed here.
Relevant tests are an audit inventory, not part of this fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from re_te_system.controlled_experiment.contracts import sha256_payload, sha256_text


SYSTEM_FILES = (
    "src/re_te_system/parsing/tsv.py",
    "src/re_te_system/normalization/basic.py",
    "src/re_te_system/validation/structural.py",
    "src/re_te_system/runner/run.py",
)
EVALUATION_FILES = (
    "src/re_te_evaluation/eligibility.py",
    "src/re_te_evaluation/contracts.py",
    "src/re_te_evaluation/context.py",
    "src/re_te_evaluation/triplet/metrics.py",
    "src/re_te_evaluation/matching/exact.py",
    "src/re_te_evaluation/aggregation/classification.py",
)
AUDIT_TESTS = ("tests/test_controlled_experiment.py",)


def collect_implementation_parts(
    *,
    system_root: Path,
    benchmark_root: Path | None,
    evaluation_root: Path | None,
    shell_text: str,
    slots: Mapping[str, str],
    protocol_text: str | None,
) -> dict[str, str]:
    parts: dict[str, str] = {
        "re_te_benchmark/conditioning/shell-v2-template": shell_text,
        "re_te_benchmark/docs/CONTROLLED_EXPERIMENT_PROTOCOL.md": protocol_text or "",
    }
    for condition, text in sorted(slots.items()):
        parts[f"re_te_benchmark/conditioning/slot/{condition}"] = text
    package = system_root / "src" / "re_te_system" / "controlled_experiment"
    for path in sorted(package.glob("*.py")):
        label = f"re_te_system/src/re_te_system/controlled_experiment/{path.name}"
        parts[label] = path.read_text(encoding="utf-8")
    for relative in SYSTEM_FILES:
        parts[f"re_te_system/{relative}"] = _read(system_root / relative)
    for relative in EVALUATION_FILES:
        root = evaluation_root
        parts[f"re_te_evaluation/{relative}"] = "" if root is None else _read(root / relative)
    if benchmark_root is not None:
        payload = benchmark_root / "src" / "re_te_benchmark" / "conditioning" / "c2.py"
        parts["re_te_benchmark/src/re_te_benchmark/conditioning/c2.py"] = _read(payload)
    return parts


def implementation_fingerprint(
    parts: Mapping[str, str],
    repositories: Mapping[str, Mapping[str, object]],
) -> str:
    """Hash content and repository identity. Paths and timestamps are absent."""
    payload = {
        "content_sha256": {key: sha256_text(parts[key]) for key in sorted(parts)},
        "repositories": {
            name: {
                "commit_identifies_tree": record.get("commit_identifies_tree"),
                "dirty": record.get("dirty"),
                "head": record.get("head"),
            }
            for name, record in sorted(repositories.items())
        },
    }
    return sha256_payload(payload)


def audit_inventory(system_root: Path) -> dict[str, str]:
    inventory = {}
    for relative in AUDIT_TESTS:
        inventory[relative] = sha256_text(_read(system_root / relative))
    return inventory


def _read(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")
