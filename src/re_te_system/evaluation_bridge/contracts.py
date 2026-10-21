"""Contracts for the offline EXP.4.1 evaluation bridge.

The bridge reads a finished controlled-experiment directory and the benchmark
files it claims to score. It does not generate text, load a model, or choose
an estimand after seeing a score. P0 remains the only matcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BRIDGE_CONTRACT_VERSION = "exp4.1-1"
RESULT_ROLE_SYNTHETIC = "synthetic_contract_test_not_experimental_f1"
REGISTRY_PROTOCOL_SYNTHETIC = "SYNTHETIC_CONTRACT"
FROZEN_POLICY = {
    "context_mode": "document",
    "eligibility_policy": "strict_inline",
    "matching_policy": "exact_multiset_v1",
    "metric_family": "triplet",
    "normalization_policy": "unicode_nfc+outer_strip+lowercase_false",
}
ESTIMANDS = ("conservative", "conditional", "paired")


class BridgeError(RuntimeError):
    """The bridge could not write a report. Campaign defects are reported, not raised."""


@dataclass(frozen=True)
class BridgeRequest:
    """Explicit paths. Nothing is inferred from the process working directory."""

    campaign_dir: Path
    gold_path: Path
    output_dir: Path
    documents_path: Path | None = None
    benchmark_manifest_path: Path | None = None
    protocol_path: Path | None = None
    registry_protocols_path: Path | None = None
    expected_document_ids: tuple[str, ...] | None = None
