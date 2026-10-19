"""Versioned campaign records for the controlled experiment.

``exp3b1-1`` replaces ``exp3a-1``. Scientific configuration identity still
excludes local paths, durations, and timestamps. Implementation content now
has its own fingerprint, and P0 scores are read only from per-condition files
produced by ``export_evaluation``. A pooled five-condition file is not a
contrast and is not written as a scoring source.

Harness incidents are not principal states of the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


CAMPAIGN_CONTRACT_VERSION = "exp3b1-1"
PREVIOUS_CAMPAIGN_CONTRACT_VERSION = "exp3a-1"
STATUS_PUBLISHABLE = "operationally_complete_publishable"
STATUS_INCOMPLETE = "incomplete_external_or_interrupted"
STATUS_PUBLICATION_BLOCKED = "finished_publication_blocked"
STATUS_ABORTED_CONTRACT = "aborted_contract"
STATUS_ABORTED_CONFIGURATION = "aborted_configuration"
STATUS_IN_PROGRESS = "in_progress"
SYNTHETIC_CONTRACT_TEST = "SYNTHETIC_CONTRACT_TEST"
SHELL_ID_REQUIRED = "common-instruction-shell-v2"
REQUIRED_CONDITIONS = ("C0", "C1", "C2", "C3", "C3-NL")
PRINCIPAL_STATES = (
    "not_attempted",
    "context_blocked",
    "external_infrastructure_error",
    "timeout",
    "model_execution_error",
    "completed",
)
RETRYABLE_STATES = frozenset({"external_infrastructure_error", "timeout"})
REAL_C3_CONDITIONS = frozenset({"C3", "C3-NL"})
SYNTHETIC_COUNTER_ID = "synthetic-character-counter-v1"


class CampaignError(RuntimeError):
    """The campaign refused to proceed under the EXP.3a authorization."""


class RealC3CampaignRejected(CampaignError):
    """A non-synthetic mode tried to run C3 or C3-NL."""


class PreflightNotAssessed(CampaignError):
    """No verifiable tokenizer or context capacity was available."""


class ExternalInfrastructureError(Exception):
    """Explicit external failure. Retryable. This is not an OSError subclass."""


class HarnessPairError(Exception):
    """The harness failed on one pair. This is not a model failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        prompt_sha256: str | None = None,
        input_sha256: str | None = None,
        slot_sha256: str | None = None,
        shell_sha256: str | None = None,
    ) -> None:
        self.code = code
        self.prompt_sha256 = prompt_sha256
        self.input_sha256 = input_sha256
        self.slot_sha256 = slot_sha256
        self.shell_sha256 = shell_sha256
        super().__init__(message)


class ScriptMissing(LookupError):
    """The synthetic extractor has no script for this attempt."""


class SimulatedInfrastructureError(ExternalInfrastructureError):
    """Synthetic stand-in for a transient external failure. Not a model error."""


class SimulatedTimeout(TimeoutError):
    """Synthetic stand-in for a timeout. Real process limits are not enforced."""


class SimulatedModelError(RuntimeError):
    """Synthetic stand-in for an extractor failure after the call starts."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_payload(value: Any) -> str:
    return sha256_text(canonical_json(value))


def campaign_scientific_id(payload: Mapping[str, Any]) -> str:
    return "campaign-" + sha256_payload(payload)[:20]


def omit_absent(record: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None. False is retained only when stored on purpose."""
    return {key: value for key, value in record.items() if value is not None}


@dataclass(frozen=True)
class AttemptRecord:
    condition: str
    document_id: str
    example_id: str
    attempt_index: int | None
    principal_state: str | None
    harness_incident: str | None = None
    harness_message: str | None = None
    prompt_sha256: str | None = None
    slot_sha256: str | None = None
    shell_sha256: str | None = None
    input_sha256: str | None = None
    submitted: bool = True
    preflight: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    model_output: str | None = None
    parse_issues: tuple[Mapping[str, Any], ...] = ()
    exception_type: str | None = None
    exception_message: str | None = None
    seed: int | None = None
    reserved_output_tokens: int | None = None
    effective_context_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return omit_absent(
            {
                "attempt_index": self.attempt_index,
                "condition": self.condition,
                "diagnostics": dict(self.diagnostics) or None,
                "document_id": self.document_id,
                "effective_context_limit": self.effective_context_limit,
                "example_id": self.example_id,
                "exception_message": self.exception_message,
                "exception_type": self.exception_type,
                "harness_incident": self.harness_incident,
                "harness_message": self.harness_message,
                "input_sha256": self.input_sha256,
                "model_output": self.model_output,
                "parse_issues": [dict(item) for item in self.parse_issues] or None,
                "preflight": None if self.preflight is None else dict(self.preflight),
                "principal_state": self.principal_state,
                "prompt_sha256": self.prompt_sha256,
                "reserved_output_tokens": self.reserved_output_tokens,
                "seed": self.seed,
                "shell_sha256": self.shell_sha256,
                "slot_sha256": self.slot_sha256,
                "submitted": self.submitted,
            }
        )
