"""Versioned campaign records for the controlled experiment.

Scientific identity excludes local paths, durations, and timestamps. Those
belong in operational metadata and must not change a campaign identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


CAMPAIGN_CONTRACT_VERSION = "exp3a-1"
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


class ScriptMissing(LookupError):
    """The synthetic extractor has no script for this attempt."""


class SimulatedInfrastructureError(RuntimeError):
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
    principal_state: str
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
