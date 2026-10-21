"""Context-window preflight for a fully rendered prompt.

The check is:

    required_tokens = prompt_tokens + reserved_output_tokens + special_token_allowance
    fits = required_tokens <= effective_context_limit

``prompt_tokens`` must already include a chat template or other serialization
when that serialization is what the backbone would send. ``special_token_allowance``
covers markers the supplied counter does not count. This function never
truncates the prompt and never calls an extractor.

A missing tokenizer identity, counter, or context limit yields
``assurance=not_assessed``. That result is not a pass. The synthetic character
counter is labeled ``synthetic_contract`` and is not evidence that a real
model context can hold the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from re_te_system.controlled_experiment.contracts import (
    SYNTHETIC_COUNTER_ID,
    omit_absent,
    sha256_text,
)


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class PreflightRequest:
    prompt: str
    count_tokens: TokenCounter | None
    effective_context_limit: int | None
    reserved_output_tokens: int
    tokenizer_id: str | None
    tokenizer_revision: str | None = None
    serialization_id: str | None = None
    special_token_allowance: int = 0
    counter_id: str | None = None
    measurement_scope: str | None = None

    def __post_init__(self) -> None:
        if self.reserved_output_tokens < 0 or self.special_token_allowance < 0:
            raise ValueError("token reserves must be non-negative")
        if self.effective_context_limit is not None and self.effective_context_limit < 0:
            raise ValueError("effective_context_limit must be non-negative")
        if self.measurement_scope not in (None, "test_only", "tokenizer_only"):
            raise ValueError(
                "measurement_scope must be omitted, 'test_only', or 'tokenizer_only'"
            )


def assess_preflight(request: PreflightRequest) -> dict[str, object]:
    prompt_sha256 = sha256_text(request.prompt)
    base: dict[str, object] = {
        "counter_id": request.counter_id,
        "effective_context_limit": request.effective_context_limit,
        "prompt_sha256": prompt_sha256,
        "reserved_output_tokens": request.reserved_output_tokens,
        "measurement_scope": request.measurement_scope,
        "serialization_id": request.serialization_id,
        "special_token_allowance": request.special_token_allowance,
        "tokenizer_id": request.tokenizer_id,
        "tokenizer_revision": request.tokenizer_revision,
    }
    missing = _missing(request)
    if missing:
        return omit_absent(
            {
                **base,
                "assurance": "not_assessed",
                "fits": None,
                "missing": missing,
                "reason": "preflight_not_assessed:" + ",".join(missing),
            }
        )
    assert request.count_tokens is not None
    assert request.effective_context_limit is not None
    prompt_tokens = request.count_tokens(request.prompt)
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens < 0:
        raise ValueError("token counter must return a non-negative integer")
    required = prompt_tokens + request.reserved_output_tokens + request.special_token_allowance
    fits = required <= request.effective_context_limit
    assurance = (
        "synthetic_contract"
        if request.counter_id == SYNTHETIC_COUNTER_ID
        or request.tokenizer_id == SYNTHETIC_COUNTER_ID
        else "measured"
    )
    if request.measurement_scope == "test_only":
        assurance = "test_only_not_model_measurement"
    elif request.measurement_scope == "tokenizer_only":
        assurance = "tokenizer_measured_not_model_ready"
    reason = "fits" if fits else "required_tokens_exceed_effective_context_limit"
    measured: dict[str, object] = {
        **base,
        "assurance": assurance,
        "fits": fits,
        "model_compatibility": False,
        "prompt_tokens": prompt_tokens,
        "reason": reason,
        "required_tokens": required,
    }
    if request.measurement_scope == "tokenizer_only":
        measured["gpu_ready"] = False
        measured["model_ready"] = False
    result = omit_absent(measured)
    if request.measurement_scope == "tokenizer_only":
        result["approved_effective_context_limit"] = None
    return result


def _missing(request: PreflightRequest) -> list[str]:
    missing: list[str] = []
    if not request.tokenizer_id:
        missing.append("tokenizer_id")
    if request.count_tokens is None or not request.counter_id:
        missing.append("token_counter")
    if request.effective_context_limit is None:
        missing.append("effective_context_limit")
    return missing


def preflight_public_view(result: Mapping[str, object]) -> dict[str, object]:
    return dict(result)
