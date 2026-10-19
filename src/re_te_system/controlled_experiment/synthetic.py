"""Deterministic stand-ins for an extractor and a token counter.

Neither object calls a model. A scripted TSV string is not a model prediction.
Real process timeouts are not enforced here; a timeout is a raised
``SimulatedTimeout`` and remains pending for EXP.3b.
"""

from __future__ import annotations

from dataclasses import dataclass

from re_te_system.contracts import ExtractionContext, RawExtractionResult
from re_te_system.controlled_experiment.contracts import (
    SYNTHETIC_COUNTER_ID,
    ScriptMissing,
    SimulatedInfrastructureError,
    SimulatedModelError,
    SimulatedTimeout,
)


@dataclass(frozen=True)
class ScriptedTurn:
    """One predefined attempt. Only the supplied failure or output is used."""

    kind: str
    output: str | None = None
    output_reached_limit: bool | None = None
    truncated_input: bool | None = None
    message: str = "scripted synthetic failure"

    def __post_init__(self) -> None:
        if self.kind not in {"output", "infrastructure", "timeout", "model_error"}:
            raise ValueError(f"unknown scripted turn {self.kind!r}")
        if self.kind == "output" and self.output is None:
            raise ValueError("an output turn requires a string, which may be empty")
        if self.kind != "output" and self.output is not None:
            raise ValueError("a failure turn must not carry model output")


class SyntheticTokenCounter:
    """Count one token per Unicode character. This is not a model tokenizer."""

    counter_id = SYNTHETIC_COUNTER_ID
    tokenizer_id = SYNTHETIC_COUNTER_ID
    tokenizer_revision = "character-length-v1"

    def count_tokens(self, prompt: str) -> int:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        return len(prompt)


class ScriptedTsvExtractor:
    """Gold-blind extractor. Keys are ``(condition, document_id, attempt)``.

    ``condition`` may be ``*`` to share one script across the five arms.
    ``extract`` implements the system extractor signature and does not repair,
    truncate, or interpret the prompt.
    """

    model_name = "synthetic-tsv-script"
    model_revision = "exp3a-script-v1"

    def __init__(self, scripts: dict[tuple[str, str, int], ScriptedTurn]) -> None:
        self.scripts = dict(scripts)
        self.calls: list[tuple[str, str, int, str]] = []

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        if not isinstance(text, str):
            raise TypeError("extractor text must be a string")
        try:
            attempt = int(context.input_id)
        except (TypeError, ValueError) as exc:
            raise ScriptMissing("input_id must be the attempt index") from exc
        self.calls.append((context.condition, context.document_id, attempt, text))
        turn = self.scripts.get((context.condition, context.document_id, attempt))
        if turn is None:
            turn = self.scripts.get(("*", context.document_id, attempt))
        if turn is None:
            raise ScriptMissing(
                f"no script for {context.condition} {context.document_id} attempt {attempt}"
            )
        if turn.kind == "infrastructure":
            raise SimulatedInfrastructureError(turn.message)
        if turn.kind == "timeout":
            raise SimulatedTimeout(turn.message)
        if turn.kind == "model_error":
            raise SimulatedModelError(turn.message)
        metadata: dict[str, object] = {
            "deterministic": True,
            "do_sample": False,
            "prediction_source": "synthetic_script_not_model",
            "timeout_enforcement": "simulated_exception_only_pending_exp_3b",
        }
        if turn.output_reached_limit is True:
            metadata["output_reached_limit"] = True
        if turn.truncated_input is True:
            metadata["truncated_input"] = True
        return RawExtractionResult(turn.output or "", metadata)
