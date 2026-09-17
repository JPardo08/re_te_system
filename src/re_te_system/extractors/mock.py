"""Deterministic dependency-free extractor for contract tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from re_te_system.contracts import ExtractionContext, RawExtractionResult


@dataclass
class MockExtractor:
    model_name: str = "mock/mrebel-contract"
    model_revision: str = "mock-v1"
    outputs: dict[str, str] = field(default_factory=dict)
    fail_on: set[str] = field(default_factory=set)

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        if context.input_id in self.fail_on:
            raise RuntimeError("configured mock failure")
        words = text.split()
        subject = words[0] if words else ""
        object_ = words[-1].rstrip(".,;:") if words else ""
        output = self.outputs.get(
            context.input_id,
            f"<triplet> {subject} <concept> {object_} <concept> related_to",
        )
        return RawExtractionResult(
            model_output=output,
            generation_metadata={
                "deterministic": True,
                "finish_reason": "mock_complete",
                "input_token_count": len(words),
                "number_of_sequences": 1,
                "output_token_count": len(output.split()),
                "truncated": False,
            },
        )
