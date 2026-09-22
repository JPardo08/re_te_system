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


@dataclass
class MockRebelExtractor:
    """REBEL-shaped deterministic fixture using the same extractor protocol."""

    model_name: str = "mock/rebel-contract"
    model_revision: str = "mock-rebel-v1"
    outputs: dict[str, str] = field(default_factory=dict)
    fail_on: set[str] = field(default_factory=set)

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        if context.input_id in self.fail_on:
            raise RuntimeError("configured mock REBEL failure")
        words = text.split()
        subject = words[0] if words else ""
        object_ = words[-1].rstrip(".,;:") if words else ""
        output = self.outputs.get(
            context.input_id,
            f"<triplet> {subject} <subj> {object_} <obj> related to",
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


@dataclass
class MockPythiaExtractor:
    """Pythia-shaped fixture with exact Turtle output and no ML dependencies."""

    model_name: str = "mock/pythia-spacekbp-contract"
    model_revision: str = "mock-pythia-v1"
    outputs: dict[str, str] = field(default_factory=dict)
    fail_on: set[str] = field(default_factory=set)

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        if context.input_id in self.fail_on:
            raise RuntimeError("configured mock Pythia failure")
        output = self.outputs.get(
            context.input_id,
            (
                "<https://example.org/mission> "
                "<https://example.org/hasDescription> "
                f'"{text}" .'
            ),
        )
        return RawExtractionResult(
            model_output=output,
            generation_metadata={
                "deterministic": True,
                "effective_input_limit": 1536,
                "input_token_count": len(text.split()),
                "max_new_tokens": 512,
                "model_max_length": 2048,
                "number_of_sequences": 1,
                "output_reached_limit": False,
                "output_token_count": len(output.split()),
                "prompt_profile": "basic",
                "prompt_profile_version": "basic-v1",
                "quantization": "none",
                "truncated_input": False,
            },
        )


@dataclass
class MockUIEExtractor:
    """UIE-shaped fixture preserving exact SEL output without model weights."""

    model_name: str = "mock/uie-contract"
    model_revision: str = "mock-uie-v1"
    outputs: dict[str, str] = field(default_factory=dict)
    fail_on: set[str] = field(default_factory=set)

    def extract(self, text: str, context: ExtractionContext) -> RawExtractionResult:
        if context.input_id in self.fail_on:
            raise RuntimeError("configured mock UIE failure")
        output = self.outputs.get(
            context.input_id,
            "<pad><extra_id_0><extra_id_1></s>",
        )
        return RawExtractionResult(
            model_output=output,
            generation_metadata={
                "constraint_decoding": False,
                "decoded_with_special_tokens": True,
                "deterministic": True,
                "effective_input_limit": 256,
                "input_token_count": len(text.split()),
                "max_target_tokens": 192,
                "model_max_length": 512,
                "number_of_sequences": 1,
                "output_reached_limit": False,
                "output_token_count": len(output.split()),
                "schema_hash": "mock-schema-hash",
                "schema_id": "mock-uie-schema",
                "ssi_version": "uie-ssi-v1",
                "truncated_input": False,
            },
        )
