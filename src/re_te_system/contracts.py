"""Versioned, model-independent extraction contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


PREDICTION_CONTRACT_VERSION = "1.0"
RUN_MANIFEST_VERSION = "1.0"
PACKAGE_VERSION = "0.1.0"


@dataclass(frozen=True)
class ExtractionContext:
    input_id: str
    document_id: str
    segment_id: str | None
    condition: str = "C0"
    segment_start: int | None = None
    segment_end: int | None = None

    def __post_init__(self) -> None:
        if not self.condition:
            raise ValueError("condition must be non-empty")


@dataclass(frozen=True)
class RawExtractionResult:
    model_output: str
    generation_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedTriple:
    subject: str
    relation: str
    object: str
    subject_type: str | None = None
    object_type: str | None = None
    subject_span: tuple[int, int] | None = None
    object_span: tuple[int, int] | None = None
    span_source: str | None = None
    segment_id: str | None = None
    subject_is_uri: bool | None = None
    predicate_is_uri: bool | None = None
    object_is_uri: bool | None = None
    object_is_literal: bool | None = None
    datatype: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class ParseIssue:
    code: str
    message: str
    severity: str = "soft"
    chunk_index: int | None = None


@dataclass(frozen=True)
class ParseResult:
    triples: tuple[ParsedTriple, ...]
    issues: tuple[ParseIssue, ...] = ()


@dataclass(frozen=True)
class Violation:
    code: str
    severity: str
    message: str
    triple_index: int | None = None


@dataclass(frozen=True)
class ValidatedResult:
    triples: tuple[ParsedTriple, ...]
    violations: tuple[Violation, ...]
    status: str


@dataclass(frozen=True)
class InputRecord:
    example_id: str
    document_id: str
    text: str

    def extractor_payload(self) -> dict[str, str]:
        """The complete Gold-blind payload available to extraction."""
        return {
            "example_id": self.example_id,
            "document_id": self.document_id,
            "text": self.text,
        }


@dataclass(frozen=True)
class Segment:
    segment_id: str | None
    text: str
    start: int
    end: int


def triple_to_dict(triple: ParsedTriple) -> dict[str, Any]:
    value = asdict(triple)
    if value["subject_span"] is not None:
        value["subject_span"] = list(value["subject_span"])
    if value["object_span"] is not None:
        value["object_span"] = list(value["object_span"])
    for field_name in (
        "subject_is_uri",
        "predicate_is_uri",
        "object_is_uri",
        "object_is_literal",
        "datatype",
        "language",
    ):
        if value[field_name] is None:
            del value[field_name]
    return value
