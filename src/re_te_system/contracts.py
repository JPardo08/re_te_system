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
class ParsedAssociation:
    label: str
    span: str
    span_start: int | None = None
    span_end: int | None = None
    alignment_status: str = "not_attempted"


@dataclass(frozen=True)
class ParsedSpot:
    label: str
    span: str
    associations: tuple[ParsedAssociation, ...] = ()
    segment_id: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    alignment_status: str = "not_attempted"


@dataclass(frozen=True)
class ParseIssue:
    code: str
    message: str
    severity: str = "soft"
    chunk_index: int | None = None


@dataclass(frozen=True)
class ParsedGoLLIEArgument:
    name: str | None
    value: Any
    alignment_status: str = "not_attempted"
    span_start: int | None = None
    span_end: int | None = None


@dataclass(frozen=True)
class ParsedGoLLIERecord:
    kind: str
    class_name: str
    arguments: tuple[ParsedGoLLIEArgument, ...]
    schema_known: bool
    segment_id: str | None = None
    alignment_status: str = "not_attempted"


@dataclass(frozen=True)
class ParseResult:
    triples: tuple[ParsedTriple, ...]
    issues: tuple[ParseIssue, ...] = ()
    structures: tuple[ParsedSpot, ...] = ()
    gollie_records: tuple[ParsedGoLLIERecord, ...] = ()


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


def spot_to_dict(spot: ParsedSpot) -> dict[str, Any]:
    return {
        "alignment_status": spot.alignment_status,
        "associations": [
            {
                "alignment_status": association.alignment_status,
                "label": association.label,
                "span": association.span,
                "span_end": association.span_end,
                "span_start": association.span_start,
            }
            for association in spot.associations
        ],
        "label": spot.label,
        "segment_id": spot.segment_id,
        "span": spot.span,
        "span_end": spot.span_end,
        "span_start": spot.span_start,
    }


def gollie_record_to_dict(record: ParsedGoLLIERecord) -> dict[str, Any]:
    return {
        "alignment_status": record.alignment_status,
        "arguments": [
            {
                "alignment_status": argument.alignment_status,
                "name": argument.name,
                "span_end": argument.span_end,
                "span_start": argument.span_start,
                "value": argument.value,
            }
            for argument in record.arguments
        ],
        "class_name": record.class_name,
        "kind": record.kind,
        "schema_known": record.schema_known,
        "segment_id": record.segment_id,
    }
