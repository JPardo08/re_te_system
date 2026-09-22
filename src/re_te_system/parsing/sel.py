"""Conservative parser for UIE Spot-Association SEL output."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable

from re_te_system.conditioning.uie_schema import (
    NULL_SPAN,
    SPAN_START,
    TYPE_END,
    TYPE_START,
)
from re_te_system.contracts import (
    ParseIssue,
    ParseResult,
    ParsedAssociation,
    ParsedSpot,
    ParsedTriple,
)


_SPECIAL_TOKEN = re.compile(r"(<extra_id_\d+>|<pad>|</s>|<unk>)")


class _SELParseError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _tokens(text: str) -> list[str]:
    result: list[str] = []
    for piece in _SPECIAL_TOKEN.split(text):
        if not piece:
            continue
        if _SPECIAL_TOKEN.fullmatch(piece):
            result.append(piece)
        else:
            result.extend(piece.split())
    return result


def parse_sel(
    text: str,
    segment_id: str | None = None,
    *,
    spot_labels: Iterable[str] = (),
    association_labels: Iterable[str] = (),
) -> ParseResult:
    """Parse one complete SEL tree without repair, truncation, or deduplication."""
    tokens = _tokens(text)
    spot_set = set(spot_labels)
    association_set = set(association_labels)
    issues: list[ParseIssue] = []
    structures: list[ParsedSpot] = []
    cursor = 0

    while cursor < len(tokens) and tokens[cursor] == "<pad>":
        cursor += 1

    if cursor >= len(tokens):
        return ParseResult(
            (),
            (ParseIssue("EMPTY_STRUCTURE", "SEL output has no structural tree"),),
            (),
        )
    if tokens[cursor] != TYPE_START:
        return ParseResult(
            (),
            (
                ParseIssue(
                    "UNEXPECTED_TOKEN",
                    f"SEL must start with {TYPE_START}; found {tokens[cursor]!r}",
                    "hard",
                ),
            ),
            (),
        )
    cursor += 1

    def collect_until(stops: set[str], context: str) -> tuple[str, str]:
        nonlocal cursor
        values: list[str] = []
        while cursor < len(tokens) and tokens[cursor] not in stops:
            token = tokens[cursor]
            if token == NULL_SPAN:
                values.append(token)
                cursor += 1
                continue
            if token.startswith("<extra_id_") or token in {"<pad>", "</s>"}:
                raise _SELParseError(
                    "UNEXPECTED_TOKEN",
                    f"Unexpected token {token!r} while reading {context}",
                )
            values.append(token)
            cursor += 1
        if cursor >= len(tokens):
            raise _SELParseError(
                "TRUNCATED_SEL",
                f"SEL ended while reading {context}",
            )
        return " ".join(values).strip(), tokens[cursor]

    def parse_node(kind: str) -> tuple[str, str, list[ParsedAssociation]]:
        nonlocal cursor
        if cursor >= len(tokens) or tokens[cursor] != TYPE_START:
            found = tokens[cursor] if cursor < len(tokens) else "<eof>"
            raise _SELParseError(
                "MALFORMED_SEL",
                f"Expected {TYPE_START} for {kind}; found {found!r}",
            )
        cursor += 1
        label, delimiter = collect_until({SPAN_START, TYPE_END, TYPE_START}, f"{kind} label")
        if delimiter != SPAN_START:
            raise _SELParseError(
                "MALFORMED_SEL",
                f"{kind} label must be followed by {SPAN_START}",
            )
        cursor += 1
        span, delimiter = collect_until({TYPE_START, TYPE_END}, f"{kind} span")
        if not label or not span:
            raise _SELParseError(
                "MALFORMED_SEL",
                f"{kind} requires a non-empty label and span",
            )
        associations: list[ParsedAssociation] = []
        if kind == "association" and delimiter == TYPE_START:
            raise _SELParseError(
                "MALFORMED_SEL",
                "Association nodes cannot contain nested nodes",
            )
        while kind == "spot" and delimiter == TYPE_START:
            association_label, association_span, nested = parse_node("association")
            if nested:
                raise _SELParseError(
                    "MALFORMED_SEL",
                    "Association contains unexpected nested structures",
                )
            associations.append(
                ParsedAssociation(
                    label=association_label,
                    span=association_span,
                )
            )
            if cursor >= len(tokens):
                raise _SELParseError(
                    "TRUNCATED_SEL",
                    "SEL ended before closing spot",
                )
            delimiter = tokens[cursor]
        if delimiter != TYPE_END:
            raise _SELParseError(
                "MALFORMED_SEL",
                f"Expected {TYPE_END} after {kind}; found {delimiter!r}",
            )
        cursor += 1
        return label, span, associations

    try:
        while cursor < len(tokens) and tokens[cursor] == TYPE_START:
            label, span, associations = parse_node("spot")
            if label not in spot_set:
                issues.append(
                    ParseIssue(
                        "UNKNOWN_LABEL",
                        f"Unknown spot label: {label}",
                        "soft",
                        len(structures),
                    )
                )
            if "<unk>" in span:
                issues.append(
                    ParseIssue(
                        "UNKNOWN_SPAN",
                        f"Spot span contains unresolved <unk>: {span}",
                        "soft",
                        len(structures),
                    )
                )
            for association in associations:
                if association.label not in association_set:
                    issues.append(
                        ParseIssue(
                            "UNKNOWN_LABEL",
                            f"Unknown association label: {association.label}",
                            "soft",
                            len(structures),
                        )
                    )
                if "<unk>" in association.span:
                    issues.append(
                        ParseIssue(
                            "UNKNOWN_SPAN",
                            f"Association span contains unresolved <unk>: {association.span}",
                            "soft",
                            len(structures),
                        )
                    )
            structures.append(
                ParsedSpot(
                    label=label,
                    span=span,
                    associations=tuple(associations),
                    segment_id=segment_id,
                )
            )
        if cursor >= len(tokens):
            raise _SELParseError("TRUNCATED_SEL", "SEL outer structure is not closed")
        if tokens[cursor] != TYPE_END:
            raise _SELParseError(
                "UNEXPECTED_TOKEN",
                f"Expected outer {TYPE_END}; found {tokens[cursor]!r}",
            )
        cursor += 1
    except _SELParseError as exc:
        issues.append(ParseIssue(exc.code, str(exc), "hard"))
        return ParseResult((), tuple(issues), tuple(structures))

    while cursor < len(tokens) and tokens[cursor] in {"</s>", "<pad>"}:
        cursor += 1
    if cursor < len(tokens):
        issues.append(
            ParseIssue(
                "UNEXPECTED_TOKEN",
                f"Unexpected content after complete SEL tree: {' '.join(tokens[cursor:])}",
                "hard",
            )
        )
    if not structures:
        issues.append(ParseIssue("EMPTY_STRUCTURE", "SEL tree contains no spots"))
    return ParseResult((), tuple(issues), tuple(structures))


def align_spot_structures(
    structures: tuple[ParsedSpot, ...],
    source_text: str,
    segment_start: int = 0,
) -> tuple[tuple[ParsedSpot, ...], tuple[ParseIssue, ...]]:
    """Apply exact, non-fuzzy alignment without deleting ambiguous predictions."""

    issues: list[ParseIssue] = []

    def align_span(span: str, context: str, index: int) -> tuple[int | None, int | None, str]:
        if span == NULL_SPAN:
            return None, None, "null"
        if "<unk>" in span:
            return None, None, "not_found"
        starts = [match.start() for match in re.finditer(re.escape(span), source_text)]
        if len(starts) == 1:
            start = segment_start + starts[0]
            return start, start + len(span), "exact"
        status = "ambiguous" if starts else "not_found"
        issues.append(
            ParseIssue(
                "UNKNOWN_SPAN",
                f"{context} span is {status}: {span}",
                "soft",
                index,
            )
        )
        return None, None, status

    aligned: list[ParsedSpot] = []
    for index, spot in enumerate(structures):
        start, end, status = align_span(spot.span, "Spot", index)
        associations: list[ParsedAssociation] = []
        for association in spot.associations:
            asoc_start, asoc_end, asoc_status = align_span(
                association.span,
                "Association",
                index,
            )
            associations.append(
                replace(
                    association,
                    span_start=asoc_start,
                    span_end=asoc_end,
                    alignment_status=asoc_status,
                )
            )
        aligned.append(
            replace(
                spot,
                associations=tuple(associations),
                span_start=start,
                span_end=end,
                alignment_status=status,
            )
        )
    return tuple(aligned), tuple(issues)


def project_binary_relations(
    structures: tuple[ParsedSpot, ...],
) -> tuple[ParsedTriple, ...]:
    """Project only explicit Spot-Association pairs into canonical triples."""
    triples: list[ParsedTriple] = []
    for spot in structures:
        if spot.span == NULL_SPAN:
            continue
        for association in spot.associations:
            if association.span == NULL_SPAN:
                continue
            triples.append(
                ParsedTriple(
                    subject=spot.span,
                    relation=association.label,
                    object=association.span,
                    subject_type=spot.label,
                    subject_span=(
                        (spot.span_start, spot.span_end)
                        if spot.span_start is not None and spot.span_end is not None
                        else None
                    ),
                    object_span=(
                        (association.span_start, association.span_end)
                        if association.span_start is not None
                        and association.span_end is not None
                        else None
                    ),
                    span_source=(
                        "uie_exact_string_alignment"
                        if spot.alignment_status == "exact"
                        and association.alignment_status == "exact"
                        else (
                            "uie_structure_alignment:"
                            f"{spot.alignment_status}/{association.alignment_status}"
                        )
                    ),
                    segment_id=spot.segment_id,
                )
            )
    return tuple(triples)
