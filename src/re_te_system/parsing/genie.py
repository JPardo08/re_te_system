"""Ordered conservative parser for GenIE control-token linearizations."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, Mapping

from re_te_system.contracts import (
    ParseIssue,
    ParseResult,
    ParsedGenieOccurrence,
    ParsedTriple,
)


TAG_SUB = "<sub>"
TAG_REL = "<rel>"
TAG_OBJ = "<obj>"
TAG_ET = "<et>"
CONTROL_TAGS = (TAG_SUB, TAG_REL, TAG_OBJ, TAG_ET)
TAG_PATTERN = re.compile(r"<sub>|<rel>|<obj>|<et>")
EXPECTED_CYCLE = (TAG_SUB, TAG_REL, TAG_OBJ, TAG_ET)


def _inventory_status(value: str, inventory: set[str] | None) -> str | None:
    if inventory is None:
        return None
    return "in_inventory" if value in inventory else "out_of_inventory"


def _id_status(candidates: tuple[str, ...]) -> tuple[str | None, str]:
    if len(candidates) == 1:
        return candidates[0], "resolved"
    if len(candidates) > 1:
        return None, "ambiguous"
    return None, "unresolved"


def parse_genie(
    text: str,
    segment_id: str | None = None,
    *,
    entity_inventory: Iterable[str] | None = None,
    relation_inventory: Iterable[str] | None = None,
) -> ParseResult:
    """Parse RAW GenIE text into ordered occurrences. Never uses a set."""
    issues: list[ParseIssue] = []
    occurrences: list[ParsedGenieOccurrence] = []
    seen: dict[tuple[str, str, str], int] = {}
    entities = set(entity_inventory) if entity_inventory is not None else None
    relations = set(relation_inventory) if relation_inventory is not None else None

    if not text.strip():
        issues.append(
            ParseIssue("EMPTY_STRUCTURE", "GenIE output is empty", "hard")
        )
        return ParseResult((), tuple(issues), genie_occurrences=())

    matches = list(TAG_PATTERN.finditer(text))
    if not matches:
        issues.append(
            ParseIssue(
                "MALFORMED_GENIE",
                "GenIE output has no <sub>/<rel>/<obj>/<et> structure",
                "hard",
            )
        )
        return ParseResult((), tuple(issues), genie_occurrences=())

    prefix = text[: matches[0].start()]
    if prefix.strip():
        issues.append(
            ParseIssue(
                "MALFORMED_GENIE",
                "Non-tag text precedes the first GenIE control token",
                "hard",
            )
        )

    expected_index = 0
    fields: list[str] = []

    def flush_incomplete(message: str) -> None:
        nonlocal fields, expected_index
        if not fields and expected_index == 0:
            return
        subject = fields[0] if len(fields) > 0 else ""
        relation = fields[1] if len(fields) > 1 else ""
        obj = fields[2] if len(fields) > 2 else ""
        occurrences.append(
            ParsedGenieOccurrence(
                subject_text=subject,
                relation_text=relation,
                object_text=obj,
                occurrence_index=len(occurrences),
                complete=False,
                subject_inventory_status=_inventory_status(subject, entities),
                relation_inventory_status=_inventory_status(relation, relations),
                object_inventory_status=_inventory_status(obj, entities),
                segment_id=segment_id,
            )
        )
        issues.append(
            ParseIssue(
                "INCOMPLETE_TRIPLET",
                message,
                "hard",
                chunk_index=occurrences[-1].occurrence_index,
            )
        )
        fields = []
        expected_index = 0

    for index, match in enumerate(matches):
        tag = match.group()
        following_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        payload = text[match.end() : following_end].strip()
        expected = EXPECTED_CYCLE[expected_index]
        if tag != expected:
            issues.append(
                ParseIssue(
                    "UNEXPECTED_CONTROL_TOKEN",
                    f"Expected {expected} but found {tag}",
                    "hard",
                    chunk_index=len(occurrences),
                )
            )
            if tag == TAG_SUB:
                flush_incomplete("Incomplete triplet discarded before a new <sub>")
                expected_index = 1
                fields = [payload]
                continue
            flush_incomplete(f"Incomplete triplet interrupted by unexpected {tag}")
            if tag == TAG_ET:
                expected_index = 0
            continue
        if tag in {TAG_SUB, TAG_REL, TAG_OBJ}:
            fields.append(payload)
            expected_index += 1
            continue
        if len(fields) != 3:
            flush_incomplete("End-triplet reached without subject/relation/object")
            continue
        subject, relation, obj = fields
        key = (subject, relation, obj)
        if key in seen:
            issues.append(
                ParseIssue(
                    "DUPLICATE_OCCURRENCE",
                    f"Duplicate GenIE occurrence of index {seen[key]}",
                    "soft",
                    chunk_index=len(occurrences),
                )
            )
        else:
            seen[key] = len(occurrences)
        occurrences.append(
            ParsedGenieOccurrence(
                subject_text=subject,
                relation_text=relation,
                object_text=obj,
                occurrence_index=len(occurrences),
                complete=True,
                subject_inventory_status=_inventory_status(subject, entities),
                relation_inventory_status=_inventory_status(relation, relations),
                object_inventory_status=_inventory_status(obj, entities),
                segment_id=segment_id,
            )
        )
        fields = []
        expected_index = 0

    if fields or expected_index != 0:
        flush_incomplete("Trailing incomplete GenIE occurrence")

    triples = tuple(
        ParsedTriple(
            occurrence.subject_text,
            occurrence.relation_text,
            occurrence.object_text,
            segment_id=segment_id,
        )
        for occurrence in occurrences
        if occurrence.complete
    )
    return ParseResult(triples, tuple(issues), genie_occurrences=tuple(occurrences))


def project_genie_occurrences(
    occurrences: tuple[ParsedGenieOccurrence, ...],
) -> tuple[ParsedTriple, ...]:
    return tuple(
        ParsedTriple(
            occurrence.subject_text,
            occurrence.relation_text,
            occurrence.object_text,
            segment_id=occurrence.segment_id,
        )
        for occurrence in occurrences
        if occurrence.complete
    )


def map_genie_ids(
    occurrences: tuple[ParsedGenieOccurrence, ...],
    entity_name_to_ids: Mapping[str, tuple[str, ...]],
    relation_name_to_ids: Mapping[str, tuple[str, ...]],
) -> tuple[ParsedGenieOccurrence, ...]:
    """Optional post-parser ID mapping. Never last-write-wins; never mutates RAW."""
    mapped: list[ParsedGenieOccurrence] = []
    for occurrence in occurrences:
        subject_id, subject_status = _id_status(
            tuple(entity_name_to_ids.get(occurrence.subject_text, ()))
        )
        relation_id, relation_status = _id_status(
            tuple(relation_name_to_ids.get(occurrence.relation_text, ()))
        )
        object_id, object_status = _id_status(
            tuple(entity_name_to_ids.get(occurrence.object_text, ()))
        )
        mapped.append(
            replace(
                occurrence,
                subject_id=subject_id,
                relation_id=relation_id,
                object_id=object_id,
                subject_id_status=subject_status,
                relation_id_status=relation_status,
                object_id_status=object_status,
            )
        )
    return tuple(mapped)
