"""Representational normalization only."""

from __future__ import annotations

import re
import unicodedata

from re_te_system.contracts import ParsedTriple


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def normalize_triples(triples: tuple[ParsedTriple, ...]) -> tuple[ParsedTriple, ...]:
    """Normalize serialization representation; retain order and duplicates."""
    return tuple(
        ParsedTriple(
            subject=normalize_text(triple.subject),
            relation=normalize_text(triple.relation),
            object=normalize_text(triple.object),
            subject_type=normalize_text(triple.subject_type) if triple.subject_type else None,
            object_type=normalize_text(triple.object_type) if triple.object_type else None,
            subject_span=triple.subject_span,
            object_span=triple.object_span,
            span_source=triple.span_source,
            segment_id=triple.segment_id,
        )
        for triple in triples
    )


def align_triples(
    triples: tuple[ParsedTriple, ...],
    segment_text: str,
    segment_start: int = 0,
) -> tuple[ParsedTriple, ...]:
    """Add post-hoc string offsets without changing or dropping predictions."""

    def locate(query: str) -> tuple[int, int] | None:
        if not query:
            return None
        pattern = re.escape(query)
        pattern = re.sub(r"(?:\\\s)+", r"\\s+", pattern)
        match = re.search(pattern, segment_text, flags=re.IGNORECASE)
        if not match:
            return None
        return segment_start + match.start(), segment_start + match.end()

    return tuple(
        ParsedTriple(
            subject=triple.subject,
            relation=triple.relation,
            object=triple.object,
            subject_type=triple.subject_type,
            object_type=triple.object_type,
            subject_span=locate(triple.subject),
            object_span=locate(triple.object),
            span_source="posthoc_string_alignment",
            segment_id=triple.segment_id,
        )
        for triple in triples
    )
