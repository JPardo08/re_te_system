"""Deterministic parser for common mREBEL linearizations."""

from __future__ import annotations

import re

from re_te_system.contracts import ParseIssue, ParseResult, ParsedTriple


_SERIALIZATION_TOKENS = ("<s>", "</s>", "<pad>", "tp_XX", "<tp_XX>", "__es__", "__en__")
_TAG = re.compile(r"<([^<>]+)>")
_TRIPLET = re.compile(r"(?i)<(?:triplet|relation)>")


def clean_decoded(text: str) -> str:
    result = text
    for token in _SERIALIZATION_TOKENS:
        result = result.replace(token, " ")
    return re.sub(r"\s+", " ", result).strip()


def _typed_chunk(chunk: str, segment_id: str | None) -> ParsedTriple | None:
    tags = list(_TAG.finditer(chunk))
    if len(tags) < 2:
        return None
    first, second = tags[0], tags[1]
    subject = chunk[: first.start()].strip()
    object_ = chunk[first.end() : second.start()].strip()
    relation = _TAG.sub(" ", chunk[second.end() :])
    relation = re.sub(r"\s+", " ", relation).strip()
    if not (subject or object_ or relation):
        return None
    return ParsedTriple(
        subject=subject,
        subject_type=first.group(1).strip() or None,
        relation=relation,
        object=object_,
        object_type=second.group(1).strip() or None,
        segment_id=segment_id,
    )


def _subj_obj_chunk(chunk: str, segment_id: str | None) -> ParsedTriple | None:
    match = re.match(
        r"(?is)^\s*<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*(?:<rel>|<relation>)\s*(.*?)\s*$",
        chunk,
    )
    if not match:
        return None
    return ParsedTriple(
        subject=match.group(1).strip(),
        object=match.group(2).strip(),
        relation=match.group(3).strip(),
        segment_id=segment_id,
    )


def _pipe_chunk(chunk: str, segment_id: str | None) -> ParsedTriple | None:
    parts = [part.strip() for part in chunk.split("|")]
    if len(parts) < 3:
        return None
    return ParsedTriple(
        subject=parts[0],
        relation=parts[1],
        object=" | ".join(parts[2:]),
        segment_id=segment_id,
    )


def parse_mrebel(text: str, segment_id: str | None = None) -> ParseResult:
    """Parse output without deduplication or semantic relation rewriting."""
    cleaned = clean_decoded(text)
    issues: list[ParseIssue] = []
    triples: list[ParsedTriple] = []
    if not cleaned:
        return ParseResult((), (ParseIssue("EMPTY_MODEL_OUTPUT", "Model output is empty"),))

    chunks = [chunk.strip() for chunk in _TRIPLET.split(cleaned) if chunk.strip()]
    if not chunks:
        chunks = [cleaned]
    for index, chunk in enumerate(chunks):
        triple = (
            _subj_obj_chunk(chunk, segment_id)
            or _typed_chunk(chunk, segment_id)
            or _pipe_chunk(chunk, segment_id)
        )
        if triple is None:
            issues.append(
                ParseIssue(
                    "UNPARSEABLE_CHUNK",
                    "Chunk does not match a supported mREBEL linearization",
                    "soft",
                    index,
                )
            )
            continue
        triples.append(triple)
        if not (triple.subject and triple.relation and triple.object):
            issues.append(
                ParseIssue(
                    "INCOMPLETE_TRIPLE",
                    "Parsed chunk contains an empty required field",
                    "soft",
                    index,
                )
            )
    return ParseResult(tuple(triples), tuple(issues))
