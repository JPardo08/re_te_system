"""Parser for the standard monolingual REBEL control-token stream."""

from __future__ import annotations

import re

from re_te_system.contracts import ParseIssue, ParseResult, ParsedTriple


def _clean(text: str) -> str:
    for token in ("<s>", "</s>", "<pad>"):
        text = text.replace(token, " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_rebel(text: str, segment_id: str | None = None) -> ParseResult:
    """Parse ``<triplet> subject <subj> object <obj> relation`` occurrences."""
    cleaned = _clean(text)
    if not cleaned:
        return ParseResult((), (ParseIssue("EMPTY_MODEL_OUTPUT", "Model output is empty"),))

    issues: list[ParseIssue] = []
    triples: list[ParsedTriple] = []
    pieces = re.split(r"(?i)<triplet>", cleaned)
    preamble = pieces[0].strip()
    if preamble:
        issues.append(
            ParseIssue("UNEXPECTED_PREAMBLE", "Text before first <triplet> was ignored")
        )
    chunks = [piece.strip() for piece in pieces[1:]]
    if not chunks:
        return ParseResult(
            (),
            tuple(issues)
            + (ParseIssue("UNPARSEABLE_OUTPUT", "No <triplet> control token found"),),
        )

    for index, chunk in enumerate(chunks):
        subj_markers = list(re.finditer(r"(?i)<subj>", chunk))
        obj_markers = list(re.finditer(r"(?i)<obj>", chunk))
        if len(subj_markers) != 1 or len(obj_markers) != 1:
            issues.append(
                ParseIssue(
                    "TRUNCATED_TRIPLE",
                    "Triple requires exactly one <subj> and one <obj> marker",
                    "soft",
                    index,
                )
            )
            continue
        subj_marker, obj_marker = subj_markers[0], obj_markers[0]
        if subj_marker.start() > obj_marker.start():
            issues.append(
                ParseIssue(
                    "MALFORMED_CONTROL_ORDER",
                    "<subj> must precede <obj>",
                    "soft",
                    index,
                )
            )
            continue
        triple = ParsedTriple(
            subject=chunk[: subj_marker.start()].strip(),
            object=chunk[subj_marker.end() : obj_marker.start()].strip(),
            relation=chunk[obj_marker.end() :].strip(),
            segment_id=segment_id,
        )
        triples.append(triple)
        if not (triple.subject and triple.object and triple.relation):
            issues.append(
                ParseIssue(
                    "INCOMPLETE_TRIPLE",
                    "Parsed triple contains an empty required field",
                    "soft",
                    index,
                )
            )
    return ParseResult(tuple(triples), tuple(issues))
