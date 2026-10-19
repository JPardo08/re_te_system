"""Condition-blind parser for tab-separated extractor output.

A valid line has exactly two tab separators and three fields that each contain
at least one non-whitespace character, in the order subject, relation, object.
Accepted fields are copied unchanged. A field that is empty after ``strip`` is
rejected and does not become a triple. This module does not case-fold,
deduplicate, sort, repair separators, or consult a relation inventory.
Generation failures stay outside the parser: truncation is reported only when
caller-supplied generation metadata says so.
"""

from __future__ import annotations

from typing import Mapping

from re_te_system.contracts import ParseIssue, ParseResult, ParsedTriple


def _logical_lines(text: str) -> list[str]:
    """Split on LF and CRLF. A final newline is a terminator, not a new line."""
    if text == "":
        return []
    normalized = text.replace("\r\n", "\n")
    parts = normalized.split("\n")
    if normalized.endswith("\n"):
        parts = parts[:-1]
    return parts


def _line_issue(index: int, message: str) -> ParseIssue:
    return ParseIssue("UNPARSEABLE_CHUNK", message, "soft", index)


def parse_tsv(text: str, segment_id: str | None = None) -> ParseResult:
    """Parse one TSV response into ordered triples and parse issues.

    ``segment_id`` is copied onto every accepted triple. No other optional
    ``ParsedTriple`` field is populated.
    """
    if not isinstance(text, str):
        raise TypeError("model output must be a string")
    lines = _logical_lines(text)
    if not lines or all(line == "" for line in lines):
        return ParseResult((), (ParseIssue("EMPTY_MODEL_OUTPUT", "Model output is empty"),))

    triples: list[ParsedTriple] = []
    issues: list[ParseIssue] = []
    for index, line in enumerate(lines):
        tabs = line.count("\t")
        if tabs == 0:
            issues.append(_line_issue(index, "Line has no tab separator"))
            continue
        if tabs != 2:
            issues.append(
                _line_issue(index, f"Line has {tabs} tab separators; expected 2")
            )
            continue
        subject, relation, object_ = line.split("\t")
        if subject.strip() == "" or relation.strip() == "" or object_.strip() == "":
            issues.append(_line_issue(index, "Line has an empty required field"))
            continue
        triples.append(
            ParsedTriple(
                subject=subject,
                relation=relation,
                object=object_,
                segment_id=segment_id,
            )
        )
    return ParseResult(tuple(triples), tuple(issues))


def generation_issue(metadata: Mapping[str, object] | None) -> ParseIssue | None:
    """Report truncation only from generation metadata, never from line shape."""
    if metadata is None or metadata.get("output_reached_limit") is not True:
        return None
    return ParseIssue(
        "TRUNCATED_GENERATION",
        "Generation reached the configured output token limit",
        "soft",
    )
