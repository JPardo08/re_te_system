"""Synthetic unit tests for the condition-blind TSV adapter.

Every string in this module is invented. The tests do not read benchmark Gold
and do not treat any relation inventory as a parse rule.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from re_te_system.contracts import ParseIssue, ParsedTriple
from re_te_system.normalization.basic import normalize_triples
from re_te_system.parsing import tsv as tsv_module
from re_te_system.parsing.tsv import generation_issue, parse_tsv
from re_te_system.validation.structural import validate_structural


def _fields(result) -> list[tuple[str, str, str]]:
    return [(item.subject, item.relation, item.object) for item in result.triples]


def _optional_fields_are_unset(triple: ParsedTriple) -> None:
    assert triple.subject_type is None
    assert triple.object_type is None
    assert triple.subject_span is None
    assert triple.object_span is None
    assert triple.span_source is None
    assert triple.subject_is_uri is None
    assert triple.predicate_is_uri is None
    assert triple.object_is_uri is None
    assert triple.object_is_literal is None
    assert triple.datatype is None
    assert triple.language is None


def test_one_valid_triple() -> None:
    result = parse_tsv("Acme\tsupplies\tWidget", "seg-1")
    assert _fields(result) == [("Acme", "supplies", "Widget")]
    assert result.issues == ()
    assert result.triples[0].segment_id == "seg-1"
    _optional_fields_are_unset(result.triples[0])


def test_several_valid_triples_keep_order() -> None:
    raw = "Acme\tsupplies\tWidget\nAcme\tships\tCrate\nNorth\towns\tDock"
    result = parse_tsv(raw)
    assert _fields(result) == [
        ("Acme", "supplies", "Widget"),
        ("Acme", "ships", "Crate"),
        ("North", "owns", "Dock"),
    ]
    assert result.issues == ()


def test_empty_string() -> None:
    result = parse_tsv("")
    assert result.triples == ()
    assert result.issues == (ParseIssue("EMPTY_MODEL_OUTPUT", "Model output is empty"),)


def test_only_blank_lines() -> None:
    for raw in ("\n", "\n\n", "\r\n", "\r\n\r\n"):
        result = parse_tsv(raw)
        assert result.triples == ()
        assert result.issues == (
            ParseIssue("EMPTY_MODEL_OUTPUT", "Model output is empty"),
        )


def test_empty_field_is_not_a_triple() -> None:
    for raw in ("Acme\t\tWidget", "\tsupplies\tWidget", "Acme\tsupplies\t"):
        result = parse_tsv(raw)
        assert result.triples == ()
        assert result.issues == (
            ParseIssue(
                "UNPARSEABLE_CHUNK",
                "Line has an empty required field",
                "soft",
                0,
            ),
        )


def test_one_tab_is_rejected() -> None:
    result = parse_tsv("Acme\tWidget")
    assert result.triples == ()
    assert result.issues[0].code == "UNPARSEABLE_CHUNK"
    assert result.issues[0].message == "Line has 1 tab separators; expected 2"


def test_three_tabs_are_not_joined() -> None:
    result = parse_tsv("Acme\tsupplies\tWidget\textra")
    assert result.triples == ()
    assert result.issues[0].message == "Line has 3 tab separators; expected 2"


def test_spaces_are_not_separators() -> None:
    result = parse_tsv("Acme supplies Widget")
    assert result.triples == ()
    assert result.issues[0].message == "Line has no tab separator"


def test_identical_duplicates_are_preserved() -> None:
    raw = "Acme\tsupplies\tWidget\nAcme\tsupplies\tWidget"
    result = parse_tsv(raw)
    assert _fields(result) == [
        ("Acme", "supplies", "Widget"),
        ("Acme", "supplies", "Widget"),
    ]
    assert result.issues == ()
    validated = validate_structural(result.triples)
    assert len(validated.triples) == 2
    assert [item.code for item in validated.violations] == ["DUPLICATE_TRIPLE"]
    assert validated.status == "soft_fail"


def test_same_subject_and_relation_with_different_objects() -> None:
    raw = "Acme\tsupplies\tWidget\nAcme\tsupplies\tCrate"
    result = parse_tsv(raw)
    assert _fields(result) == [
        ("Acme", "supplies", "Widget"),
        ("Acme", "supplies", "Crate"),
    ]
    validated = validate_structural(result.triples)
    assert validated.violations == ()
    assert validated.status == "ok"


def test_open_label_is_preserved() -> None:
    result = parse_tsv("Acme\topen-link\tWidget")
    assert result.triples[0].relation == "open-link"
    assert result.issues == ()


def test_label_outside_a_local_inventory_is_preserved() -> None:
    inventory = ("open-link", "owns-item")
    result = parse_tsv("Acme\toutside-inventory\tWidget")
    relation = result.triples[0].relation
    assert relation == "outside-inventory"
    assert relation not in inventory
    assert result.issues == ()


def test_case_differences_stay_distinct() -> None:
    lower = parse_tsv("Acme\topen-link\tWidget")
    upper = parse_tsv("Acme\tOpen-Link\tWidget")
    assert lower.triples[0].relation == "open-link"
    assert upper.triples[0].relation == "Open-Link"
    assert lower.triples[0].relation != upper.triples[0].relation


def test_internal_and_outer_spaces_are_not_trimmed_by_the_parser() -> None:
    raw = "  Acme   North  \t  ships   goods  \t  Widget  "
    result = parse_tsv(raw)
    assert _fields(result) == [("  Acme   North  ", "  ships   goods  ", "  Widget  ")]
    assert result.issues == ()


def test_whitespace_only_fields_are_rejected_without_creating_triples() -> None:
    cases = {
        "subject": "   \tsupplies\tWidget",
        "relation": "Acme\t   \tWidget",
        "object": "Acme\tsupplies\t   ",
        "subject_and_relation": " \t \tWidget",
        "relation_and_object": "Acme\t \t  ",
        "nbsp_relation": "Acme\t\u00a0\tWidget",
    }
    for raw in cases.values():
        result = parse_tsv(raw)
        assert result.triples == ()
        assert result.issues == (
            ParseIssue(
                "UNPARSEABLE_CHUNK",
                "Line has an empty required field",
                "soft",
                0,
            ),
        )
    crlf = parse_tsv("Acme\t   \tWidget\r\n")
    assert crlf.triples == ()
    assert crlf.issues[0].message == "Line has an empty required field"
    surrounded = parse_tsv(
        "Acme\tsupplies\tWidget\r\nAcme\t \tWidget\r\nNorth\towns\tDock\r\n"
    )
    assert _fields(surrounded) == [
        ("Acme", "supplies", "Widget"),
        ("North", "owns", "Dock"),
    ]
    assert surrounded.issues == (
        ParseIssue(
            "UNPARSEABLE_CHUNK",
            "Line has an empty required field",
            "soft",
            1,
        ),
    )
    raw = "Acme\topen-link\tWidget\nAcme\topen-link\tWidget\nAcme\t   \tWidget\n"
    duplicated = parse_tsv(raw, "seg")
    assert _fields(duplicated) == [
        ("Acme", "open-link", "Widget"),
        ("Acme", "open-link", "Widget"),
    ]
    assert duplicated.issues[0].chunk_index == 2
    assert "condition" not in inspect.signature(parse_tsv).parameters
    for _condition in ("C0", "C1", "C2", "C3", "C3-NL"):
        assert parse_tsv(raw, "seg") == duplicated


def test_crlf_does_not_keep_carriage_return() -> None:
    raw = "Acme\tsupplies\tWidget\r\nNorth\towns\tDock\r\n"
    result = parse_tsv(raw)
    assert _fields(result) == [
        ("Acme", "supplies", "Widget"),
        ("North", "owns", "Dock"),
    ]
    assert all("\r" not in field for triple in _fields(result) for field in triple)


def test_final_newline_is_not_an_extra_prediction() -> None:
    assert parse_tsv("Acme\tsupplies\tWidget\n") == parse_tsv("Acme\tsupplies\tWidget")
    assert len(parse_tsv("Acme\tsupplies\tWidget\n").triples) == 1


def test_partially_valid_response_keeps_valid_lines() -> None:
    raw = "Acme\tsupplies\tWidget\nnot a triple\nNorth\towns\tDock"
    result = parse_tsv(raw)
    assert _fields(result) == [
        ("Acme", "supplies", "Widget"),
        ("North", "owns", "Dock"),
    ]
    assert result.issues == (
        ParseIssue("UNPARSEABLE_CHUNK", "Line has no tab separator", "soft", 1),
    )
    mixed_blank = parse_tsv("Acme\tsupplies\tWidget\n\nNorth\towns\tDock")
    assert _fields(mixed_blank) == [
        ("Acme", "supplies", "Widget"),
        ("North", "owns", "Dock"),
    ]
    assert mixed_blank.issues == (
        ParseIssue("UNPARSEABLE_CHUNK", "Line has no tab separator", "soft", 1),
    )


def test_truncated_line_is_a_parse_issue_unless_metadata_says_so() -> None:
    raw = "Acme\tsupplies\tWidget\nAcme\tsupplies"
    parsed = parse_tsv(raw)
    assert _fields(parsed) == [("Acme", "supplies", "Widget")]
    assert [item.code for item in parsed.issues] == ["UNPARSEABLE_CHUNK"]
    assert generation_issue(None) is None
    assert generation_issue({"output_reached_limit": False}) is None
    assert generation_issue({}) is None
    truncated = generation_issue(
        {"output_reached_limit": True, "max_new_tokens": 8, "do_sample": False}
    )
    assert truncated == ParseIssue(
        "TRUNCATED_GENERATION",
        "Generation reached the configured output token limit",
        "soft",
    )
    complete = parse_tsv("Acme\tsupplies\tWidget\n")
    assert complete.issues == ()
    assert generation_issue({"output_reached_limit": True}) is not None


def test_raw_output_is_not_replaced_by_normalization() -> None:
    raw = "  Acme  \tsupplies\tWidget\nbroken"
    parsed = parse_tsv(raw, "seg-9")
    normalized = normalize_triples(parsed.triples)
    record = {
        "raw": {"model_output": raw, "generation_metadata": {"output_reached_limit": False}},
        "parsed": _fields(parsed),
        "normalized": [
            (item.subject, item.relation, item.object) for item in normalized
        ],
    }
    assert record["raw"]["model_output"] == raw
    assert record["parsed"] == [("  Acme  ", "supplies", "Widget")]
    assert record["normalized"] == [("Acme", "supplies", "Widget")]
    assert "broken" not in str(record["parsed"])
    assert parsed.issues[0].code == "UNPARSEABLE_CHUNK"
    assert parsed.issues[0].code != "MODEL_FAILURE"


def test_condition_does_not_change_the_parse() -> None:
    raw = "alpha\topen-link\tbeta\n"
    baseline = parse_tsv(raw, "seg")
    assert "condition" not in inspect.signature(parse_tsv).parameters
    for _condition in ("C0", "C1", "C2", "C3", "C3-NL"):
        assert parse_tsv(raw, "seg") == baseline
    source = Path(tsv_module.__file__).read_text(encoding="utf-8")
    for marker in ("Duty", "Privilege", "NoRight", "Hohfeld"):
        assert marker not in source
