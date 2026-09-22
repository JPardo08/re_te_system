"""P0 structural validation; no ontology or Hohfeld semantics."""

from __future__ import annotations

from re_te_system.contracts import ParseIssue, ParsedTriple, ValidatedResult, Violation


def validate_structural(
    triples: tuple[ParsedTriple, ...],
    parse_issues: tuple[ParseIssue, ...] = (),
) -> ValidatedResult:
    violations: list[Violation] = [
        Violation(
            code=f"PARSE_{issue.code}",
            severity=issue.severity,
            message=issue.message,
            triple_index=issue.chunk_index,
        )
        for issue in parse_issues
    ]
    seen: dict[tuple[str, str, str], int] = {}
    for index, triple in enumerate(triples):
        required_are_strings = True
        for field_name, value in (
            ("SUBJECT", triple.subject),
            ("RELATION", triple.relation),
            ("OBJECT", triple.object),
        ):
            if not isinstance(value, str):
                required_are_strings = False
                violations.append(
                    Violation(
                        f"INVALID_{field_name}_TYPE",
                        "hard",
                        f"{field_name.lower()} must be a string",
                        index,
                    )
                )
            elif not value:
                violations.append(
                    Violation(
                        f"EMPTY_{field_name}",
                        "hard",
                        f"{field_name.lower()} is empty",
                        index,
                    )
                )
        for field_name, value in (
            ("SUBJECT_TYPE", triple.subject_type),
            ("OBJECT_TYPE", triple.object_type),
            ("SPAN_SOURCE", triple.span_source),
            ("SEGMENT_ID", triple.segment_id),
            ("DATATYPE", triple.datatype),
            ("LANGUAGE", triple.language),
        ):
            if value is not None and not isinstance(value, str):
                violations.append(
                    Violation(
                        f"INVALID_{field_name}_TYPE",
                        "hard",
                        f"{field_name.lower()} must be a string or null",
                        index,
                    )
                )
        for field_name, value in (
            ("SUBJECT_IS_URI", triple.subject_is_uri),
            ("PREDICATE_IS_URI", triple.predicate_is_uri),
            ("OBJECT_IS_URI", triple.object_is_uri),
            ("OBJECT_IS_LITERAL", triple.object_is_literal),
        ):
            if value is not None and not isinstance(value, bool):
                violations.append(
                    Violation(
                        f"INVALID_{field_name}_TYPE",
                        "hard",
                        f"{field_name.lower()} must be a boolean or null",
                        index,
                    )
                )
        for field_name, value in (
            ("SUBJECT_SPAN", triple.subject_span),
            ("OBJECT_SPAN", triple.object_span),
        ):
            if value is not None and (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(not isinstance(point, int) for point in value)
            ):
                violations.append(
                    Violation(
                        f"INVALID_{field_name}_TYPE",
                        "hard",
                        f"{field_name.lower()} must be a two-integer tuple or null",
                        index,
                    )
                )
        if required_are_strings:
            key = (triple.subject, triple.relation, triple.object)
            if key in seen:
                violations.append(
                    Violation(
                        "DUPLICATE_TRIPLE",
                        "soft",
                        f"Duplicates triple at index {seen[key]}",
                        index,
                    )
                )
            else:
                seen[key] = index
    if any(item.severity == "hard" for item in violations):
        status = "hard_fail"
    elif violations:
        status = "soft_fail"
    else:
        status = "ok"
    return ValidatedResult(triples, tuple(violations), status)
