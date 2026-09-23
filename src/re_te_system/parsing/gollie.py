"""Safe, non-executing parser for GoLLIE Python-like constructor lists."""

from __future__ import annotations

import ast
from dataclasses import replace
import re
from typing import Any

from re_te_system.conditioning.gollie_schema import (
    KIND_RELATION,
    GoLLIEClass,
    GoLLIESchema,
)
from re_te_system.contracts import (
    ParseIssue,
    ParseResult,
    ParsedGoLLIEArgument,
    ParsedGoLLIERecord,
    ParsedTriple,
)


SAFE_NODE_TYPES = (
    ast.Constant,
    ast.Expression,
    ast.List,
    ast.Load,
    ast.Name,
    ast.Call,
    ast.keyword,
)
SCALAR_TYPES = (str, int, float, bool, type(None))


class _GoLLIEParseError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _list_payload(text: str) -> str:
    payload = text.strip()
    if "result =" in payload:
        payload = payload.rsplit("result =", 1)[-1].strip()
    return payload


def _literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, SCALAR_TYPES):
            raise _GoLLIEParseError(
                "UNSAFE_AST_NODE",
                f"Unsupported constant type: {type(node.value).__name__}",
            )
        return node.value
    if isinstance(node, ast.List):
        return [_literal_value(element) for element in node.elts]
    raise _GoLLIEParseError(
        "UNSAFE_AST_NODE",
        f"Unsupported value node: {type(node).__name__}",
    )


def _assert_safe_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, SAFE_NODE_TYPES):
            raise _GoLLIEParseError(
                "UNSAFE_AST_NODE",
                f"Rejected AST node: {type(node).__name__}",
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise _GoLLIEParseError(
                    "UNSAFE_AST_NODE",
                    "Constructor callee must be a bare class name",
                )
            if node.keywords and any(keyword.arg is None for keyword in node.keywords):
                raise _GoLLIEParseError(
                    "UNSAFE_AST_NODE",
                    "Starred keyword arguments are not allowed",
                )


def _expected_python_type(type_name: str) -> Any:
    mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }
    return mapping.get(type_name)


def _type_matches(value: Any, type_name: str) -> bool:
    stripped = type_name.replace(" ", "")
    if stripped.startswith("Optional[") and stripped.endswith("]"):
        inner = stripped[len("Optional[") : -1]
        return value is None or _type_matches(value, inner)
    if stripped.startswith("List[") and stripped.endswith("]"):
        inner = stripped[len("List[") : -1]
        return isinstance(value, list) and all(_type_matches(item, inner) for item in value)
    expected = _expected_python_type(stripped)
    if expected is None:
        return isinstance(value, str)
    return isinstance(value, expected)


def _bind_arguments(
    call: ast.Call,
    schema_class: GoLLIEClass | None,
) -> tuple[tuple[ParsedGoLLIEArgument, ...], list[ParseIssue]]:
    issues: list[ParseIssue] = []
    bound: list[ParsedGoLLIEArgument] = []
    known = {argument.name: argument for argument in schema_class.arguments} if schema_class else {}
    used: set[str] = set()

    if schema_class is not None:
        for index, value_node in enumerate(call.args):
            if index >= len(schema_class.arguments):
                issues.append(
                    ParseIssue(
                        "UNKNOWN_ARGUMENT",
                        f"Unexpected positional argument at index {index}",
                        "soft",
                    )
                )
                bound.append(ParsedGoLLIEArgument(name=None, value=_literal_value(value_node)))
                continue
            spec = schema_class.arguments[index]
            value = _literal_value(value_node)
            used.add(spec.name)
            if not _type_matches(value, spec.type_name):
                issues.append(
                    ParseIssue(
                        "ARGUMENT_TYPE_MISMATCH",
                        f"{spec.name} expected {spec.type_name}",
                        "soft",
                    )
                )
            bound.append(ParsedGoLLIEArgument(name=spec.name, value=value))
    else:
        for value_node in call.args:
            bound.append(ParsedGoLLIEArgument(name=None, value=_literal_value(value_node)))

    for keyword in call.keywords:
        name = keyword.arg
        value = _literal_value(keyword.value)
        if schema_class is not None and name not in known:
            issues.append(
                ParseIssue(
                    "UNKNOWN_ARGUMENT",
                    f"Unknown argument: {name}",
                    "soft",
                )
            )
        elif schema_class is not None and name in known:
            spec = known[name]
            used.add(name)
            if not _type_matches(value, spec.type_name):
                issues.append(
                    ParseIssue(
                        "ARGUMENT_TYPE_MISMATCH",
                        f"{name} expected {spec.type_name}",
                        "soft",
                    )
                )
        bound.append(ParsedGoLLIEArgument(name=name, value=value))

    if schema_class is not None:
        for spec in schema_class.arguments:
            if spec.required and spec.name not in used:
                issues.append(
                    ParseIssue(
                        "MISSING_REQUIRED_ARGUMENT",
                        f"Missing required argument: {spec.name}",
                        "soft",
                    )
                )
    return tuple(bound), issues


def parse_gollie(
    text: str,
    segment_id: str | None = None,
    *,
    schema: GoLLIESchema | None = None,
) -> ParseResult:
    """Parse one GoLLIE constructor list without executing it."""
    payload = _list_payload(text)
    if not payload:
        return ParseResult(
            (),
            (ParseIssue("EMPTY_STRUCTURE", "GoLLIE output is empty"),),
            gollie_records=(),
        )
    try:
        tree = ast.parse(payload, filename="<gollie-output>", mode="eval")
        _assert_safe_tree(tree)
    except SyntaxError as exc:
        return ParseResult(
            (),
            (
                ParseIssue(
                    "INVALID_GOLLIE_SYNTAX",
                    f"GoLLIE output is not valid Python syntax: {exc.msg}",
                    "hard",
                ),
            ),
            gollie_records=(),
        )
    except _GoLLIEParseError as exc:
        return ParseResult(
            (),
            (ParseIssue(exc.code, str(exc), "hard"),),
            gollie_records=(),
        )

    if not isinstance(tree.body, ast.List):
        return ParseResult(
            (),
            (
                ParseIssue(
                    "INVALID_GOLLIE_SYNTAX",
                    "GoLLIE output must be a list of class constructors",
                    "hard",
                ),
            ),
            gollie_records=(),
        )

    schema_index = schema.class_by_name() if schema is not None else {}
    records: list[ParsedGoLLIERecord] = []
    issues: list[ParseIssue] = []
    for element in tree.body.elts:
        if not isinstance(element, ast.Call) or not isinstance(element.func, ast.Name):
            issues.append(
                ParseIssue(
                    "UNSAFE_AST_NODE",
                    f"List element is not a class constructor: {type(element).__name__}",
                    "hard",
                )
            )
            continue
        class_name = element.func.id
        schema_class = schema_index.get(class_name)
        schema_known = schema_class is not None
        if not schema_known:
            issues.append(
                ParseIssue(
                    "UNKNOWN_CLASS",
                    f"Unknown class: {class_name}",
                    "soft",
                )
            )
        try:
            arguments, argument_issues = _bind_arguments(element, schema_class)
        except _GoLLIEParseError as exc:
            issues.append(ParseIssue(exc.code, str(exc), "hard"))
            continue
        issues.extend(argument_issues)
        records.append(
            ParsedGoLLIERecord(
                kind=schema_class.kind if schema_class is not None else "unknown",
                class_name=class_name,
                arguments=arguments,
                schema_known=schema_known,
                segment_id=segment_id,
            )
        )
    if not records and not any(issue.code == "UNSAFE_AST_NODE" for issue in issues):
        issues.append(ParseIssue("EMPTY_STRUCTURE", "GoLLIE list contains no constructors"))
    return ParseResult((), tuple(issues), gollie_records=tuple(records))


def _align_string(value: str, source_text: str, segment_start: int) -> tuple[int | None, int | None, str]:
    starts = [match.start() for match in re.finditer(re.escape(value), source_text)]
    if len(starts) == 1:
        start = segment_start + starts[0]
        return start, start + len(value), "exact"
    if starts:
        return None, None, "ambiguous"
    return None, None, "not_found"


def align_gollie_records(
    records: tuple[ParsedGoLLIERecord, ...],
    source_text: str,
    segment_start: int = 0,
) -> tuple[tuple[ParsedGoLLIERecord, ...], tuple[ParseIssue, ...]]:
    """Exact, case-sensitive argument alignment; never force a first match."""
    aligned: list[ParsedGoLLIERecord] = []
    issues: list[ParseIssue] = []
    for index, record in enumerate(records):
        arguments: list[ParsedGoLLIEArgument] = []
        statuses: list[str] = []
        for argument in record.arguments:
            if isinstance(argument.value, str) and argument.value:
                start, end, status = _align_string(
                    argument.value,
                    source_text,
                    segment_start,
                )
                if status != "exact":
                    issues.append(
                        ParseIssue(
                            "UNKNOWN_SPAN",
                            f"{record.class_name}.{argument.name} span is {status}: {argument.value}",
                            "soft",
                            index,
                        )
                    )
            else:
                start, end, status = None, None, "not_attempted"
            statuses.append(status)
            arguments.append(
                replace(
                    argument,
                    alignment_status=status,
                    span_start=start,
                    span_end=end,
                )
            )
        if statuses and all(status == "exact" for status in statuses):
            record_status = "exact"
        elif any(status == "ambiguous" for status in statuses):
            record_status = "ambiguous"
        elif any(status == "not_found" for status in statuses):
            record_status = "not_found"
        else:
            record_status = "not_attempted"
        aligned.append(
            replace(
                record,
                arguments=tuple(arguments),
                alignment_status=record_status,
            )
        )
    return tuple(aligned), tuple(issues)


def _argument_map(record: ParsedGoLLIERecord) -> dict[str, ParsedGoLLIEArgument]:
    return {
        argument.name: argument
        for argument in record.arguments
        if argument.name is not None
    }


def project_gollie_relations(
    records: tuple[ParsedGoLLIERecord, ...],
) -> tuple[ParsedTriple, ...]:
    """Project only schema-declared binary Relation records to triples."""
    triples: list[ParsedTriple] = []
    for record in records:
        if not record.schema_known or record.kind != KIND_RELATION:
            continue
        arguments = _argument_map(record)
        left = arguments.get("arg1")
        right = arguments.get("arg2")
        if left is None or right is None:
            continue
        if not isinstance(left.value, str) or not isinstance(right.value, str):
            continue
        if not left.value or not right.value:
            continue
        subject_span = (
            (left.span_start, left.span_end)
            if left.span_start is not None and left.span_end is not None
            else None
        )
        object_span = (
            (right.span_start, right.span_end)
            if right.span_start is not None and right.span_end is not None
            else None
        )
        triples.append(
            ParsedTriple(
                subject=left.value,
                relation=record.class_name,
                object=right.value,
                subject_span=subject_span,
                object_span=object_span,
                span_source=(
                    "gollie_exact_string_alignment"
                    if left.alignment_status == "exact" and right.alignment_status == "exact"
                    else (
                        "gollie_structure_alignment:"
                        f"{left.alignment_status}/{right.alignment_status}"
                    )
                ),
                segment_id=record.segment_id,
            )
        )
    return tuple(triples)
