"""Plan-level constants for the campaign registry.

These names are intentionally distinct from the principal states in
``controlled_experiment.contracts``.
"""

from __future__ import annotations

from typing import Any, Mapping

REGISTRY_CONTRACT_VERSION = "exp3c1-1"

PLAN_STATES = (
    "PLANNED",
    "PREPARED",
    "BLOCKED",
    "AWAITING_RESOURCES",
    "AWAITING_AUTHORIZATION",
    "RUNNING",
    "VALIDATED",
    "FAILED",
)
CLAIMING_STATES = frozenset({"PREPARED", "RUNNING", "VALIDATED"})
PROGRAMS = frozenset({"SOTA", "CONDITIONING"})
AUTHORIZATIONS = frozenset({"not_authorized", "authorized"})
COMPARABILITIES = frozenset(
    {"scientific_comparable", "diagnostic_only", "not_comparable"}
)
LOCKED_CONDITIONS = frozenset({"C2", "C3", "C3-NL"})
EXECUTOR_KINDS = frozenset({"existing_sbatch", "not_versioned", "none"})

STRUCTURALLY_VALID = "STRUCTURALLY_VALID"
NOT_READY = "NOT_READY"
INVALID = "INVALID"


def schema_errors(instance: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Check a small JSON Schema subset: type, required, properties, enum, items.

    ``additionalProperties: false`` is enforced. ``$ref`` and combinators other
    than a list of types are not. The Python validator remains the authority
    for cross-file rules.
    """
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None and not _type_matches(instance, expected):
        errors.append(f"{path} has type {_type_name(instance)}, schema wants {expected}")
        return errors
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path} value {instance!r} is outside {schema['enum']}")
    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                errors.extend(schema_errors(value, properties[key], f"{path}.{key}"))
            elif additional is False:
                errors.append(f"{path}.{key} is not allowed")
    if isinstance(instance, list):
        minimum = schema.get("minItems")
        if isinstance(minimum, int) and len(instance) < minimum:
            errors.append(f"{path} needs at least {minimum} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, value in enumerate(instance):
                errors.extend(schema_errors(value, item_schema, f"{path}[{index}]"))
    return errors


def _type_matches(instance: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(instance, item) for item in expected)
    checks = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "boolean": lambda value: isinstance(value, bool),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": lambda value: value is None,
    }
    check = checks.get(expected)
    return False if check is None else check(instance)


def _type_name(instance: Any) -> str:
    if instance is None:
        return "null"
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, dict):
        return "object"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if isinstance(instance, str):
        return "string"
    return type(instance).__name__
