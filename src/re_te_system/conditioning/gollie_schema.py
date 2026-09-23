"""Baseline-specific GoLLIE schema, guideline fields, and prompt serializer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


GOLLIE_SCHEMA_VERSION = "gollie-schema-v1"
GOLLIE_PROMPT_SERIALIZER_VERSION = "gollie-prompt-v1"
GOLLIE_ORDERING_POLICY = "schema_order"
GOLLIE_BLACK_LINE_LENGTH = 119
GOLLIE_BLACK_TARGET_VERSION = "black>=24.8,<25"
KIND_ENTITY = "entity"
KIND_RELATION = "relation"
KIND_EVENT = "event"
KIND_TEMPLATE = "template"
VALID_KINDS = {KIND_ENTITY, KIND_RELATION, KIND_EVENT, KIND_TEMPLATE}


@dataclass(frozen=True)
class GoLLIEArgument:
    name: str
    type_name: str
    guideline: str | None = None
    examples: tuple[str, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("argument name must be non-empty")
        if not self.type_name or not self.type_name.strip():
            raise ValueError("argument type_name must be non-empty")


@dataclass(frozen=True)
class GoLLIEClass:
    name: str
    kind: str
    base: str
    definition: str
    arguments: tuple[GoLLIEArgument, ...]
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("class name must be non-empty")
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unsupported GoLLIE class kind: {self.kind}")
        if not self.base or not self.base.strip():
            raise ValueError("class base must be non-empty")
        names = [argument.name for argument in self.arguments]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate argument names in {self.name}")

    def field_guidelines(self) -> dict[str, str]:
        return {
            argument.name: argument.guideline
            for argument in self.arguments
            if argument.guideline
        }


@dataclass(frozen=True)
class GoLLIESchema:
    schema_id: str
    classes: tuple[GoLLIEClass, ...]
    schema_version: str = GOLLIE_SCHEMA_VERSION
    prompt_serializer_version: str = GOLLIE_PROMPT_SERIALIZER_VERSION

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("schema_id must be non-empty")
        names = [item.name for item in self.classes]
        if any(not name for name in names):
            raise ValueError("class names must be non-empty")
        if len(names) != len(set(names)):
            raise ValueError("schema classes must not contain duplicate names")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GoLLIESchema":
        classes = []
        for item in value.get("classes", []):
            arguments = tuple(
                GoLLIEArgument(
                    name=str(argument["name"]),
                    type_name=str(argument["type_name"]),
                    guideline=(
                        None
                        if argument.get("guideline") is None
                        else str(argument["guideline"])
                    ),
                    examples=tuple(str(example) for example in argument.get("examples", [])),
                    required=bool(argument.get("required", True)),
                )
                for argument in item.get("arguments", [])
            )
            classes.append(
                GoLLIEClass(
                    name=str(item["name"]),
                    kind=str(item["kind"]),
                    base=str(item["base"]),
                    definition=str(item.get("definition", "")),
                    arguments=arguments,
                    examples=tuple(str(example) for example in item.get("examples", [])),
                )
            )
        return cls(
            schema_id=str(value["schema_id"]),
            classes=tuple(classes),
            schema_version=str(value.get("schema_version", GOLLIE_SCHEMA_VERSION)),
            prompt_serializer_version=str(
                value.get("prompt_serializer_version", GOLLIE_PROMPT_SERIALIZER_VERSION)
            ),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "GoLLIESchema":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def class_by_name(self) -> dict[str, GoLLIEClass]:
        return {item.name: item for item in self.classes}

    def structural_representation(self) -> dict[str, Any]:
        return {
            "classes": [
                {
                    "arguments": [
                        {
                            "name": argument.name,
                            "required": argument.required,
                            "type_name": argument.type_name,
                        }
                        for argument in item.arguments
                    ],
                    "base": item.base,
                    "kind": item.kind,
                    "name": item.name,
                }
                for item in self.classes
            ],
            "ordering_policy": GOLLIE_ORDERING_POLICY,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
        }

    def guideline_representation(self) -> dict[str, Any]:
        return {
            "classes": [
                {
                    "definition": item.definition,
                    "examples": list(item.examples),
                    "field_examples": {
                        argument.name: list(argument.examples)
                        for argument in item.arguments
                        if argument.examples
                    },
                    "field_guidelines": item.field_guidelines(),
                    "name": item.name,
                }
                for item in self.classes
            ],
            "schema_id": self.schema_id,
        }

    def _hash(self, payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def schema_hash(self) -> str:
        return self._hash(self.structural_representation())

    def guideline_hash(self) -> str:
        return self._hash(self.guideline_representation())

    def _class_source(self, item: GoLLIEClass) -> str:
        lines = ["@dataclass", f"class {item.name}({item.base}):"]
        if item.definition:
            lines.append(f'    """{item.definition}"""')
            lines.append("")
        if not item.arguments:
            lines.append("    pass")
            return "\n".join(lines)
        for argument in item.arguments:
            line = f"    {argument.name}: {argument.type_name}"
            comment_parts: list[str] = []
            if argument.guideline:
                comment_parts.append(argument.guideline)
            if argument.examples:
                quoted = ", ".join(f'"{example}"' for example in argument.examples)
                comment_parts.append(f'Such as: {quoted}')
            if comment_parts:
                line = f"{line}  # {'; '.join(comment_parts)}"
            lines.append(line)
        return "\n".join(lines)

    def render_unformatted_prompt(self, text: str) -> str:
        class_block = "\n\n".join(self._class_source(item) for item in self.classes)
        if class_block:
            class_block = "\n" + class_block + "\n"
        return (
            "# The following lines describe the task definition"
            f"{class_block}\n"
            "# This is the text to analyze\n"
            f"text = {text!r}\n\n"
            "# The annotation instances that take place in the text above are listed here\n"
            "result = []\n"
        )

    def _black_version(self) -> str:
        import black

        return getattr(black, "__version__", "unknown")

    def serialize_prompt(self, text: str) -> str:
        """Render the inference prefix using Black as the formatting contract."""
        try:
            import black
        except ImportError as exc:
            raise ImportError(
                'Install optional dependencies with "pip install -e .[gollie]"'
            ) from exc

        formatted = black.format_str(
            self.render_unformatted_prompt(text),
            mode=black.Mode(
                line_length=GOLLIE_BLACK_LINE_LENGTH,
                target_versions={black.TargetVersion.PY311},
            ),
        )
        prefix, _separator, _suffix = formatted.partition("result =")
        return prefix + "result ="

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "black_line_length": GOLLIE_BLACK_LINE_LENGTH,
            "black_policy": "explicit_black_serialization_contract",
            "black_target_version": GOLLIE_BLACK_TARGET_VERSION,
            "black_version": self._black_version(),
            "definitions_guidelines": True,
            "formal_ontology": False,
            "guideline_hash": self.guideline_hash(),
            "guidelines": self.guideline_representation(),
            "ordering_policy": GOLLIE_ORDERING_POLICY,
            "prompt_serializer_version": self.prompt_serializer_version,
            "schema_hash": self.schema_hash(),
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "structural_schema": self.structural_representation(),
            "target_schema_knowledge_components": [
                "structural_schema",
                "definitions_guidelines",
            ],
        }


def official_re_ana_mary_schema() -> GoLLIESchema:
    """Exact audited official Relation Extraction notebook schema."""
    return GoLLIESchema(
        schema_id="official-re-ana-mary",
        classes=(
            GoLLIEClass(
                name="PhysicalRelation",
                kind=KIND_RELATION,
                base="Relation",
                definition=(
                    "The Physical Relation captures the physical location relation of entities such as:\n"
                    "    a Person entity located in a Facility, Location or GPE; or two entities that are near,\n"
                    "    but neither entity is a part of the other or located in/at the other."
                ),
                arguments=(
                    GoLLIEArgument(name="arg1", type_name="str"),
                    GoLLIEArgument(name="arg2", type_name="str"),
                ),
            ),
            GoLLIEClass(
                name="PersonalSocialRelation",
                kind=KIND_RELATION,
                base="Relation",
                definition=(
                    "The Personal-Social Relation describe the relationship between people. Both arguments must be entities\n"
                    "    of type Person. Please note: The arguments of these Relations are not ordered. The Relations are\n"
                    "    symmetric."
                ),
                arguments=(
                    GoLLIEArgument(name="arg1", type_name="str"),
                    GoLLIEArgument(name="arg2", type_name="str"),
                ),
            ),
        ),
    )


OFFICIAL_RE_ANA_MARY_TEXT = (
    "Ana and Mary are sisters. Mary was at the supermarket while Ana was at home."
)
