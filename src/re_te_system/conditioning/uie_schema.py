"""Baseline-specific UIE structural schema and deterministic SSI construction."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


UIE_SCHEMA_VERSION = "uie-structural-schema-v1"
UIE_SSI_VERSION = "uie-ssi-v1"
UIE_ORDERING_POLICY = "schema_order"

TYPE_START = "<extra_id_0>"
TYPE_END = "<extra_id_1>"
TEXT_START = "<extra_id_2>"
SPAN_START = "<extra_id_5>"
NULL_SPAN = "<extra_id_6>"
SPOT_PROMPT = "<spot>"
ASOC_PROMPT = "<asoc>"


@dataclass(frozen=True)
class UIESchema:
    schema_id: str
    spot_labels: tuple[str, ...]
    association_labels: tuple[str, ...]
    spot_to_association: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    schema_version: str = UIE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("schema_id must be non-empty")
        for name, labels in (
            ("spot_labels", self.spot_labels),
            ("association_labels", self.association_labels),
        ):
            if any(not isinstance(label, str) or not label.strip() for label in labels):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(labels) != len(set(labels)):
                raise ValueError(f"{name} must not contain duplicates")
        spot_set = set(self.spot_labels)
        association_set = set(self.association_labels)
        for spot, associations in self.spot_to_association.items():
            if spot not in spot_set:
                raise ValueError(f"spot_to_association contains unknown spot: {spot}")
            unknown = set(associations) - association_set
            if unknown:
                raise ValueError(
                    f"spot_to_association contains unknown associations: {sorted(unknown)}"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UIESchema":
        mapping = {
            str(spot): tuple(str(label) for label in labels)
            for spot, labels in value.get("spot_to_association", {}).items()
        }
        return cls(
            schema_id=str(value["schema_id"]),
            spot_labels=tuple(str(label) for label in value.get("spot_labels", [])),
            association_labels=tuple(
                str(label) for label in value.get("association_labels", [])
            ),
            spot_to_association=mapping,
            schema_version=str(value.get("schema_version", UIE_SCHEMA_VERSION)),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "UIESchema":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def stable_representation(self) -> dict[str, Any]:
        return {
            "association_labels": list(self.association_labels),
            "ordering_policy": UIE_ORDERING_POLICY,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "spot_labels": list(self.spot_labels),
            "spot_to_association": {
                spot: list(self.spot_to_association.get(spot, ()))
                for spot in self.spot_labels
                if spot in self.spot_to_association
            },
            "ssi_version": UIE_SSI_VERSION,
        }

    def stable_hash(self) -> str:
        payload = json.dumps(
            self.stable_representation(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def build_ssi(self) -> str:
        spot_prefix = "".join(f"{SPOT_PROMPT} {label}" for label in self.spot_labels)
        association_prefix = "".join(
            f"{ASOC_PROMPT} {label}" for label in self.association_labels
        )
        return f"{spot_prefix}{association_prefix}{TEXT_START} "

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            **self.stable_representation(),
            "schema_hash": self.stable_hash(),
            "serialized_ssi": self.build_ssi(),
        }
