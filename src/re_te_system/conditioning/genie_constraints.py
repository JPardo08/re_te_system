"""Clean-room GenIE constraint profiles, token tries, and structural state machine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROFILE_UNCONSTRAINED = "unconstrained"
PROFILE_SMALL = "small"
PROFILE_LARGE = "large"
PROFILE_CUSTOM_FULL = "custom_full"
PROFILE_CLOSED_SCHEMA = "closed_schema"

NATIVE_CONSTRAINT_PROFILES = frozenset(
    {
        PROFILE_UNCONSTRAINED,
        PROFILE_SMALL,
        PROFILE_LARGE,
        PROFILE_CUSTOM_FULL,
        PROFILE_CLOSED_SCHEMA,
    }
)
CONSTRAINED_PROFILES = NATIVE_CONSTRAINT_PROFILES - {PROFILE_UNCONSTRAINED}
NON_NATIVE_RELATION_ONLY = "relation_only"
NON_NATIVE_ABLATION = "NON_NATIVE_ABLATION"

STATE_OUTSIDE = "outside"
STATE_SUBJECT = "entity_subject"
STATE_RELATION = "relation"
STATE_OBJECT = "entity_object"

TAG_SUB = "sub"
TAG_REL = "rel"
TAG_OBJ = "obj"
TAG_ET = "et"
CONTROL_TAGS = (TAG_SUB, TAG_REL, TAG_OBJ, TAG_ET)

OFFICIAL_STRING_INVENTORY_FILES = {
    "small_entity": "tries/small/entity_trie_original_strings.jsonl",
    "small_relation": "tries/small/relation_trie_original_strings.jsonl",
    "large_entity": "tries/large/entity_trie_original_strings.jsonl",
    "large_relation": "tries/large/relation_trie_original_strings.jsonl",
}
OFFICIAL_TRIE_DESERIALIZATION = "RECONSTRUCT_FROM_STRINGS_PREFERRED"


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text)
        if isinstance(encoded, Mapping):
            return list(encoded.get("input_ids", []))
        return list(encoded)
    encoded = tokenizer(text)
    return list(encoded.get("input_ids", []))


class TokenPrefixTrie:
    """Deterministic token-prefix trie. Does not unpickle official GenIE tries."""

    def __init__(self, sequences: Iterable[Sequence[int]]) -> None:
        next_sets: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        for sequence in sequences:
            if sequence:
                next_sets[int(sequence[0])].append(tuple(int(item) for item in sequence[1:]))
        self._leaves = {key: TokenPrefixTrie(values) for key, values in next_sets.items()}

    def get(self, prefix: Sequence[int]) -> list[int]:
        if not prefix:
            return sorted(self._leaves)
        head = int(prefix[0])
        if head not in self._leaves:
            return []
        return self._leaves[head].get(prefix[1:])

    @classmethod
    def from_strings(
        cls,
        names: Iterable[str],
        tokenizer: Any,
        *,
        add_leading_space: bool = True,
        remove_leading_bos: bool = True,
    ) -> "TokenPrefixTrie":
        sequences: list[list[int]] = []
        for name in sorted(names):
            text = f" {name}" if add_leading_space else name
            ids = _encode_ids(tokenizer, text)
            if remove_leading_bos and ids:
                ids = ids[1:]
            sequences.append(ids)
        return cls(sequences)


@dataclass(frozen=True)
class StructuralCodes:
    bos_token_id: int
    eos_token_id: int
    start_of_tag: int
    end_of_tag: int
    subject_token: int
    relation_token: int
    object_token: int
    end_of_triple_token: int
    bos_as_first_token_generated: bool = True

    @property
    def tag_ids(self) -> set[int]:
        return {
            self.subject_token,
            self.relation_token,
            self.object_token,
            self.end_of_triple_token,
        }

    def tag_for_state(self, state: str) -> int:
        return {
            STATE_OUTSIDE: self.subject_token,
            STATE_SUBJECT: self.relation_token,
            STATE_RELATION: self.object_token,
            STATE_OBJECT: self.end_of_triple_token,
        }[state]


@dataclass(frozen=True)
class GenIEConstraintSpec:
    profile: str
    entity_inventory: tuple[str, ...] = ()
    relation_inventory: tuple[str, ...] = ()
    official_inventory_ref: str | None = None

    def __post_init__(self) -> None:
        if self.profile == NON_NATIVE_RELATION_ONLY:
            raise ValueError(
                "RELATION_ONLY is not a native GenIE profile; a future "
                "relation-only constraint would be NON_NATIVE_ABLATION"
            )
        if self.profile not in NATIVE_CONSTRAINT_PROFILES:
            raise ValueError(f"unsupported GenIE constraint profile: {self.profile}")
        if self.profile in {PROFILE_CUSTOM_FULL, PROFILE_CLOSED_SCHEMA}:
            if not self.entity_inventory or not self.relation_inventory:
                raise ValueError(
                    "official constrained decoding requires both an entity "
                    "inventory and a relation inventory"
                )

    @property
    def constrained(self) -> bool:
        return self.profile != PROFILE_UNCONSTRAINED

    @property
    def entity_inventory_required(self) -> bool:
        return self.constrained

    @property
    def relation_inventory_required(self) -> bool:
        return self.constrained

    @property
    def ready_for_constrained_decode(self) -> bool:
        return bool(self.entity_inventory) and bool(self.relation_inventory)

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "constraint_profile": self.profile,
            "constrained": self.constrained,
            "entity_inventory_required": self.entity_inventory_required,
            "entity_inventory_size": len(self.entity_inventory),
            "formal_ontology": False,
            "official_inventory_ref": self.official_inventory_ref,
            "ready_for_constrained_decode": self.ready_for_constrained_decode,
            "relation_inventory_required": self.relation_inventory_required,
            "relation_inventory_size": len(self.relation_inventory),
            "semantic_ontology_understanding": False,
        }


def unconstrained_spec() -> GenIEConstraintSpec:
    return GenIEConstraintSpec(PROFILE_UNCONSTRAINED)


def named_schema_spec(profile: str) -> GenIEConstraintSpec:
    if profile not in {PROFILE_SMALL, PROFILE_LARGE}:
        raise ValueError("named schema specs are small or large only")
    return GenIEConstraintSpec(profile, official_inventory_ref=profile)


def closed_schema_spec(
    entities: Iterable[str],
    relations: Iterable[str],
    *,
    profile: str = PROFILE_CUSTOM_FULL,
) -> GenIEConstraintSpec:
    entity_inventory = tuple(entities)
    relation_inventory = tuple(relations)
    if not entity_inventory or not relation_inventory:
        raise ValueError(
            "official constrained decoding requires both an entity "
            "inventory and a relation inventory"
        )
    return GenIEConstraintSpec(
        profile,
        entity_inventory=entity_inventory,
        relation_inventory=relation_inventory,
    )


def structural_state(generated_ids: Sequence[int], codes: StructuralCodes) -> str:
    complete_tags = 0
    index = 0
    while index < len(generated_ids) - 2:
        if (
            generated_ids[index] == codes.start_of_tag
            and generated_ids[index + 1] in codes.tag_ids
            and generated_ids[index + 2] == codes.end_of_tag
        ):
            complete_tags += 1
            index += 3
            continue
        index += 1
    return (STATE_OUTSIDE, STATE_SUBJECT, STATE_RELATION, STATE_OBJECT)[complete_tags % 4]


def last_complete_tag_end(generated_ids: Sequence[int], codes: StructuralCodes) -> int | None:
    index = len(generated_ids) - 3
    while index >= 0:
        if (
            generated_ids[index] == codes.start_of_tag
            and generated_ids[index + 1] in codes.tag_ids
            and generated_ids[index + 2] == codes.end_of_tag
        ):
            return index + 2
        index -= 1
    return None


def allowed_next_token_ids(
    generated_ids: Sequence[int],
    codes: StructuralCodes,
    *,
    entity_trie: TokenPrefixTrie | None = None,
    relation_trie: TokenPrefixTrie | None = None,
) -> list[int]:
    """Token-prefix structural + inventory filter. No semantic type checks."""
    if len(generated_ids) > 1 and generated_ids[-1] == codes.eos_token_id:
        return []
    if codes.bos_as_first_token_generated and len(generated_ids) == 1:
        return [codes.bos_token_id]
    state = structural_state(generated_ids, codes)
    if generated_ids and generated_ids[-1] == codes.start_of_tag:
        return [codes.tag_for_state(state)]
    if (
        len(generated_ids) > 1
        and generated_ids[-2] == codes.start_of_tag
    ):
        if generated_ids[-1] in codes.tag_ids:
            return [codes.end_of_tag]
        return []
    if state == STATE_OUTSIDE:
        return sorted({codes.start_of_tag, codes.eos_token_id})
    trie = entity_trie if state in {STATE_SUBJECT, STATE_OBJECT} else relation_trie
    if trie is None:
        raise ValueError(
            "official constrained decoding requires both entity and relation tries"
        )
    tag_end = last_complete_tag_end(generated_ids, codes)
    prefix = [] if tag_end is None else list(generated_ids[tag_end + 1 :])
    allowed = list(trie.get(prefix))
    if codes.eos_token_id in allowed:
        allowed = [token for token in allowed if token != codes.eos_token_id]
        allowed.append(codes.start_of_tag)
    return allowed


def official_string_inventory_paths(data_dir: str | Path) -> dict[str, Path]:
    root = Path(data_dir)
    return {key: root / relative for key, relative in OFFICIAL_STRING_INVENTORY_FILES.items()}


def official_string_inventories_available(data_dir: str | Path) -> bool:
    return all(path.is_file() for path in official_string_inventory_paths(data_dir).values())
