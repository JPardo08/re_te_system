"""Disabled design for a one-time UIE safetensors conversion.

The origin pickle stays in the Hugging Face snapshot. This module does not
load it. PyTorch 2.5.1 is not a legal runtime for that load. Transformers'
torch.load gate stays enabled; this code never sets weights_only=False.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from uie_origin_pin import (
    CONVERSION_ENABLED,
    ORIGIN_BIN_SHA256,
    ORIGIN_BIN_SIZE,
    deserialization_authorized,
    origin_identity_matches,
)


class ConversionDisabled(RuntimeError):
    """The exceptional conversion has not been authorized."""


MINIMUM_DESERIALIZATION_TORCH = (2, 6)
INTENDED_LOAD = "torch.load(path, map_location='cpu', weights_only=True)"
CONVERSION_REQUIREMENTS = {
    "enabled": False,
    "deserialization_authorized": False,
    "independent_environment": True,
    "minimum_torch": "2.6",
    "forbidden_runtime": "2.5.1",
    "weights_only": True,
    "transformers_security_check": "must_remain_enabled",
    "mutates_origin_snapshot": False,
    "intended_load": INTENDED_LOAD,
}


def torch_version_can_deserialize_origin(version: str) -> bool:
    """Version predicate only. A true result is not a load authorization."""
    numeric = version.split("+", 1)[0]
    pieces = []
    for piece in numeric.split("."):
        if not piece.isdigit():
            break
        pieces.append(int(piece))
    while len(pieces) < 3:
        pieces.append(0)
    return tuple(pieces[:3]) >= MINIMUM_DESERIALIZATION_TORCH


def assert_conversion_runtime(torch_version: str) -> None:
    if not CONVERSION_ENABLED or not deserialization_authorized():
        raise ConversionDisabled(
            "origin SHA-256 match does not authorize deserialization; "
            "conversion is disabled"
        )
    if torch_version.split("+", 1)[0].startswith("2.5.1") or not torch_version_can_deserialize_origin(
        torch_version
    ):
        raise ConversionDisabled("PyTorch 2.5.1 must not open pytorch_model.bin")


def convert_origin_to_safetensors(
    origin_bin: Path,
    destination: Path,
    *,
    loader: object = None,
) -> dict[str, object]:
    """Refuse before any loader runs. The origin snapshot is not a destination."""
    del loader
    if destination.resolve() == origin_bin.resolve() or destination.resolve().is_relative_to(
        origin_bin.resolve().parent
    ):
        raise ConversionDisabled("refusing to write inside the origin snapshot")
    if origin_bin.name != "pytorch_model.bin":
        raise ConversionDisabled("conversion input must be the preserved pytorch_model.bin")
    assert_conversion_runtime("2.6.0")
    raise ConversionDisabled("conversion did not pass the authorization gate")


def compare_tensor_records(
    origin: Sequence[Mapping[str, object]],
    derived: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare names, shapes, dtypes, values, and shared-storage groups.

    Records are plain mappings. This function does not read a checkpoint.
    """
    def index(records: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
        return {str(item["name"]): item for item in records}

    left = index(origin)
    right = index(derived)
    names_equal = set(left) == set(right)
    shapes_equal = names_equal and all(left[name]["shape"] == right[name]["shape"] for name in left)
    dtypes_equal = names_equal and all(left[name]["dtype"] == right[name]["dtype"] for name in left)
    values_equal = names_equal and all(
        left[name]["values_sha256"] == right[name]["values_sha256"] for name in left
    )

    def groups(records: dict[str, Mapping[str, object]]) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = {}
        for name, item in records.items():
            grouped.setdefault(str(item["storage_id"]), set()).add(name)
        return {key: names for key, names in grouped.items() if len(names) > 1}

    shared_equal = names_equal and groups(left) == groups(right)
    return {
        "dtypes_equal": dtypes_equal,
        "equivalent": names_equal and shapes_equal and dtypes_equal and values_equal and shared_equal,
        "names_equal": names_equal,
        "shapes_equal": shapes_equal,
        "shared_storage_groups_equal": shared_equal,
        "values_equal": values_equal,
    }


def load_equivalence(
    *,
    origin_bin_sha256: str,
    derived_file_sha256: str,
    reported_loaded_sha256: str,
) -> dict[str, bool]:
    """A reported load matches the derived file and is not the origin pickle."""
    derived_match = reported_loaded_sha256 == derived_file_sha256
    origin_not_loaded = reported_loaded_sha256 != origin_bin_sha256
    distinct = origin_bin_sha256 != derived_file_sha256
    return {
        "derived_file_matches_reported_load": derived_match,
        "equivalent": derived_match and origin_not_loaded and distinct,
        "origin_bin_was_not_the_loaded_file": origin_not_loaded,
    }


def derived_provenance_accepted(
    identity: Mapping[str, object] | None,
    provenance: Mapping[str, object] | None,
    origin_observation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Recompute pin identity, tensor equivalence, and load identity.

    A boolean claim is not evidence. Tensor records are compared here. The
    derived SHA-256 is the one measured on model.safetensors. If the origin
    file was actually read and it is not the pin, the claim is rejected.
    """
    measured = _measured_derived_sha(identity)
    origin_size = provenance.get("origin_size") if isinstance(provenance, Mapping) else None
    origin_sha = provenance.get("origin_sha256") if isinstance(provenance, Mapping) else None
    origin_ok = (
        isinstance(origin_size, int)
        and isinstance(origin_sha, str)
        and origin_identity_matches(origin_size, origin_sha)
    )
    if isinstance(origin_observation, Mapping) and origin_observation.get("observed") is True:
        origin_ok = origin_ok and origin_observation.get("matches_pin") is True
    compared = _compared_records(provenance)
    reported = provenance.get("reported_loaded_sha256") if isinstance(provenance, Mapping) else None
    if isinstance(origin_sha, str) and isinstance(measured, str) and isinstance(reported, str):
        loaded = load_equivalence(
            origin_bin_sha256=origin_sha,
            derived_file_sha256=measured,
            reported_loaded_sha256=reported,
        )
    else:
        loaded = {
            "derived_file_matches_reported_load": False,
            "equivalent": False,
            "origin_bin_was_not_the_loaded_file": False,
        }
    identity_ok = isinstance(identity, Mapping) and identity.get("ok") is True and isinstance(measured, str)
    return {
        "accepted": bool(identity_ok and origin_ok and compared["equivalent"] and loaded["equivalent"]),
        "equivalence": compared,
        "load": loaded,
        "origin_matches_pin": origin_ok,
    }


def _measured_derived_sha(identity: Mapping[str, object] | None) -> str | None:
    if not isinstance(identity, Mapping):
        return None
    files = identity.get("files")
    if not isinstance(files, Mapping):
        return None
    item = files.get("model.safetensors")
    if not isinstance(item, Mapping):
        return None
    digest = item.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    return digest


def _compared_records(provenance: Mapping[str, object] | None) -> dict[str, object]:
    empty = {
        "dtypes_equal": False,
        "equivalent": False,
        "names_equal": False,
        "shapes_equal": False,
        "shared_storage_groups_equal": False,
        "values_equal": False,
    }
    if not isinstance(provenance, Mapping):
        return empty
    origin_tensors = provenance.get("origin_tensors")
    derived_tensors = provenance.get("derived_tensors")
    if not isinstance(origin_tensors, list) or not isinstance(derived_tensors, list):
        return empty
    if not origin_tensors or not derived_tensors:
        return empty
    try:
        return compare_tensor_records(origin_tensors, derived_tensors)
    except (KeyError, TypeError):
        return empty


def origin_pin_summary() -> dict[str, object]:
    return {
        "deserialization_authorized": deserialization_authorized(),
        "matches_known_pin": origin_identity_matches(ORIGIN_BIN_SIZE, ORIGIN_BIN_SHA256),
        "sha256": ORIGIN_BIN_SHA256,
        "size": ORIGIN_BIN_SIZE,
    }
