"""Future UIE phase machine. Live flags stay closed.

evaluate_uie_transition accepts hypothetical evidence so the order of gates
can be tested. It does not read or flip the module constants and it does not
load a checkpoint. current_uie_gates is the only view of the live flags.
A J12 technical pass does not grant J13.

The upstream trainer still calls AutoModelForSeq2SeqLM.from_pretrained with
mirror='tuna' in _legacy/UIE/run_uie_finetune.py. Compatibility of that
argument with Transformers 4.57.6 is unverified. This module does not patch
the trainer and does not import Transformers.
"""

from __future__ import annotations

from uie_origin_pin import (
    CONVERSION_ENABLED,
    ORIGIN_BIN_SHA256,
    ORIGIN_BIN_SIZE,
    UIE_J12_DERIVED_LOAD_AUTHORIZED,
    UIE_J13_TRAINING_AUTHORIZED,
    deserialization_authorized,
    origin_identity_matches,
)
from uie_safetensors_conversion import derived_provenance_accepted, torch_version_can_deserialize_origin


UPSTREAM_TRAINER = "_legacy/UIE/run_uie_finetune.py"
UPSTREAM_MIRROR_ARGUMENT = "tuna"
UPSTREAM_MIRROR_COMPATIBILITY = "unverified_against_transformers_4.57.6"


def current_uie_gates() -> dict[str, object]:
    """Live gates. Origin identity of the pin is recorded. No later phase is on."""
    return {
        "conversion_enabled": CONVERSION_ENABLED is True,
        "deserialization_authorized": deserialization_authorized(),
        "j12_derived_load_authorized": UIE_J12_DERIVED_LOAD_AUTHORIZED is True,
        "j13_training_authorized": UIE_J13_TRAINING_AUTHORIZED is True,
        "origin_bin_sha256": ORIGIN_BIN_SHA256,
        "origin_bin_size": ORIGIN_BIN_SIZE,
        "origin_pin_recorded": origin_identity_matches(ORIGIN_BIN_SIZE, ORIGIN_BIN_SHA256),
        "phases_active": False,
        "upstream_mirror_argument": UPSTREAM_MIRROR_ARGUMENT,
        "upstream_mirror_compatibility": UPSTREAM_MIRROR_COMPATIBILITY,
        "upstream_trainer": UPSTREAM_TRAINER,
    }


def _runtime_can_deserialize(
    torch_version: str,
    *,
    security_check_enabled: bool,
    isolated_corrected_environment: bool,
) -> bool:
    if security_check_enabled is not True or isolated_corrected_environment is not True:
        return False
    if torch_version.split("+", 1)[0].startswith("2.5.1"):
        return False
    return torch_version_can_deserialize_origin(torch_version)


def evaluate_uie_transition(
    *,
    origin_matches_pin: bool,
    deserialization_requested: bool,
    isolated_corrected_environment: bool,
    torch_version: str,
    transformers_security_check_enabled: bool,
    conversion_finished: bool,
    derived_identity_ok: bool,
    equivalence_ok: bool,
    load_is_derived_not_origin: bool,
    j12_load_authorized: bool,
    batch_passed: bool,
    j12_job_id: str | None,
    j13_authorized: bool,
    provenance: dict | None = None,
    derived_identity: dict | None = None,
) -> dict[str, object]:
    """Return each phase. Later phases stay false until every earlier one is true.

    Authorization arguments do not replace evidence. Equivalence is recomputed
    from tensor records and from the measured derived SHA-256. Passing true
    here does not change the live module flags.
    """
    declared_size = provenance.get("origin_size") if isinstance(provenance, dict) else None
    declared_sha = provenance.get("origin_sha256") if isinstance(provenance, dict) else None
    origin_identified = (
        origin_matches_pin is True
        and isinstance(declared_size, int)
        and isinstance(declared_sha, str)
        and origin_identity_matches(declared_size, declared_sha)
    )
    deserialization_granted = origin_identified and deserialization_requested is True and _runtime_can_deserialize(
        torch_version,
        security_check_enabled=transformers_security_check_enabled,
        isolated_corrected_environment=isolated_corrected_environment,
    )
    conversion_accepted = deserialization_granted and conversion_finished is True
    provenance_ok = (
        derived_identity_ok is True
        and equivalence_ok is True
        and load_is_derived_not_origin is True
        and derived_provenance_accepted(derived_identity, provenance)["accepted"] is True
    )
    equivalence_accepted = conversion_accepted and provenance_ok
    j12_technical_pass = (
        equivalence_accepted
        and j12_load_authorized is True
        and batch_passed is True
        and isinstance(j12_job_id, str)
        and bool(j12_job_id)
    )
    j13_granted = j12_technical_pass and j13_authorized is True
    return {
        "conversion_accepted": conversion_accepted,
        "derived_identity_ok": derived_identity_ok is True,
        "deserialization_granted": deserialization_granted,
        "equivalence_accepted": equivalence_accepted,
        "j12_technical_pass": j12_technical_pass,
        "j13_granted": j13_granted,
        "j12_pass_authorizes_j13": False,
        "origin_identified": origin_identified,
    }


def derived_load_permitted(*, identity_ok: bool, equivalence_ok: bool) -> bool:
    """Live permission to open a derived directory. Identity alone is not enough."""
    gates = current_uie_gates()
    return bool(
        identity_ok
        and equivalence_ok
        and gates["deserialization_authorized"] is True
        and gates["conversion_enabled"] is True
        and gates["j12_derived_load_authorized"] is True
    )
