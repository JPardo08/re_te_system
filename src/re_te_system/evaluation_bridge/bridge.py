"""Offline evaluation bridge from a finished campaign directory to P0.

The bridge does not run extraction, does not submit work to a scheduler, and
does not select among the three estimands. A synthetic campaign can receive
contract numbers and still cannot be a scientific F1.

Invoke it without editing this source::

    python -m re_te_system.evaluation_bridge \\
        --campaign-dir CAMPAIGN --gold GOLD.jsonl --output-dir OUT

``OUT`` must not be inside ``CAMPAIGN``. The same call is
``evaluate_campaign(BridgeRequest(...))``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from re_te_system.controlled_experiment.contracts import STATUS_PUBLISHABLE, SYNTHETIC_CONTRACT_TEST
from re_te_system.controlled_experiment.persistence import atomic_write_json
from re_te_system.evaluation_bridge.contracts import (
    BRIDGE_CONTRACT_VERSION,
    ESTIMANDS,
    BridgeError,
    BridgeRequest,
)
from re_te_system.evaluation_bridge.integrity import inspect_campaign
from re_te_system.evaluation_bridge.scoring import frozen_config, model_dirname, score_inspection


def evaluate_campaign(request: BridgeRequest) -> dict[str, Any]:
    """Validate ``request`` and write ``campaign_evaluation_report.json``."""
    output_dir = Path(request.output_dir)
    campaign_dir = Path(request.campaign_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _is_inside(output_dir, campaign_dir):
        raise BridgeError("output_dir must not be inside the campaign directory")
    found = inspect_campaign(request)
    estimands = score_inspection(request, found)
    report = _report(found, estimands)
    atomic_write_json(output_dir / "campaign_evaluation_report.json", report)
    return report


def _report(found: Any, estimands: dict[str, Any]) -> dict[str, Any]:
    config = frozen_config()
    identity = found.identity or {}
    manifest = found.manifest or {}
    scientific = _scientific(found)
    return {
        "blockers": list(found.blockers),
        "bridge_contract_version": BRIDGE_CONTRACT_VERSION,
        "campaign_scientific_id": manifest.get("campaign_scientific_id"),
        "campaign_status": {
            "matches_recorded": (
                found.recomputed_status is not None and "campaign_status_mismatch" not in found.blockers
            ),
            "recomputed": found.recomputed_status,
        },
        "comparability_source": "campaign_registry.validator synthetic and comparability rules",
        "coverage": found.coverage,
        "estimand_selection": {
            "published_definitions": list(ESTIMANDS),
            "selected_after_observing_scores": False,
            "selected_estimand": None,
        },
        "estimands": estimands,
        "evaluation_identity": {
            "bridge_contract_version": BRIDGE_CONTRACT_VERSION,
            "campaign_contract_version": identity.get("campaign_contract_version"),
            "campaign_scientific_id": manifest.get("campaign_scientific_id"),
            "config": config.identity_payload(),
            "config_id": config.config_id,
            "implementation_fingerprint": manifest.get("implementation_fingerprint"),
            "p0_interface": "read_records+evaluate_triplets+write_report",
            "result_role": identity.get("evaluation_policy", {}).get("result_role")
            if isinstance(identity.get("evaluation_policy"), dict)
            else None,
        },
        "hashes": found.hashes,
        "model": {"dirname": model_dirname(found), "identity": found.model_identity},
        "normalization": found.normalization,
        "preservation": {
            key: value
            for key, value in found.preservation.items()
            if key != "final_attempts"
        },
        "publication_eligibility": {
            "comparability": found.comparability,
            "experimental_f1": scientific,
            "reasons": list(found.scientific_reasons),
            "registry_protocol_id": found.registry_protocol_id,
            "scientific_result": scientific,
            "technically_scorable": found.technically_scorable,
        },
        "scientific_reasons": list(found.scientific_reasons),
    }


def _scientific(found: Any) -> bool:
    """Scientific F1 requires a clean, comparable, technically scorable registry row."""
    if found.blockers or found.scientific_reasons:
        return False
    if found.mode == SYNTHETIC_CONTRACT_TEST or found.comparability != "scientific_comparable":
        return False
    if found.technically_scorable is not True:
        return False
    status = (found.recomputed_status or {}).get("status")
    if status != STATUS_PUBLISHABLE or found.contract_abort or found.truncated_input:
        return False
    return True


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
