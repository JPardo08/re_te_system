"""Offline UIE transition checks for EXP.4.5.1.

Synthetic files and hypothetical phase arguments only. No checkpoint is
deserialized and no GPU is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cesvima"))

import j12_uie_preflight as j12  # noqa: E402
import j13_uie_train as j13  # noqa: E402
import uie_fingerprint as fingerprint  # noqa: E402
import uie_origin_pin as origin  # noqa: E402
import uie_recovery_phases as phases  # noqa: E402
import uie_safetensors_conversion as conversion  # noqa: E402


def _scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    scratch = tmp_path / "scratch"
    runs = scratch / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    monkeypatch.setenv("SLURM_JOB_ID", "1349123")
    return runs


def _shared_tensors() -> list[dict[str, object]]:
    return [
        {"name": "a", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s"},
        {"name": "b", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s"},
    ]


def _provenance(derived_sha: str = "d" * 64) -> dict[str, object]:
    records = _shared_tensors()
    return {
        "derived_tensors": records,
        "origin_sha256": origin.ORIGIN_BIN_SHA256,
        "origin_size": origin.ORIGIN_BIN_SIZE,
        "origin_tensors": records,
        "reported_loaded_sha256": derived_sha,
    }


def _derived_identity(derived_sha: str = "d" * 64) -> dict[str, object]:
    return {
        "errors": [],
        "files": {"model.safetensors": {"sha256": derived_sha, "size": 3}},
        "ok": True,
    }


def _hypothetical(**overrides: object) -> dict[str, object]:
    derived_sha = "d" * 64
    evidence: dict[str, object] = {
        "origin_matches_pin": True,
        "deserialization_requested": True,
        "isolated_corrected_environment": True,
        "torch_version": "2.6.0",
        "transformers_security_check_enabled": True,
        "conversion_finished": True,
        "derived_identity_ok": True,
        "derived_identity": _derived_identity(derived_sha),
        "equivalence_ok": True,
        "load_is_derived_not_origin": True,
        "j12_load_authorized": True,
        "batch_passed": True,
        "j12_job_id": "77",
        "j13_authorized": False,
        "provenance": _provenance(derived_sha),
    }
    evidence.update(overrides)
    return phases.evaluate_uie_transition(**evidence)  # type: ignore[arg-type]


def test_live_gates_record_the_origin_pin_and_activate_nothing() -> None:
    assert origin.ORIGIN_BIN_FILENAME == "pytorch_model.bin"
    assert origin.ORIGIN_BIN_SIZE == 990285657
    assert origin.ORIGIN_BIN_SHA256 == (
        "04baf49c28f4be89424f3a61dd30d674c4ab165ff16c1b007525829a8c9441f7"
    )
    gates = phases.current_uie_gates()
    assert gates["origin_pin_recorded"] is True
    assert gates["deserialization_authorized"] is False
    assert gates["conversion_enabled"] is False
    assert gates["j12_derived_load_authorized"] is False
    assert gates["j13_training_authorized"] is False
    assert gates["phases_active"] is False
    assert origin.deserialization_authorized() is False
    live = _hypothetical(
        deserialization_requested=origin.deserialization_authorized(),
        isolated_corrected_environment=False,
        torch_version="2.5.1+cu121",
        conversion_finished=False,
        equivalence_ok=False,
        j12_load_authorized=origin.UIE_J12_DERIVED_LOAD_AUTHORIZED,
        batch_passed=False,
        j12_job_id=None,
        j13_authorized=origin.UIE_J13_TRAINING_AUTHORIZED,
    )
    assert live["origin_identified"] is True
    assert live["deserialization_granted"] is False
    assert live["conversion_accepted"] is False
    assert live["j12_technical_pass"] is False
    assert live["j13_granted"] is False


def test_valid_synthetic_derivative_passes_identity_but_not_load(tmp_path: Path) -> None:
    origin_dir = tmp_path / "origin"
    derived = tmp_path / "derived"
    origin_dir.mkdir()
    derived.mkdir()
    payload = b"synthetic-safetensors"
    (derived / "model.safetensors").write_bytes(payload)
    identity = fingerprint.derived_weight_identity(derived, origin_dir)
    assert identity["ok"] is True
    assert identity["errors"] == []
    recorded = identity["files"]["model.safetensors"]
    assert recorded["size"] == len(payload)
    assert isinstance(recorded["sha256"], str) and len(recorded["sha256"]) == 64
    shared = [
        {"name": "a", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s"},
        {"name": "b", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s"},
    ]
    assert conversion.compare_tensor_records(shared, shared)["equivalent"] is True
    loaded = conversion.load_equivalence(
        origin_bin_sha256=origin.ORIGIN_BIN_SHA256,
        derived_file_sha256=recorded["sha256"],
        reported_loaded_sha256=recorded["sha256"],
    )
    assert loaded["equivalent"] is True
    assessment = fingerprint.assess_preflight_inputs(None, derived, None, origin_dir)
    assert assessment["derived_identity_ok"] is True
    assert assessment["ready"] is False
    assert "safetensors conversion is disabled" in assessment["errors"]
    assert "J12 derived load is not authorized" in assessment["errors"]
    assert phases.derived_load_permitted(identity_ok=True, equivalence_ok=True) is False


def test_invalid_or_unauthorized_derivative_is_rejected(tmp_path: Path) -> None:
    origin_dir = tmp_path / "origin"
    origin_dir.mkdir()
    (origin_dir / "pytorch_model.bin").write_bytes(b"pickle")
    (origin_dir / "model.safetensors").write_bytes(b"safe")
    mixed = fingerprint.derived_weight_identity(origin_dir, origin_dir)
    assert mixed["ok"] is False
    assert any("pickle" in item or "origin" in item for item in mixed["errors"])
    empty = tmp_path / "empty"
    empty.mkdir()
    assert fingerprint.derived_weight_identity(empty, origin_dir)["ok"] is False
    copied = [
        {"name": "a", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s1"},
        {"name": "b", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s2"},
    ]
    shared = [
        {"name": "a", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s"},
        {"name": "b", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s"},
    ]
    assert conversion.compare_tensor_records(shared, copied)["equivalent"] is False
    pickle_load = conversion.load_equivalence(
        origin_bin_sha256=origin.ORIGIN_BIN_SHA256,
        derived_file_sha256="d" * 64,
        reported_loaded_sha256=origin.ORIGIN_BIN_SHA256,
    )
    assert pickle_load["equivalent"] is False
    blocked_runtime = _hypothetical(torch_version="2.5.1+cu121")
    assert blocked_runtime["deserialization_granted"] is False
    assert blocked_runtime["j12_technical_pass"] is False
    blocked_check = _hypothetical(transformers_security_check_enabled=False)
    assert blocked_check["deserialization_granted"] is False
    blocked_origin = _hypothetical(origin_matches_pin=False)
    assert blocked_origin["origin_identified"] is False
    assert blocked_origin["deserialization_granted"] is False
    blocked_identity = _hypothetical(derived_identity_ok=False)
    assert blocked_identity["derived_identity_ok"] is False
    assert blocked_identity["j12_technical_pass"] is False
    assert phases.derived_load_permitted(identity_ok=False, equivalence_ok=False) is False


def test_j12_acceptance_does_not_authorize_j13(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    accepted = _hypothetical()
    assert accepted["j12_technical_pass"] is True
    assert accepted["j13_granted"] is False
    assert accepted["j12_pass_authorizes_j13"] is False
    assert origin.UIE_J13_TRAINING_AUTHORIZED is False
    assert origin.deserialization_authorized() is False
    runs = _scratch(tmp_path, monkeypatch)
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "model.safetensors").write_bytes(b"synthetic-safetensors")
    (runs / "uie").mkdir()
    (runs / "uie" / "derived_snapshot.json").write_text(
        json.dumps({"local_path": str(derived)}),
        encoding="utf-8",
    )

    def refuse_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("loaded")

    monkeypatch.setattr(j12, "_run_batch", refuse_load)
    assert j12.main() == 1
    report = json.loads(
        (runs / "uie" / "preflight" / "jobs" / "1349123" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "FAIL"
    assert report["deserialization_authorized"] is False
    assert "safetensors conversion is disabled" in report["error"]

    job = "77"
    preflight = {
        "derived_model_dir": str(derived),
        "derived_weight_hashes": {"model.safetensors": {"sha256": "b" * 64, "size": 1}},
        "deserialization_authorized": False,
        "fingerprint_sha256": "a" * 64,
        "frozen_10shot_hashes": dict(origin.FROZEN_10SHOT_SHA256),
        "slurm_job_id": job,
        "status": "PREFLIGHT_PASS",
    }
    report_dir = runs / "uie" / "preflight" / "jobs" / job
    report_dir.mkdir(parents=True)
    (report_dir / "preflight_report.json").write_text(json.dumps(preflight), encoding="utf-8")
    (report_dir / "environment_fingerprint.json").write_text("{}", encoding="utf-8")
    (runs / "uie" / "preflight" / "accepted_job_id.txt").write_text(job + "\n", encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    with pytest.raises(SystemExit):
        j13.main()
    assert "J13 training is not authorized" in capsys.readouterr().out


def test_old_and_foreign_uie_reports_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    legacy = runs / "uie" / "preflight" / "preflight_report.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"status":"PREFLIGHT_PASS"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        j13.require_identified_preflight()
    assert "unidentified" in capsys.readouterr().out
    job = "77"
    report_dir = runs / "uie" / "preflight" / "jobs" / job
    report_dir.mkdir(parents=True)
    (runs / "uie" / "preflight" / "accepted_job_id.txt").write_text(job + "\n", encoding="utf-8")
    foreign = {"slurm_job_id": "88", "status": "PREFLIGHT_PASS", "deserialization_authorized": False}
    (report_dir / "preflight_report.json").write_text(json.dumps(foreign), encoding="utf-8")
    (report_dir / "environment_fingerprint.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        j13.require_identified_preflight()
    assert "another JOBID" in capsys.readouterr().out


def test_flags_without_records_do_not_accept_a_derived() -> None:
    bare = phases.evaluate_uie_transition(
        origin_matches_pin=True,
        deserialization_requested=True,
        isolated_corrected_environment=True,
        torch_version="2.6.0",
        transformers_security_check_enabled=True,
        conversion_finished=True,
        derived_identity_ok=True,
        equivalence_ok=True,
        load_is_derived_not_origin=True,
        j12_load_authorized=True,
        batch_passed=True,
        j12_job_id="77",
        j13_authorized=False,
        provenance={"equivalent": True, "origin_sha256": origin.ORIGIN_BIN_SHA256, "origin_size": origin.ORIGIN_BIN_SIZE},
        derived_identity={"ok": True, "files": {}},
    )
    assert bare["origin_identified"] is True
    assert bare["equivalence_accepted"] is False
    assert bare["j12_technical_pass"] is False
    assert bare["j13_granted"] is False


def test_j12_requires_recomputed_provenance_before_the_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = _scratch(tmp_path, monkeypatch)
    derived = tmp_path / "derived"
    derived.mkdir()
    payload = b"synthetic-safetensors"
    target = derived / "model.safetensors"
    target.write_bytes(payload)
    measured = conversion.load_equivalence(
        origin_bin_sha256=origin.ORIGIN_BIN_SHA256,
        derived_file_sha256="d" * 64,
        reported_loaded_sha256="d" * 64,
    )
    assert measured["equivalent"] is True
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    provenance = _provenance(digest)
    snapshot = {"local_path": str(derived), **provenance}
    (runs / "uie").mkdir()
    (runs / "uie" / "shot_dir.txt").write_text(str(tmp_path / "shot") + "\n", encoding="utf-8")
    (runs / "uie" / "derived_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(origin, "CONVERSION_ENABLED", True)
    monkeypatch.setattr(origin, "UIE_J12_DERIVED_LOAD_AUTHORIZED", True)
    monkeypatch.setattr(
        fingerprint,
        "frozen_shot_identity",
        lambda _shot: {"errors": [], "files": {}, "ok": True},
    )
    calls: list[str] = []

    def refuse_load(*_args: object, **_kwargs: object) -> None:
        calls.append("batch")

    monkeypatch.setattr(j12, "_run_batch", refuse_load)
    (runs / "uie" / "derived_snapshot.json").write_text(
        json.dumps({"local_path": str(derived), "equivalent": True}),
        encoding="utf-8",
    )
    assert j12.main() == 1
    assert calls == []
    assert origin.UIE_J13_TRAINING_AUTHORIZED is False

    (runs / "uie" / "preflight" / "jobs" / "1349123" / "preflight_report.json").unlink()
    (runs / "uie" / "preflight" / "jobs" / "1349123" / "environment_fingerprint.json").unlink()
    (runs / "uie" / "derived_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    def fake_fingerprint(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("fingerprint")
        return {
            "derived_weight_hashes": {"model.safetensors": {"sha256": digest, "size": len(payload)}},
            "frozen_10shot_hashes": dict(origin.FROZEN_10SHOT_SHA256),
            "script_hashes": {"marker": "synthetic"},
        }

    monkeypatch.setenv("P3_UPSTREAM_ROOT", str(tmp_path / "upstream"))
    monkeypatch.setattr(j12, "build_fingerprint", fake_fingerprint)

    def allow_batch(*_args: object, **_kwargs: object) -> None:
        calls.append("batch")

    monkeypatch.setattr(j12, "_run_batch", allow_batch)
    assert j12.main() == 0
    assert calls == ["fingerprint", "batch"]
    report = json.loads(
        (runs / "uie" / "preflight" / "jobs" / "1349123" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["deserialization_authorized"] is False
    assert report["slurm_job_id"] == "1349123"

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    with pytest.raises(SystemExit):
        j13.main()
    assert "J13 training is not authorized" in capsys.readouterr().out
    assert origin.deserialization_authorized() is False
    assert origin.UIE_J13_TRAINING_AUTHORIZED is False


def test_upstream_mirror_remains_unverified_and_unpatched() -> None:
    legacy = (ROOT.parent / "_legacy" / "UIE" / "run_uie_finetune.py").read_text(encoding="utf-8")
    assert "mirror='tuna'" in legacy
    gates = phases.current_uie_gates()
    assert gates["upstream_mirror_argument"] == "tuna"
    assert gates["upstream_mirror_compatibility"] == "unverified_against_transformers_4.57.6"
    j12_source = (ROOT / "scripts" / "cesvima" / "j12_uie_preflight.py").read_text(encoding="utf-8")
    assert "mirror" not in j12_source
    assert "torch.load" not in j12_source
