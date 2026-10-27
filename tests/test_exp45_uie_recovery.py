"""Offline UIE checks for the EXP.4.5 recovery contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
sys.path.insert(0, str(SCRIPTS))

import j12_uie_preflight as j12  # noqa: E402
import j13_uie_train as j13  # noqa: E402
import uie_fingerprint as fingerprint  # noqa: E402
import uie_origin_pin as origin  # noqa: E402
import uie_safetensors_conversion as conversion  # noqa: E402


def _scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    scratch = tmp_path / "scratch"
    runs = scratch / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    monkeypatch.setenv("SLURM_JOB_ID", "1349123")
    return runs

def test_origin_pin_match_does_not_authorize_deserialization() -> None:
    assert origin.origin_identity_matches(origin.ORIGIN_BIN_SIZE, origin.ORIGIN_BIN_SHA256)
    assert origin.deserialization_authorized() is False
    assert origin.DESERIALIZATION_AUTHORIZED is False
    assert origin.CONVERSION_ENABLED is False
    assert origin.UIE_J12_DERIVED_LOAD_AUTHORIZED is False
    assert origin.UIE_J13_TRAINING_AUTHORIZED is False
    summary = conversion.origin_pin_summary()
    assert summary["matches_known_pin"] is True
    assert summary["deserialization_authorized"] is False


def test_small_file_is_not_the_origin_pin(tmp_path: Path) -> None:
    path = tmp_path / "pytorch_model.bin"
    path.write_bytes(b"not-the-checkpoint")
    report = origin.verify_origin_bin(path)
    assert report["observed"] is True
    assert report["matches_pin"] is False
    assert report["deserialization_authorized"] is False


def test_frozen_shot_hashes_match_the_scientific_manifest() -> None:
    manifest = json.loads(
        (ROOT / "artifacts" / "uie_conll04_10shot_seed1_manifest.json").read_text(encoding="utf-8")
    )
    split = manifest["split"]
    assert origin.FROZEN_10SHOT_SHA256["train.json"] == split["split_hash"]
    assert origin.FROZEN_10SHOT_SHA256["val.json"] == split["validation_sha256"]
    assert origin.FROZEN_10SHOT_SHA256["test.json"] == split["test_sha256"]
    assert origin.FROZEN_10SHOT_SHA256["record.schema"] == split["record_schema_sha256"]


def test_conversion_stays_disabled_and_does_not_call_a_loader(tmp_path: Path) -> None:
    assert conversion.torch_version_can_deserialize_origin("2.5.1+cu121") is False
    assert conversion.torch_version_can_deserialize_origin("2.6.0") is True
    with pytest.raises(conversion.ConversionDisabled):
        conversion.assert_conversion_runtime("2.6.0")
    origin_bin = tmp_path / "snapshot" / "pytorch_model.bin"
    origin_bin.parent.mkdir()
    origin_bin.write_bytes(b"preserved")
    called: list[str] = []

    def loader(*_args: object, **_kwargs: object) -> None:
        called.append("called")

    with pytest.raises(conversion.ConversionDisabled):
        conversion.convert_origin_to_safetensors(
            origin_bin,
            tmp_path / "derived" / "model.safetensors",
            loader=loader,
        )
    with pytest.raises(conversion.ConversionDisabled):
        conversion.convert_origin_to_safetensors(
            origin_bin,
            origin_bin.parent / "model.safetensors",
            loader=loader,
        )
    assert called == []


def test_tensor_comparison_checks_shared_storage_and_load_identity() -> None:
    shared = [
        {"name": "a", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s"},
        {"name": "b", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s"},
    ]
    copied = [
        {"name": "a", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s1"},
        {"name": "b", "shape": (2,), "dtype": "float32", "values_sha256": "aa", "storage_id": "s2"},
    ]
    assert conversion.compare_tensor_records(shared, shared)["equivalent"] is True
    diverged = conversion.compare_tensor_records(shared, copied)
    assert diverged["values_equal"] is True
    assert diverged["shared_storage_groups_equal"] is False
    assert diverged["equivalent"] is False
    loaded = conversion.load_equivalence(
        origin_bin_sha256=origin.ORIGIN_BIN_SHA256,
        derived_file_sha256="d" * 64,
        reported_loaded_sha256="d" * 64,
    )
    assert loaded["equivalent"] is True
    pickle_load = conversion.load_equivalence(
        origin_bin_sha256=origin.ORIGIN_BIN_SHA256,
        derived_file_sha256="d" * 64,
        reported_loaded_sha256=origin.ORIGIN_BIN_SHA256,
    )
    assert pickle_load["equivalent"] is False


def test_j12_blocks_before_load_and_preserves_a_legacy_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _scratch(tmp_path, monkeypatch)
    legacy = runs / "uie" / "preflight" / "preflight_report.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"status":"FAIL","evidence":"1349123"}\n', encoding="utf-8")
    def refuse_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("loaded")

    monkeypatch.setattr(j12, "_run_batch", refuse_load)
    assert j12.main() == 1
    assert legacy.read_text(encoding="utf-8") == '{"status":"FAIL","evidence":"1349123"}\n'
    report = json.loads(
        (runs / "uie" / "preflight" / "jobs" / "1349123" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "FAIL"
    assert report["slurm_job_id"] == "1349123"
    assert report["deserialization_authorized"] is False
    assert report["origin_bin_sha256_pin"] == origin.ORIGIN_BIN_SHA256
    assert "safetensors conversion is disabled" in report["error"]
    with pytest.raises(SystemExit):
        j12.main()


def test_derived_directory_with_a_pickle_is_not_ready(tmp_path: Path) -> None:
    shot = tmp_path / "10shot"
    shot.mkdir()
    model = tmp_path / "derived"
    model.mkdir()
    (model / "pytorch_model.bin").write_bytes(b"pickle")
    (model / "model.safetensors").write_bytes(b"safe")
    assessment = fingerprint.assess_preflight_inputs(shot, model, None)
    assert assessment["ready"] is False
    assert assessment["derived"]["ok"] is False
    assert any("pickle" in item for item in assessment["derived"]["errors"])


def test_j13_rejects_failed_incomplete_foreign_and_authorized_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    legacy = runs / "uie" / "preflight" / "preflight_report.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"status":"PASS"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        j13.require_identified_preflight()
    assert "unidentified" in capsys.readouterr().out

    job = "77"
    report_dir = runs / "uie" / "preflight" / "jobs" / job
    report_dir.mkdir(parents=True)
    pointer = runs / "uie" / "preflight" / "accepted_job_id.txt"
    pointer.write_text(job + "\n", encoding="utf-8")
    failed = {
        "slurm_job_id": job,
        "status": "FAIL",
        "deserialization_authorized": False,
    }
    (report_dir / "preflight_report.json").write_text(json.dumps(failed), encoding="utf-8")
    (report_dir / "environment_fingerprint.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        j13.require_identified_preflight()
    assert "status is FAIL" in capsys.readouterr().out

    incomplete = dict(failed)
    incomplete["status"] = "PREFLIGHT_PASS"
    (report_dir / "preflight_report.json").write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(SystemExit):
        j13.require_identified_preflight()
    assert "incomplete" in capsys.readouterr().out

    foreign = _complete_uie_report("88", tmp_path)
    (report_dir / "preflight_report.json").write_text(json.dumps(foreign), encoding="utf-8")
    with pytest.raises(SystemExit):
        j13.require_identified_preflight()
    assert "another JOBID" in capsys.readouterr().out

    complete = _complete_uie_report(job, tmp_path)
    (report_dir / "preflight_report.json").write_text(json.dumps(complete), encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    with pytest.raises(SystemExit):
        j13.main()
    assert "J13 training is not authorized" in capsys.readouterr().out


def _complete_uie_report(job_id: str, tmp_path: Path) -> dict[str, object]:
    return {
        "derived_model_dir": str(tmp_path / "derived"),
        "derived_weight_hashes": {"model.safetensors": {"sha256": "b" * 64, "size": 1}},
        "deserialization_authorized": False,
        "fingerprint_sha256": "a" * 64,
        "frozen_10shot_hashes": dict(origin.FROZEN_10SHOT_SHA256),
        "slurm_job_id": job_id,
        "status": "PREFLIGHT_PASS",
    }

def test_load_paths_do_not_bypass_transformers_or_patch_upstream() -> None:
    j12_source = (SCRIPTS / "j12_uie_preflight.py").read_text(encoding="utf-8")
    j13_source = (SCRIPTS / "j13_uie_train.py").read_text(encoding="utf-8")
    fingerprint_source = (SCRIPTS / "uie_fingerprint.py").read_text(encoding="utf-8")
    for source in (j12_source, j13_source, fingerprint_source):
        assert "torch.load" not in source
        assert "weights_only=False" not in source
        assert "mirror" not in source
    conversion_source = (SCRIPTS / "uie_safetensors_conversion.py").read_text(encoding="utf-8")
    assert "weights_only = False" not in conversion_source
    assert conversion.CONVERSION_REQUIREMENTS["weights_only"] is True
    assert "weights_only=True" in conversion.INTENDED_LOAD
    assert "2.5.1" in conversion_source
    legacy = ROOT.parent / "_legacy" / "UIE" / "run_uie_finetune.py"
    legacy_source = legacy.read_text(encoding="utf-8")
    assert "mirror='tuna'" in legacy_source
    assert "J13 training is not authorized" in j13_source
    assert "differs from J12" in j13_source
