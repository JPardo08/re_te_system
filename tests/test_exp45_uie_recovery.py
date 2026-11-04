"""Offline UIE checks for the EXP.4.5 recovery contract."""

from __future__ import annotations

import hashlib
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
    assert 'torch.load(path, map_location="cpu", weights_only=True)' in conversion_source
    assert "weights_only=False" not in conversion_source
    assert "add_safe_globals" not in conversion_source
    assert "use_safetensors=True" in j12_source
    submitter = (SCRIPTS / "submit_train_first.sh").read_text(encoding="utf-8")
    sbatch = (ROOT / "jobs" / "cesvima" / "uie_safetensors_convert.sbatch").read_text(encoding="utf-8")
    j12_sbatch = (ROOT / "jobs" / "cesvima" / "j12_uie_preflight.sbatch").read_text(encoding="utf-8")
    assert "uie_safetensors_convert.sbatch" not in submitter
    assert "UIE_SAFETENSORS_CONVERT_AUTHORIZED=1" not in sbatch
    assert "UIE_J12_DERIVED_LOAD_AUTHORIZED=1" not in j12_sbatch
    assert "uie-train" in sbatch
    document = (ROOT / "docs" / "CESVIMA_TRAIN_FIRST.md").read_text(encoding="utf-8")
    assert "descart" not in document.lower()
    assert "rejected" not in document.lower()


def _conversion_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job_id: str) -> tuple[Path, Path, Path]:
    scratch = tmp_path / "scratch"
    runs = scratch / "runs"
    models = scratch / "models"
    for path in (runs, models):
        path.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    monkeypatch.setenv("P3_MODELS_ROOT", str(models))
    monkeypatch.setenv("SLURM_JOB_ID", job_id)
    monkeypatch.delenv(conversion.CONVERT_ENV, raising=False)
    monkeypatch.delenv(conversion.J12_LOAD_ENV, raising=False)
    return scratch, runs, models


def _origin_dir(scratch: Path) -> Path:
    origin_dir = (
        scratch
        / "hf-cache"
        / "hub"
        / "models--luyaojie--uie-base-en"
        / "snapshots"
        / origin.ORIGIN_REVISION
    )
    origin_dir.mkdir(parents=True)
    (origin_dir / "config.json").write_text('{"model_type":"t5"}\n', encoding="utf-8")
    (origin_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (origin_dir / "special_tokens_map.json").write_text("{}\n", encoding="utf-8")
    return origin_dir


def _point_at_origin(runs: Path, origin_dir: Path) -> None:
    pointer = runs / "uie" / "base_snapshot.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps(
            {
                "local_path": str(origin_dir),
                "repository": conversion.UIE_REPOSITORY,
                "revision": origin.ORIGIN_REVISION,
            }
        ),
        encoding="utf-8",
    )


def test_conversion_authorization_and_unsafe_runtimes_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scratch_root, runs, _models = _conversion_runtime(tmp_path, monkeypatch, "1353001")
    with pytest.raises(SystemExit):
        conversion.main()
    assert not (runs / "uie" / "conversion").exists()
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    with pytest.raises(conversion.ConversionDisabled):
        conversion.assert_conversion_runtime("2.5.1+cu121")
    with pytest.raises(conversion.ConversionDisabled):
        conversion.assert_conversion_runtime("2.5.0")
    monkeypatch.setattr(conversion, "_runtime_prefix_name", lambda: "uie-train")
    with pytest.raises(conversion.ConversionDisabled):
        conversion.assert_conversion_runtime("2.6.0")
    assert origin.deserialization_authorized() is False
    assert origin.CONVERSION_ENABLED is False
    assert origin.UIE_J12_DERIVED_LOAD_AUTHORIZED is False
    assert origin.UIE_J13_TRAINING_AUTHORIZED is False
    assert conversion.j12_load_authorized() is False


def test_wrong_origin_hash_never_calls_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    scratch, _runs, _models = _conversion_runtime(tmp_path, monkeypatch, "1353002")
    origin_dir = _origin_dir(scratch)
    bin_path = origin_dir / "pytorch_model.bin"
    bin_path.write_bytes(b"not-the-checkpoint")
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    calls: list[dict[str, object]] = []
    real_load = torch.load

    def spy(path: object, *args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    destination = scratch / "models" / "uie" / "derived"
    with pytest.raises(conversion.ConversionDisabled, match="not the pinned checkpoint"):
        conversion.convert_pinned_origin(origin_dir, destination, scratch, "1353002")
    assert calls == []
    assert not destination.exists()
    assert not conversion.partial_snapshot_dir(destination, "1353002").exists()
    with pytest.raises(conversion.ConversionDisabled, match="inside the origin snapshot"):
        conversion.convert_origin_to_safetensors(
            bin_path,
            origin_dir / "model.safetensors",
            loader=spy,
        )


def test_synthetic_tensors_round_trip_and_reject_partial_views(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import load_file

    weight = torch.tensor([1.25, -2.0, 3.5], dtype=torch.float32)
    bias = torch.tensor([0.5], dtype=torch.float32)
    changed = weight.clone()
    changed[0] = 0
    assert conversion.tensor_values_sha256(weight) == conversion.tensor_values_sha256(weight.clone())
    assert conversion.tensor_values_sha256(weight) != conversion.tensor_values_sha256(changed)
    destination = tmp_path / "derived" / "model.safetensors"
    verified = conversion.write_verified_safetensors(
        {"bias": bias, "lm_head.weight": weight, "shared.weight": weight},
        destination,
    )
    assert verified["physical_alias_preserved"] is False
    assert verified["origin_shared_storage_groups"] == [["lm_head.weight", "shared.weight"]]
    reloaded = load_file(str(destination))
    assert set(reloaded) == {"bias", "lm_head.weight", "shared.weight"}
    assert torch.equal(reloaded["shared.weight"], weight)
    assert torch.equal(reloaded["lm_head.weight"], weight)
    assert torch.equal(reloaded["bias"], bias)
    assert reloaded["shared.weight"].data_ptr() != reloaded["lm_head.weight"].data_ptr()
    origin_ids = {item["name"]: item["storage_id"] for item in verified["origin_tensors"]}
    derived_ids = {item["name"]: item["storage_id"] for item in verified["derived_tensors"]}
    assert origin_ids["shared.weight"] == "shared:lm_head.weight|shared.weight"
    assert origin_ids["lm_head.weight"] == origin_ids["shared.weight"]
    assert derived_ids["shared.weight"] == "unique:shared.weight"
    assert derived_ids["lm_head.weight"] == "unique:lm_head.weight"
    assert not any(destination.parent.glob("*.bin"))
    identity = fingerprint.derived_weight_identity(destination.parent, tmp_path / "origin")
    provenance = {
        "alias_contract": verified["alias_contract"],
        "derived_tensors": verified["derived_tensors"],
        "origin_sha256": origin.ORIGIN_BIN_SHA256,
        "origin_size": origin.ORIGIN_BIN_SIZE,
        "origin_tensors": verified["origin_tensors"],
        "reported_loaded_sha256": identity["files"]["model.safetensors"]["sha256"],
    }
    accepted = conversion.derived_provenance_accepted(identity, provenance)
    assert accepted["accepted"] is True
    assert accepted["equivalence"]["values_equal"] is True
    assert accepted["equivalence"]["shared_storage_groups_equal"] is False
    lied = json.loads(json.dumps(provenance))
    lied["alias_contract"]["physical_alias_preserved"] = True
    assert conversion.derived_provenance_accepted(identity, lied)["accepted"] is False
    base = torch.arange(8, dtype=torch.float32)
    partial = destination.parent / "partial.safetensors"
    with pytest.raises(conversion.ConversionDisabled, match="shared storage cannot be preserved"):
        conversion.write_verified_safetensors({"full": base, "narrow": base.narrow(0, 0, 4)}, partial)
    assert not partial.exists()


def test_failed_conversion_keeps_the_report_and_does_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    scratch, runs, _models = _conversion_runtime(tmp_path, monkeypatch, "1353003")
    origin_dir = _origin_dir(scratch)
    (origin_dir / "tokenizer_config.json").unlink()
    torch.save({"weight": torch.tensor([1.0, 2.0])}, origin_dir / "pytorch_model.bin")
    before = (origin_dir / "pytorch_model.bin").read_bytes()
    _point_at_origin(runs, origin_dir)
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    monkeypatch.setattr(conversion, "origin_identity_matches", lambda _size, _digest: True)
    assert conversion.main() == 1
    pointer = runs / "uie" / "conversion" / "accepted_job_id.txt"
    report_path = runs / "uie" / "conversion" / "jobs" / "1353003" / "conversion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "CONVERSION_FAIL"
    assert report["verified"] is False
    assert not pointer.exists()
    assert (origin_dir / "pytorch_model.bin").read_bytes() == before
    destination = conversion.derived_snapshot_dir(scratch / "models")
    assert not destination.exists()
    assert not conversion.partial_snapshot_dir(destination, "1353003").exists()
    assert not (runs / "uie" / "conversion" / "publish.lock").exists()
    saved = report_path.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        conversion.main()
    assert report_path.read_text(encoding="utf-8") == saved
    assert not pointer.exists()


def test_authorized_synthetic_conversion_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from safetensors.torch import load_file

    scratch, runs, _models = _conversion_runtime(tmp_path, monkeypatch, "1353004")
    origin_dir = _origin_dir(scratch)
    weight = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    torch.save({"lm_head.weight": weight, "shared.weight": weight}, origin_dir / "pytorch_model.bin")
    (origin_dir / "ignored.bin").write_bytes(b"other-weights")
    before = {path.name: path.read_bytes() for path in origin_dir.iterdir() if path.is_file()}
    _point_at_origin(runs, origin_dir)
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    monkeypatch.setattr(conversion, "origin_identity_matches", lambda _size, _digest: True)
    calls: list[dict[str, object]] = []
    real_load = torch.load

    def spy(path: object, *args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    assert conversion.main() == 0
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert (runs / "uie" / "conversion" / "accepted_job_id.txt").read_text(encoding="utf-8") == "1353004\n"
    report_path = runs / "uie" / "conversion" / "jobs" / "1353004" / "conversion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "CONVERSION_PASS"
    assert report["verified"] is True
    assert report["weights_only"] is True
    assert report["use_safetensors"] is True
    assert report["revision"] == origin.ORIGIN_REVISION
    assert report["origin_unchanged"] is True
    assert report["physical_alias_preserved"] is False
    derived = Path(report["derived_path"])
    assert sorted(path.name for path in derived.iterdir()) == [
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]
    reloaded = load_file(str(derived / "model.safetensors"))
    assert torch.equal(reloaded["shared.weight"], weight)
    assert reloaded["shared.weight"].data_ptr() != reloaded["lm_head.weight"].data_ptr()
    for name, payload in before.items():
        assert (origin_dir / name).read_bytes() == payload
    saved = report_path.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        conversion.main()
    assert report_path.read_text(encoding="utf-8") == saved


def test_j12_load_grant_is_independent_and_j13_stays_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import torch

    runs = _scratch(tmp_path, monkeypatch)
    weight = torch.tensor([1.0, 2.0], dtype=torch.float32)
    derived = runs.parent / "derived"
    verified = conversion.write_verified_safetensors(
        {"lm_head.weight": weight, "shared.weight": weight},
        derived / "model.safetensors",
    )
    config = b'{"model_type":"t5"}\n'
    tokenizer = b"{}\n"
    (derived / "config.json").write_bytes(config)
    (derived / "tokenizer_config.json").write_bytes(tokenizer)
    identity = fingerprint.derived_weight_identity(derived, runs.parent / "origin")
    job_id = "1353005"
    report = {
        "alias_contract": verified["alias_contract"],
        "conversion_job_id": job_id,
        "copied_files": {
            "config.json": {"bytes": len(config), "sha256": hashlib.sha256(config).hexdigest()},
            "tokenizer_config.json": {
                "bytes": len(tokenizer),
                "sha256": hashlib.sha256(tokenizer).hexdigest(),
            },
        },
        "derived_path": str(derived),
        "derived_tensors": verified["derived_tensors"],
        "origin_sha256": origin.ORIGIN_BIN_SHA256,
        "origin_size": origin.ORIGIN_BIN_SIZE,
        "origin_tensors": verified["origin_tensors"],
        "origin_unchanged": True,
        "reported_loaded_sha256": identity["files"]["model.safetensors"]["sha256"],
        "revision": origin.ORIGIN_REVISION,
        "slurm_job_id": job_id,
        "status": "CONVERSION_PASS",
        "use_safetensors": True,
        "verified": True,
        "weights_only": True,
    }
    assert conversion.derived_provenance_accepted(identity, report)["accepted"] is True
    report_path = runs / "uie" / "conversion" / "jobs" / job_id / "conversion_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (runs / "uie" / "conversion" / "accepted_job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    (runs / "uie").mkdir(exist_ok=True)
    (runs / "uie" / "shot_dir.txt").write_text(str(tmp_path / "shot") + "\n", encoding="utf-8")
    monkeypatch.setattr(
        fingerprint,
        "frozen_shot_identity",
        lambda _shot: {"errors": [], "files": {}, "ok": True},
    )
    calls: list[str] = []
    monkeypatch.setattr(j12, "_run_batch", lambda *_args, **_kwargs: calls.append("batch"))
    monkeypatch.setattr(
        j12,
        "build_fingerprint",
        lambda *_args, **_kwargs: {
            "derived_weight_hashes": identity["files"],
            "frozen_10shot_hashes": dict(origin.FROZEN_10SHOT_SHA256),
            "script_hashes": {"marker": "synthetic"},
        },
    )
    monkeypatch.setenv("P3_UPSTREAM_ROOT", str(tmp_path / "upstream"))
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    assert j12.main() == 1
    assert calls == []
    assert origin.UIE_J12_DERIVED_LOAD_AUTHORIZED is False
    (runs / "uie" / "preflight" / "jobs" / "1349123" / "preflight_report.json").unlink()
    (runs / "uie" / "preflight" / "jobs" / "1349123" / "environment_fingerprint.json").unlink()
    monkeypatch.setenv(conversion.J12_LOAD_ENV, "1")
    assert j12.main() == 0
    assert calls == ["batch"]
    published = json.loads(
        (runs / "uie" / "preflight" / "jobs" / "1349123" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert published["status"] == "PREFLIGHT_PASS"
    assert published["deserialization_authorized"] is False
    assert published["conversion_job_id"] == job_id
    foreign = json.loads(report_path.read_text(encoding="utf-8"))
    foreign["slurm_job_id"] = "999"
    report_path.write_text(json.dumps(foreign), encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "1353006")
    assert j12.main() == 1
    assert calls == ["batch"]
    later = json.loads(
        (runs / "uie" / "preflight" / "jobs" / "1353006" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert later["status"] == "FAIL"
    assert "another JOBID" in later["error"]
    preserved = json.loads(
        (runs / "uie" / "preflight" / "jobs" / "1349123" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert preserved == published

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    with pytest.raises(SystemExit):
        j13.main()
    assert "J13 training is not authorized" in capsys.readouterr().out


def test_half_precision_cpu_tensors_round_trip(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import load_file

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        value = torch.tensor([1.0, -2.0, 3.5], dtype=dtype)
        assert value.dtype == dtype
        assert value.device.type == "cpu"
        digest = conversion.tensor_values_sha256(value)
        destination = tmp_path / f"{str(dtype).replace('.', '_')}.safetensors"
        verified = conversion.write_verified_safetensors({"weight": value}, destination)
        reloaded = load_file(str(destination))["weight"]
        assert reloaded.dtype == dtype
        assert reloaded.device.type == "cpu"
        assert conversion.tensor_values_sha256(reloaded) == digest
        assert verified["derived_tensors"][0]["values_sha256"] == digest
        assert verified["derived_tensors"][0]["dtype"] == str(dtype)


def test_load_origin_bin_repeats_authorization_and_refuses_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    path = tmp_path / "pytorch_model.bin"
    path.write_bytes(b"unused")
    with pytest.raises(conversion.ConversionDisabled, match="conversion is disabled"):
        conversion._load_origin_bin(path)
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    monkeypatch.setattr(conversion, "_running_torch_version", lambda: "2.5.1+cu121")
    with pytest.raises(conversion.ConversionDisabled, match="2.5.1"):
        conversion._load_origin_bin(path)
    monkeypatch.setattr(conversion, "_running_torch_version", lambda: torch.__version__)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "uie-train"))
    with pytest.raises(conversion.ConversionDisabled, match="uie-train"):
        conversion._load_origin_bin(path)

    class Marker:
        class device:
            type = "cuda"

    with pytest.raises(conversion.ConversionDisabled, match="not on CPU"):
        conversion._require_cpu_tensor("weight", Marker())


def test_j11_inventory_is_the_observed_snapshot_and_only_two_files_are_required() -> None:
    assert conversion.J11_SNAPSHOT_FILES == (
        "pytorch_model.bin",
        "config.json",
        "tokenizer_config.json",
        "spiece.model",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "README.md",
        ".gitattributes",
    )
    assert conversion.REQUIRED_COMPANIONS == ("config.json", "tokenizer_config.json")
    assert "pytorch_model.bin" not in conversion.REQUIRED_COMPANIONS


def test_huggingface_blob_symlink_is_copied_as_a_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    scratch, runs, models = _conversion_runtime(tmp_path, monkeypatch, "1353010")
    repo = scratch / "hf-cache" / "hub" / "models--luyaojie--uie-base-en"
    snapshot = repo / "snapshots" / origin.ORIGIN_REVISION
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    payloads = {
        "config.json": b'{"model_type":"t5"}\n',
        "tokenizer_config.json": b"{}\n",
        "spiece.model": b"spiece",
        "tokenizer.json": b"{}\n",
        "special_tokens_map.json": b"{}\n",
        "added_tokens.json": b"[]\n",
        "README.md": b"readme\n",
        ".gitattributes": b"*\n",
    }
    for name, payload in payloads.items():
        blob = blobs / name.replace(".", "_")
        blob.write_bytes(payload)
        (snapshot / name).symlink_to(Path("../../blobs") / blob.name)
    torch.save({"weight": torch.tensor([1.0], dtype=torch.float32)}, blobs / "weights")
    (snapshot / "pytorch_model.bin").symlink_to("../../blobs/weights")
    _point_at_origin(runs, snapshot)
    monkeypatch.setenv("HF_HUB_CACHE", str(repo.parent))
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    monkeypatch.setattr(conversion, "origin_identity_matches", lambda _size, _digest: True)
    assert conversion.main() == 0
    derived = conversion.derived_snapshot_dir(models)
    names = {path.name for path in derived.iterdir()}
    assert names == (set(conversion.J11_SNAPSHOT_FILES) - {"pytorch_model.bin"}) | {"model.safetensors"}
    for name, payload in payloads.items():
        copied = derived / name
        assert copied.is_file()
        assert not copied.is_symlink()
        assert copied.read_bytes() == payload
    assert not (derived / "pytorch_model.bin").exists()
    report = json.loads(
        (runs / "uie" / "conversion" / "jobs" / "1353010" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["copied_files"]["spiece.model"]["sha256"] == hashlib.sha256(b"spiece").hexdigest()
    if report["space"]["quota_observed"]:
        assert isinstance(report["space"]["quota_remaining_bytes"], int)
    else:
        assert report["space"]["quota_remaining_bytes"] is None
    assert isinstance(report["space"]["filesystem_free_bytes"], int)


def test_symlink_outside_the_cache_repo_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    scratch, runs, _models = _conversion_runtime(tmp_path, monkeypatch, "1353011")
    origin_dir = _origin_dir(scratch)
    outside = tmp_path / "escaped-config.json"
    outside.write_bytes(b'{"model_type":"t5"}\n')
    (origin_dir / "config.json").unlink()
    (origin_dir / "config.json").symlink_to(outside)
    torch.save({"weight": torch.tensor([1.0])}, origin_dir / "pytorch_model.bin")
    _point_at_origin(runs, origin_dir)
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    monkeypatch.setattr(conversion, "origin_identity_matches", lambda _size, _digest: True)
    assert conversion.main() == 1
    report = json.loads(
        (runs / "uie" / "conversion" / "jobs" / "1353011" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "CONVERSION_FAIL"
    assert "escapes" in report["error"]
    assert not (runs / "uie" / "conversion" / "accepted_job_id.txt").exists()
    inside = scratch / "other-repo" / "config.json"
    inside.parent.mkdir()
    inside.write_bytes(b"{}\n")
    (origin_dir / "config.json").unlink()
    (origin_dir / "config.json").symlink_to(inside)
    monkeypatch.setenv("SLURM_JOB_ID", "1353012")
    assert conversion.main() == 1
    second = json.loads(
        (runs / "uie" / "conversion" / "jobs" / "1353012" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert "authorized cache" in second["error"]
    assert not (runs / "uie" / "conversion" / "accepted_job_id.txt").exists()
    monkeypatch.setenv("SLURM_JOB_ID", "1353020")
    repo = origin_dir.parent.parent
    monkeypatch.setattr(conversion, "derived_snapshot_dir", lambda _root: repo / "not-a-snapshot")
    assert conversion.main() == 1
    refused = json.loads(
        (runs / "uie" / "conversion" / "jobs" / "1353020" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert "origin snapshot" in refused["error"]
    assert not (repo / "not-a-snapshot").exists()


def test_existing_destination_and_held_lock_are_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    scratch, runs, models = _conversion_runtime(tmp_path, monkeypatch, "1353013")
    origin_dir = _origin_dir(scratch)
    torch.save({"weight": torch.tensor([1.0])}, origin_dir / "pytorch_model.bin")
    _point_at_origin(runs, origin_dir)
    destination = conversion.derived_snapshot_dir(models)
    destination.mkdir(parents=True)
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    monkeypatch.setattr(conversion, "origin_identity_matches", lambda _size, _digest: True)
    assert conversion.main() == 1
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not (runs / "uie" / "conversion" / "accepted_job_id.txt").exists()
    failed = json.loads(
        (runs / "uie" / "conversion" / "jobs" / "1353013" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert failed["status"] == "CONVERSION_FAIL"

    monkeypatch.setenv("SLURM_JOB_ID", "1353014")
    lock = runs / "uie" / "conversion" / "publish.lock"
    lock.write_text("other-job\n", encoding="utf-8")
    assert conversion.main() == 1
    assert lock.read_text(encoding="utf-8") == "other-job\n"
    held = json.loads(
        (runs / "uie" / "conversion" / "jobs" / "1353014" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert "lock" in held["error"]
    assert not (runs / "uie" / "conversion" / "accepted_job_id.txt").exists()


def test_short_filesystem_space_and_short_quota_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    real_disk_usage = conversion.shutil.disk_usage
    scratch, runs, _models = _conversion_runtime(tmp_path, monkeypatch, "1353015")
    origin_dir = _origin_dir(scratch)
    torch.save({"weight": torch.tensor([1.0])}, origin_dir / "pytorch_model.bin")
    _point_at_origin(runs, origin_dir)
    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    monkeypatch.setattr(conversion, "origin_identity_matches", lambda _size, _digest: True)
    monkeypatch.setattr(conversion.shutil, "disk_usage", lambda _path: type("Usage", (), {"free": 1})())
    assert conversion.main() == 1
    report = json.loads(
        (runs / "uie" / "conversion" / "jobs" / "1353015" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert "filesystem free space" in report["error"]
    assert not (runs / "uie" / "conversion" / "accepted_job_id.txt").exists()

    monkeypatch.setenv("SLURM_JOB_ID", "1353016")
    monkeypatch.setattr(conversion.shutil, "disk_usage", real_disk_usage)
    monkeypatch.setattr(conversion, "_user_quota_remaining", lambda _path: 1)
    assert conversion.main() == 1
    quota = json.loads(
        (runs / "uie" / "conversion" / "jobs" / "1353016" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert "user quota" in quota["error"]
    assert not (runs / "uie" / "conversion" / "accepted_job_id.txt").exists()


def test_origin_mutation_after_load_is_not_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    scratch, runs, models = _conversion_runtime(tmp_path, monkeypatch, "1353017")
    origin_dir = _origin_dir(scratch)
    torch.save({"weight": torch.tensor([1.0, 2.0])}, origin_dir / "pytorch_model.bin")
    _point_at_origin(runs, origin_dir)
    real_load = conversion._load_origin_bin

    def tamper(path: Path) -> dict:
        payload = real_load(path)
        path.write_bytes(path.read_bytes() + b"\x00")
        return payload

    monkeypatch.setenv(conversion.CONVERT_ENV, "1")
    monkeypatch.setattr(conversion, "origin_identity_matches", lambda _size, _digest: True)
    monkeypatch.setattr(conversion, "_load_origin_bin", tamper)
    assert conversion.main() == 1
    report = json.loads(
        (runs / "uie" / "conversion" / "jobs" / "1353017" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "CONVERSION_FAIL"
    assert "pytorch_model.bin changed" in report["error"]
    assert not (runs / "uie" / "conversion" / "accepted_job_id.txt").exists()
    assert not conversion.derived_snapshot_dir(models).exists()


def test_quota_parser_ignores_ambiguous_output() -> None:
    assert conversion._parse_quota_remaining("Filesystem blocks quota limit\n") is None
    assert conversion._parse_quota_remaining("/beegfs 10 100 50\n/other 1 2 3\n") is None
    assert conversion._parse_quota_remaining("/beegfs 10 100 50\n") == 40 * conversion.QUOTA_BLOCK_BYTES
    assert conversion._parse_quota_remaining("/beegfs 10 0 0\n") is None


def _bound_derivative(directory: Path) -> dict[str, object]:
    import torch

    weight = torch.tensor([1.0, 2.0], dtype=torch.float32)
    verified = conversion.write_verified_safetensors(
        {"lm_head.weight": weight, "shared.weight": weight},
        directory / "model.safetensors",
    )
    payloads = {
        "config.json": b'{"model_type":"t5"}\n',
        "tokenizer_config.json": b"{}\n",
        "tokenizer.json": b'{"version":"1"}\n',
    }
    copied = {}
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
        copied[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    digest = hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()
    return {
        "alias_contract": verified["alias_contract"],
        "copied_files": copied,
        "derived_tensors": verified["derived_tensors"],
        "origin_sha256": origin.ORIGIN_BIN_SHA256,
        "origin_size": origin.ORIGIN_BIN_SIZE,
        "origin_tensors": verified["origin_tensors"],
        "origin_unchanged": True,
        "reported_loaded_sha256": digest,
        "revision": origin.ORIGIN_REVISION,
    }


def test_j12_recomputes_tensors_and_companions_from_the_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    runs = _scratch(tmp_path, monkeypatch)
    derived = runs.parent / "derived"
    report = _bound_derivative(derived)
    job_id = "1353018"
    report.update(
        {
            "conversion_job_id": job_id,
            "derived_path": str(derived),
            "slurm_job_id": job_id,
            "status": "CONVERSION_PASS",
            "use_safetensors": True,
            "verified": True,
            "weights_only": True,
        }
    )
    report_path = runs / "uie" / "conversion" / "jobs" / job_id / "conversion_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (runs / "uie" / "conversion" / "accepted_job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    (runs / "uie" / "shot_dir.txt").write_text(str(tmp_path / "shot") + "\n", encoding="utf-8")
    monkeypatch.setattr(
        fingerprint,
        "frozen_shot_identity",
        lambda _shot: {"errors": [], "files": {}, "ok": True},
    )
    calls: list[str] = []
    monkeypatch.setattr(j12, "_run_batch", lambda *_args, **_kwargs: calls.append("batch"))
    monkeypatch.setattr(
        j12,
        "build_fingerprint",
        lambda *_args, **_kwargs: {
            "derived_weight_hashes": {"model.safetensors": {"sha256": "c" * 64, "size": 1}},
            "frozen_10shot_hashes": dict(origin.FROZEN_10SHOT_SHA256),
            "script_hashes": {"marker": "synthetic"},
        },
    )
    monkeypatch.setenv("P3_UPSTREAM_ROOT", str(tmp_path / "upstream"))
    monkeypatch.setenv(conversion.J12_LOAD_ENV, "1")
    assert j12.main() == 0
    assert calls == ["batch"]

    def fail_closed(reason: str) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "1353019")
        assert j12.main() == 1
        published = json.loads(
            (runs / "uie" / "preflight" / "jobs" / "1353019" / "preflight_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert published["status"] == "FAIL"
        assert reason in published["error"]
        assert calls == ["batch"]
        (runs / "uie" / "preflight" / "jobs" / "1353019" / "preflight_report.json").unlink()
        (runs / "uie" / "preflight" / "jobs" / "1353019" / "environment_fingerprint.json").unlink()

    original = (derived / "config.json").read_bytes()
    (derived / "config.json").write_bytes(original + b" ")
    fail_closed("companion hash does not match")
    (derived / "config.json").write_bytes(original)
    (derived / "tokenizer.json").write_bytes(b'{"version":"tampered"}\n')
    fail_closed("companion hash does not match")
    (derived / "tokenizer.json").write_bytes(b'{"version":"1"}\n')

    original_weights = (derived / "model.safetensors").read_bytes()
    tensors = load_file(str(derived / "model.safetensors"))
    tensors["lm_head.weight"] = tensors["lm_head.weight"] + torch.tensor([1.0, 0.0])
    save_file(tensors, str(derived / "model.safetensors"))
    report["reported_loaded_sha256"] = hashlib.sha256(
        (derived / "model.safetensors").read_bytes()
    ).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    fail_closed("reloaded tensor values do not match")

    (derived / "model.safetensors").write_bytes(original_weights)
    report["reported_loaded_sha256"] = hashlib.sha256(original_weights).hexdigest()
    lied = json.loads(json.dumps(report))
    shared = ["lm_head.weight", "shared.weight"]
    for row in lied["origin_tensors"] + lied["derived_tensors"]:
        row["storage_id"] = "shared:" + "|".join(shared)
    lied["alias_contract"]["origin_shared_groups"] = [shared]
    lied["alias_contract"]["derived_physical_groups"] = [shared]
    lied["alias_contract"]["physical_alias_preserved"] = True
    report_path.write_text(json.dumps(lied), encoding="utf-8")
    fail_closed("reloaded alias groups do not match")
