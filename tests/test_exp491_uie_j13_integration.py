"""Local gates for the isolated J13 integration probe. No GPU work runs here."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_exp49_uie_j13_readiness import JOBS, ROOT, SCRIPTS, _prepare


sys.path.insert(0, str(SCRIPTS))

import j13_uie_integration_probe as integration  # noqa: E402
import j13_uie_train as j13  # noqa: E402
import uie_j13_gate as gate  # noqa: E402
import uie_train_config as train_config  # noqa: E402


def _accredited_probe(runs: Path, *, batch16: int = gate.ACCEPTED_PROBE_BATCH16_PEAK_VRAM_BYTES) -> Path:
    report = {
        "batch_16_forward_backward_peak_vram_bytes": batch16,
        "checkpoint_bytes": gate.ACCEPTED_PROBE_CHECKPOINT_BYTES,
        "eval_batch_64_generation_peak_vram_bytes": gate.ACCEPTED_PROBE_GENERATION64_PEAK_VRAM_BYTES,
        "slurm_job_id": gate.ACCEPTED_PROBE_JOB_ID,
        "status": "PROBE_PASS",
    }
    destination = runs / "uie" / "j13-probe" / "jobs" / gate.ACCEPTED_PROBE_JOB_ID
    destination.mkdir(parents=True)
    path = destination / "probe_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    pointer = runs / "uie" / "j13-probe" / "accepted_job_id.txt"
    pointer.write_text(gate.ACCEPTED_PROBE_JOB_ID + "\n", encoding="utf-8")
    return path


def test_integration_grant_does_not_authorize_j13_or_the_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare(tmp_path, monkeypatch)
    monkeypatch.delenv(gate.INTEGRATION_ENV, raising=False)
    monkeypatch.delenv(gate.RUN_ENV, raising=False)
    monkeypatch.delenv(gate.LAUNCH_ENV, raising=False)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    with pytest.raises(SystemExit):
        integration.main([])
    assert "integration probe is not authorized" in capsys.readouterr().out

    monkeypatch.setenv(gate.INTEGRATION_ENV, "1")
    with pytest.raises(SystemExit):
        j13.main()
    assert "J13 training is not authorized" in capsys.readouterr().out

    monkeypatch.setenv(gate.RUN_ENV, "1")
    with pytest.raises(SystemExit):
        integration.main([])
    assert "refuses a training authorization" in capsys.readouterr().out
    monkeypatch.delenv(gate.RUN_ENV, raising=False)
    monkeypatch.setenv(gate.LAUNCH_ENV, "1")
    with pytest.raises(SystemExit):
        integration.main(["--cpu-only"])
    assert "refuses a training authorization" in capsys.readouterr().out


def test_child_trainer_and_cpu_contract_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs, _derived, _shot = _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv(gate.INTEGRATION_ENV, "1")
    monkeypatch.delenv(gate.INTEGRATION_CHILD_ENV, raising=False)
    monkeypatch.delenv(gate.RUN_ENV, raising=False)
    monkeypatch.delenv(gate.LAUNCH_ENV, raising=False)
    with pytest.raises(SystemExit):
        integration.main(["--run-limited-trainer", str(tmp_path)])
    assert "trainer child is not authorized" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        integration.main(["--cpu-only"])
    assert "accepted GPU probe pointer" in capsys.readouterr().out
    assert not (runs / "uie" / "j13-integration").exists()


def test_accepted_probe_report_is_read_without_being_rewritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs, _derived, _shot = _prepare(tmp_path, monkeypatch)
    path = _accredited_probe(runs)
    before = path.read_bytes()
    report = integration._read_only_probe(runs)
    assert report["slurm_job_id"] == gate.ACCEPTED_PROBE_JOB_ID
    assert path.read_bytes() == before
    pointer = runs / "uie" / "j13-probe" / "accepted_job_id.txt"
    assert pointer.read_text(encoding="utf-8").strip() == gate.ACCEPTED_PROBE_JOB_ID

    path.write_text(path.read_text(encoding="utf-8").replace(str(gate.ACCEPTED_PROBE_BATCH16_PEAK_VRAM_BYTES), "1"), encoding="utf-8")
    unchanged = path.read_bytes()
    with pytest.raises(SystemExit):
        integration._read_only_probe(runs)
    assert path.read_bytes() == unchanged


def test_j13_stops_for_an_integration_lock_and_the_grant_still_comes_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs, _derived, _shot = _prepare(tmp_path, monkeypatch)
    lock = runs / "uie" / "j13-integration" / "publish.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("1363999\n", encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    monkeypatch.delenv(gate.RUN_ENV, raising=False)
    with pytest.raises(SystemExit):
        j13.main()
    assert "J13 training is not authorized" in capsys.readouterr().out
    assert "integration probe lock" not in capsys.readouterr().out

    monkeypatch.setenv(gate.RUN_ENV, "1")
    with pytest.raises(SystemExit):
        j13.main()
    assert "integration probe lock" in capsys.readouterr().out
    assert lock.read_text(encoding="utf-8") == "1363999\n"


def test_probe_limits_do_not_change_the_frozen_configuration(tmp_path: Path) -> None:
    shot = tmp_path / "seed1" / "10shot"
    shot.mkdir(parents=True)
    frozen = train_config.training_argv(
        uie_root=tmp_path / "UIE",
        train_dir=shot,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "canonical",
        eval_steps=80,
    )
    assert gate.frozen_argv_problems(frozen) == []
    assert "--num_train_epochs=200" in frozen
    assert not any(item.startswith("--max_steps") or item.startswith("--save_steps") for item in frozen)
    limited = integration.limited_training_argv(
        uie_root=tmp_path / "UIE",
        train_dir=shot,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "probe-out",
    )
    assert "--num_train_epochs=200" in limited
    assert "--max_steps=1" in limited
    assert "--max_train_samples=16" in limited
    assert "--save_steps=1" in limited
    assert "--eval_strategy=steps" in limited
    assert "--learning_rate=1e-4" in limited
    assert "--per_device_train_batch_size=16" in limited
    config = (SCRIPTS / "uie_train_config.py").read_text(encoding="utf-8")
    assert "max_steps" not in config
    assert "max_train_samples" not in config


def test_checkpoint_classification_does_not_invent_equivalence() -> None:
    unexplained = integration.classify_checkpoint_keys(["kept", "lost"], ["kept"], [])
    assert unexplained["equivalence"] == "not_claimed"
    assert unexplained["unexplained_missing_keys"] == ["lost"]
    explained = integration.classify_checkpoint_keys(["kept", "tied"], ["kept"], [["kept", "tied"]])
    assert explained["equivalence"] == "not_claimed"
    assert explained["explained_missing_keys"] == ["tied"]
    assert explained["unexplained_missing_keys"] == []


def test_accepted_probe_accounting_and_launch_boundaries() -> None:
    source = (SCRIPTS / "j13_uie_gpu_probe.py").read_text(encoding="utf-8")
    stage = source.split("def _stage", 1)[1].split("def _batch", 1)[0]
    assert "reset_peak_memory_stats()" in stage
    assert "synchronize" not in source
    assert "optimizer.step" not in source
    assert source.index('eval_batch_64_generation') < source.index('report["max_vram_bytes"]')
    note = integration.accepted_probe_vram_note()
    assert "reset" in note["max_vram_bytes"]
    assert "synchronize" in note["timing"]
    assert "optimizer.step" in note["optimizer"]

    integration_source = (SCRIPTS / "j13_uie_integration_probe.py").read_text(encoding="utf-8")
    assert "j13_uie_train" not in integration_source
    assert "j14_" not in integration_source
    assert "AUTHORIZED=1" not in integration_source
    assert "authorizes_j13" in integration_source
    j13_source = (SCRIPTS / "j13_uie_train.py").read_text(encoding="utf-8")
    assert "INTEGRATION_PASS" not in j13_source
    assert "j13-integration" in j13_source
    submitter = (SCRIPTS / "submit_train_first.sh").read_text(encoding="utf-8")
    assert "j13_uie_integration_probe.sbatch" not in submitter
    sbatch = (JOBS / "j13_uie_integration_probe.sbatch").read_text(encoding="utf-8")
    assert "AUTHORIZED=1" not in sbatch
    assert "j13_uie_train" not in sbatch
    assert "j14_" not in sbatch
    assert "source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh" in sbatch
    assert "--account" not in sbatch
    assert "--mail-user" not in sbatch
    runs = ROOT / "not-created"
    with pytest.raises(SystemExit):
        integration.assert_output_not_protected(runs, runs / "uie" / "train" / "seed1-10shot")


def test_sbatch_syntax() -> None:
    path = JOBS / "j13_uie_integration_probe.sbatch"
    completed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
