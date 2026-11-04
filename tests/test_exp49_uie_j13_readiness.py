"""Local readiness gates for one UIE J13 run. No GPU training is started."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
JOBS = ROOT / "jobs" / "cesvima"
sys.path.insert(0, str(SCRIPTS))

import j13_uie_gpu_probe as probe  # noqa: E402
import j13_uie_train as j13  # noqa: E402
import uie_j13_gate as gate  # noqa: E402
import uie_origin_pin as origin  # noqa: E402
import uie_trainer_compat as compat  # noqa: E402
import uie_trainer_launch as launcher  # noqa: E402
import uie_train_config as train_config  # noqa: E402


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, job: str = gate.ACCEPTED_J12_JOB_ID):
    scratch = tmp_path / "scratch"
    runs = scratch / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    monkeypatch.setenv("P3_UPSTREAM_ROOT", str(ROOT.parent / "_legacy"))
    monkeypatch.setenv("SLURM_JOB_ID", "1364000")
    monkeypatch.delenv(gate.RUN_ENV, raising=False)
    monkeypatch.delenv(gate.PROBE_ENV, raising=False)
    monkeypatch.delenv(gate.LAUNCH_ENV, raising=False)
    shot = tmp_path / "seed1" / "10shot"
    shot.mkdir(parents=True)
    (shot / "train.json").write_text('{"text":"a"}\n{"text":"b"}\n', encoding="utf-8")
    (runs / "uie").mkdir()
    (runs / "uie" / "shot_dir.txt").write_text(str(shot) + "\n", encoding="utf-8")
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "model.safetensors").write_bytes(b"synthetic-safetensors")
    report = {
        "conversion_job_id": gate.ACCEPTED_CONVERSION_JOB_ID,
        "derived_model_dir": str(derived),
        "derived_weight_hashes": {
            "model.safetensors": {
                "sha256": gate.ACCEPTED_SAFETENSORS_SHA256,
                "size": gate.ACCEPTED_SAFETENSORS_BYTES,
            }
        },
        "deserialization_authorized": False,
        "fingerprint_sha256": gate.ACCEPTED_FINGERPRINT_SHA256,
        "frozen_10shot_hashes": dict(origin.FROZEN_10SHOT_SHA256),
        "slurm_job_id": job,
        "status": "PREFLIGHT_PASS",
    }
    report_dir = runs / "uie" / "preflight" / "jobs" / job
    report_dir.mkdir(parents=True)
    (report_dir / "preflight_report.json").write_text(json.dumps(report), encoding="utf-8")
    (report_dir / "environment_fingerprint.json").write_text("{}\n", encoding="utf-8")
    (runs / "uie" / "preflight" / "accepted_job_id.txt").write_text(job + "\n", encoding="utf-8")
    return runs, derived, shot


def _write_probe(runs: Path, *, status: str = "PROBE_PASS", job: str = "1364001") -> None:
    report = {
        "batch_16_forward_backward": "PASS" if status == "PROBE_PASS" else "OOM",
        "bf16": False,
        "changed_hyperparameters": False,
        "derived_safetensors_sha256": gate.ACCEPTED_SAFETENSORS_SHA256,
        "eval_batch_64_generation": "PASS" if status == "PROBE_PASS" else "NOT_RUN",
        "eval_batch_size": 64,
        "fingerprint_sha256": gate.ACCEPTED_FINGERPRINT_SHA256,
        "fp16": False,
        "gpu_name": "Tesla V100-PCIE-16GB",
        "max_vram_bytes": 16000000000,
        "slurm_job_id": job,
        "status": status,
        "train_batch_size": 16,
    }
    destination = runs / "uie" / "j13-probe" / "jobs" / job
    destination.mkdir(parents=True)
    (destination / "probe_report.json").write_text(json.dumps(report), encoding="utf-8")
    if status == "PROBE_PASS":
        pointer = runs / "uie" / "j13-probe" / "accepted_job_id.txt"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(job + "\n", encoding="utf-8")


def test_constant_and_missing_grant_do_not_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert origin.UIE_J13_TRAINING_AUTHORIZED is False
    _prepare(tmp_path, monkeypatch)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    with pytest.raises(SystemExit):
        j13.main()
    assert "J13 training is not authorized" in capsys.readouterr().out

    monkeypatch.setenv(gate.RUN_ENV, "1")
    monkeypatch.setattr(j13, "UIE_J13_TRAINING_AUTHORIZED", True)
    with pytest.raises(SystemExit):
        j13.main()
    assert "J13 training is not authorized" in capsys.readouterr().out


def test_rejects_unaccepted_j12_and_modified_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs, _derived, _shot = _prepare(tmp_path, monkeypatch, job="77")
    monkeypatch.setenv(gate.RUN_ENV, "1")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    with pytest.raises(SystemExit):
        j13.main()
    assert "not the accepted J12 run" in capsys.readouterr().out

    report_path = runs / "uie" / "preflight" / "jobs" / "77" / "preflight_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["slurm_job_id"] = gate.ACCEPTED_J12_JOB_ID
    report["fingerprint_sha256"] = "a" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    pointer = runs / "uie" / "preflight" / "accepted_job_id.txt"
    pointer.write_text(gate.ACCEPTED_J12_JOB_ID + "\n", encoding="utf-8")
    accepted_dir = runs / "uie" / "preflight" / "jobs" / gate.ACCEPTED_J12_JOB_ID
    accepted_dir.mkdir()
    (accepted_dir / "preflight_report.json").write_text(json.dumps(report), encoding="utf-8")
    (accepted_dir / "environment_fingerprint.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        j13.main()
    assert "fingerprint_sha256" in capsys.readouterr().out


def test_rejects_bin_occupied_output_drift_and_second_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs, derived, _shot = _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv(gate.RUN_ENV, "1")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    (derived / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(SystemExit):
        j13.main()
    assert "pytorch_model.bin" in capsys.readouterr().out
    (derived / "pytorch_model.bin").unlink()

    monkeypatch.setattr(train_config, "EPOCHS", 1)
    with pytest.raises(SystemExit):
        j13.main()
    assert "differs from the frozen" in capsys.readouterr().out
    monkeypatch.setattr(train_config, "EPOCHS", 200)
    monkeypatch.setattr(j13, "require_trainer_runtime", lambda: None)

    _write_probe(runs)
    output = runs / "uie" / "train" / "seed1-10shot"
    output.mkdir(parents=True)
    (output / "checkpoint-1").mkdir()
    with pytest.raises(SystemExit):
        j13.main()
    assert "already exists and is not empty" in capsys.readouterr().out
    (output / "checkpoint-1").rmdir()

    lock = runs / "uie" / "train" / "publish.lock"
    lock.write_text("999\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        j13.main()
    assert "concurrent training run" in capsys.readouterr().out
    assert lock.read_text(encoding="utf-8") == "999\n"


def test_rejects_probe_that_has_not_passed_and_a_different_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs, _derived, _shot = _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv(gate.RUN_ENV, "1")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trainer was launched")

    monkeypatch.setattr(j13.subprocess, "run", explode)
    monkeypatch.setattr(j13, "require_trainer_runtime", lambda: None)
    with pytest.raises(SystemExit):
        j13.main()
    assert "GPU probe has not passed" in capsys.readouterr().out

    _write_probe(runs, status="BLOCKED_RESOURCE_CONFIG")
    with pytest.raises(SystemExit):
        j13.main()
    assert "GPU probe has not passed" in capsys.readouterr().out

    blocked = runs / "uie" / "j13-probe" / "jobs" / "1364001"
    shutil_report = json.loads((blocked / "probe_report.json").read_text(encoding="utf-8"))
    shutil_report["status"] = "PROBE_PASS"
    shutil_report["batch_16_forward_backward"] = "PASS"
    shutil_report["eval_batch_64_generation"] = "PASS"
    (blocked / "probe_report.json").write_text(json.dumps(shutil_report), encoding="utf-8")
    (runs / "uie" / "j13-probe" / "accepted_job_id.txt").write_text("1364001\n", encoding="utf-8")
    monkeypatch.setattr(j13, "require_trainer_runtime", lambda: None)
    with pytest.raises(SystemExit):
        j13.main()
    assert "size differs" in capsys.readouterr().out
    assert not (runs / "uie" / "train" / "publish.lock").exists()


def test_open_gate_uses_launcher_and_does_not_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs, _derived, _shot = _prepare(tmp_path, monkeypatch)
    monkeypatch.setenv(gate.RUN_ENV, "1")
    _write_probe(runs)
    monkeypatch.setattr(j13, "require_trainer_runtime", lambda: None)
    monkeypatch.setattr(j13, "build_fingerprint", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(j13, "verify_derived_file", lambda _path: None)
    monkeypatch.setattr(j13, "_require_fingerprint_match", lambda *_args: None)
    calls: list[tuple[list[str], dict]] = []
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        command = [str(item) for item in cmd]
        if "uie_trainer_launch.py" not in " ".join(command):
            return real_run(cmd, **kwargs)
        output = runs / "uie" / "train" / "seed1-10shot"
        output.mkdir(parents=True, exist_ok=True)
        (output / "trainer_state.json").write_text(
            json.dumps({"best_metric": None, "best_model_checkpoint": None}),
            encoding="utf-8",
        )
        (output / "model.safetensors").write_bytes(b"trained")
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(j13.subprocess, "run", fake_run)
    assert j13.main() == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert Path(command[1]).name == "uie_trainer_launch.py"
    assert Path(command[2]).name == "run_uie_finetune.py"
    joined = " ".join(command)
    assert "--evaluation_strategy" not in joined
    assert "--eval_strategy=steps" in joined
    assert "--save_steps=20" in joined
    assert "--per_device_train_batch_size=16" in joined
    assert "--per_device_eval_batch_size=64" in joined
    assert "--num_train_epochs=200" in joined
    assert "--resume_from_checkpoint" not in joined
    assert kwargs["env"][gate.LAUNCH_ENV] == "1"
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"][gate.RUN_ENV] == "1"
    assert not (runs / "uie" / "train" / "publish.lock").exists()
    manifest = json.loads((runs / "uie" / "training_manifest.json").read_text(encoding="utf-8"))
    assert "model.safetensors" in manifest["files"]


def test_schedule_matches_upstream_line_counter(tmp_path: Path) -> None:
    import hashlib

    assert gate.planned_schedule(50) == {
        "train_lines": 50,
        "steps_per_epoch": 4,
        "eval_steps": 80,
        "total_optimizer_steps": 800,
        "scheduled_evaluations": 10,
        "epochs": 200,
        "train_batch_size": 16,
        "eval_batch_size": 64,
    }
    train_file = tmp_path / "train.json"
    train_file.write_text("\n".join(["{}"] * 3) + "\n", encoding="utf-8")
    script = ROOT.parent / "_legacy" / "UIE" / "scripts" / "get_eval_batch_num.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(train_file), "16", "20"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(completed.stdout.strip()) == gate.planned_schedule(3)["eval_steps"]
    shot = (
        ROOT.parent
        / "_legacy"
        / "UIE"
        / "dataset_processing"
        / "converted_data"
        / "text2spotasoc"
        / "relation"
        / "conll04_shot"
        / "seed1"
        / "10shot"
    )
    expected_records = {"train.json": 50, "val.json": 231, "test.json": 288}
    for name, records in expected_records.items():
        text = (shot / name).read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) == records
        for line in lines:
            json.loads(line)
        digest = hashlib.sha256((shot / name).read_bytes()).hexdigest()
        assert digest == origin.FROZEN_10SHOT_SHA256[name]
    schema = shot / "record.schema"
    assert hashlib.sha256(schema.read_bytes()).hexdigest() == origin.FROZEN_10SHOT_SHA256["record.schema"]
    real = subprocess.run(
        [sys.executable, str(script), str(shot / "train.json"), "16", "20"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(real.stdout.strip()) == 80


def test_argument_translation_parses_on_transformers_457() -> None:
    legacy = ROOT.parent / "_legacy" / "UIE"
    argv = train_config.training_argv(
        uie_root=legacy,
        train_dir=Path("/tmp/seed1/10shot"),
        model_path=Path("/tmp/uie-base"),
        output_dir=Path("/tmp/uie-out"),
        eval_steps=80,
    )
    assert gate.frozen_argv_problems(argv) == []
    adapted = compat.adapt_training_argv(argv)
    script = r"""
import json
import sys
from pathlib import Path

legacy = Path(sys.argv[1])
sys.path.insert(0, str(legacy))
from transformers import HfArgumentParser
from uie.seq2seq.constrained_seq2seq import ConstraintSeq2SeqTrainingArguments
from uie.seq2seq.trainer_arguments import DataTrainingArguments, ModelArguments

raw_argv = json.loads(sys.argv[2])
adapted_argv = json.loads(sys.argv[3])
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConstraintSeq2SeqTrainingArguments))
raw, raw_remaining = parser.parse_known_args(raw_argv)
parsed, remaining = parser.parse_known_args(adapted_argv)
print(json.dumps({
    "raw_remaining": raw_remaining,
    "raw_eval_strategy": str(raw.eval_strategy),
    "remaining": remaining,
    "eval_strategy": str(parsed.eval_strategy),
    "save_steps": parsed.save_steps,
    "train_batch": parsed.per_device_train_batch_size,
    "eval_batch": parsed.per_device_eval_batch_size,
    "epochs": parsed.num_train_epochs,
    "learning_rate": parsed.learning_rate,
    "metric": parsed.metric_for_best_model,
    "fp16": parsed.fp16,
    "bf16": parsed.bf16,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(legacy), json.dumps(argv[1:]), json.dumps(adapted[1:])],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(completed.stdout.strip().splitlines()[-1])
    assert any(item.startswith("--evaluation_strategy") for item in parsed["raw_remaining"])
    assert "no" in parsed["raw_eval_strategy"].lower()
    assert parsed["remaining"] == []
    assert parsed["eval_strategy"] == "steps"
    assert parsed["save_steps"] == 80
    assert parsed["train_batch"] == 16
    assert parsed["eval_batch"] == 64
    assert parsed["epochs"] == 200
    assert parsed["learning_rate"] == 1e-4
    assert parsed["metric"] == "eval_overall-F1"
    assert parsed["fp16"] is False
    assert parsed["bf16"] is False
    import importlib.util

    runtime_incomplete = importlib.util.find_spec("datasets") is None or importlib.util.find_spec("pyarrow") is None
    try:
        import importlib.metadata as metadata

        metadata.version("accelerate")
    except metadata.PackageNotFoundError:
        runtime_incomplete = True
    if runtime_incomplete:
        with pytest.raises(SystemExit):
            compat.require_trainer_runtime()
    else:
        compat.require_trainer_runtime()


@pytest.fixture
def restore_transformers_module():
    import sys

    original = sys.modules.get("transformers")
    try:
        yield
    finally:
        if original is not None:
            sys.modules["transformers"] = original


def test_trainer_signature_shims(
    monkeypatch: pytest.MonkeyPatch,
    restore_transformers_module: None,
) -> None:
    seen: list[object] = []

    def fake_step(_self, _model, _inputs, num_items_in_batch=None):
        seen.append(num_items_in_batch)
        return "loss"

    from transformers import Seq2SeqTrainer

    monkeypatch.setattr(Seq2SeqTrainer, "training_step", fake_step)
    assert compat.compatible_training_step(object(), None, {}, 4) == "loss"
    assert seen == [4]

    class Box:
        oom_batch = 0

    def oom(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(Seq2SeqTrainer, "training_step", oom)
    box = Box()
    with pytest.raises(RuntimeError, match="out of memory"):
        compat.compatible_training_step(box, None, {}, 1)
    assert box.oom_batch == 1

    class Control:
        def __init__(self) -> None:
            self.should_log = False
            self.should_evaluate = True
            self.should_save = True

    class State:
        global_step = 4
        best_metric = 0.8

    class Args:
        start_eval_step = 0
        save_better_checkpoint = True
        metric_for_best_model = "eval_overall-F1"

    saved: list[str] = []

    class Trainer:
        def __init__(self) -> None:
            self.control = Control()
            self.state = State()
            self.args = Args()

        def evaluate(self):
            return {"eval_overall-F1": 0.2}

        def _report_to_hp_search(self, *_args) -> None:
            return None

    monkeypatch.setattr(Seq2SeqTrainer, "_save_checkpoint", lambda *_args: saved.append("saved"))
    compat.compatible_maybe_log_save_evaluate(
        Trainer(), None, None, None, None, None, None, None, learning_rate=1e-4
    )
    assert saved == []

    restored = type("Restored", (), {})()
    restored.args = type("GenArgs", (), {"generation_max_length": None, "generation_num_beams": None, "fp16": False, "bf16": False})()
    compat.restore_removed_trainer_attributes(restored)
    assert restored._max_length is None
    assert restored._num_beams is None
    assert restored.use_amp is False


def test_dataset_loader_does_not_call_the_removed_script_api(capsys: pytest.CaptureFixture[str]) -> None:
    def original(*_args, **_kwargs):
        raise AssertionError("dataset script loader was called")

    wrapped = compat.patched_load_dataset(original, Path("/tmp/UIE"))
    with pytest.raises(SystemExit):
        wrapped("hub-dataset")
    assert "refuses a dataset" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        wrapped("uie_json.py", data_files={"train": "train.json"})
    assert "dataset script loader was called" not in capsys.readouterr().out


def test_launcher_and_probe_cannot_start_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(gate.RUN_ENV, raising=False)
    monkeypatch.delenv(gate.LAUNCH_ENV, raising=False)
    with pytest.raises(SystemExit):
        launcher.main()
    assert "launcher is not authorized" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        probe.main([])
    assert "GPU probe is not authorized" in capsys.readouterr().out
    monkeypatch.setenv(gate.PROBE_ENV, "1")
    monkeypatch.setenv(gate.RUN_ENV, "1")
    with pytest.raises(SystemExit):
        probe.main([])
    assert "refuses a training authorization" in capsys.readouterr().out

    source = (SCRIPTS / "j13_uie_gpu_probe.py").read_text(encoding="utf-8")
    assert "j13_uie_train" not in source
    assert "run_uie_finetune" not in source
    assert "trainer.train" not in source
    submitter = (SCRIPTS / "submit_train_first.sh").read_text(encoding="utf-8")
    assert "j13_uie_gpu_probe.sbatch" not in submitter
    for name in ("j13_uie_train.sbatch", "j13_uie_gpu_probe.sbatch"):
        text = (JOBS / name).read_text(encoding="utf-8")
        assert f"{gate.RUN_ENV}=1" not in text
        assert f"{gate.PROBE_ENV}=1" not in text
        assert "source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh" in text
    j13_source = (SCRIPTS / "j13_uie_train.py").read_text(encoding="utf-8")
    assert "torch.load" not in j13_source
    assert "mirror" not in j13_source
    assert "J13 training is not authorized" in j13_source
    compat_source = (SCRIPTS / "uie_trainer_compat.py").read_text(encoding="utf-8")
    assert "torch.load" not in compat_source
    assert "weights_only=False" not in compat_source
    document = (ROOT / "docs" / "CESVIMA_TRAIN_FIRST.md").read_text(encoding="utf-8")
    assert "descart" not in document.lower()
    assert "rejected" not in document.lower()
    assert gate.ACCEPTED_J12_JOB_ID in document
    assert gate.ACCEPTED_FINGERPRINT_SHA256 in document
