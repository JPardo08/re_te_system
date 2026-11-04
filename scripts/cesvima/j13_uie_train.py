"""J13: train only seed1/10shot after the J12 fingerprint matches."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cesvima_common import fail, read_json, require_under_scratch, sha256_file, write_json
from uie_fingerprint import build_fingerprint, fingerprint_digest
from uie_j13_gate import (
    LAUNCH_ENV,
    accredited_report_problems,
    assert_frozen_experiment,
    probe_report_problems,
    run_authorized,
    verify_derived_file,
)
from uie_origin_pin import FROZEN_10SHOT_SHA256, UIE_J13_TRAINING_AUTHORIZED
from uie_train_config import require_10shot, training_argv
from uie_trainer_compat import adapt_training_argv, require_trainer_runtime


def _incomplete(report: dict) -> list[str]:
    problems: list[str] = []
    if report.get("deserialization_authorized") is not False:
        problems.append("deserialization_authorized")
    digest = report.get("fingerprint_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        problems.append("fingerprint_sha256")
    weights = report.get("derived_weight_hashes")
    safetensors = weights.get("model.safetensors") if isinstance(weights, dict) else None
    if not isinstance(safetensors, dict) or not isinstance(safetensors.get("sha256"), str):
        problems.append("derived_weight_hashes")
    if report.get("frozen_10shot_hashes") != FROZEN_10SHOT_SHA256:
        problems.append("frozen_10shot_hashes")
    if not isinstance(report.get("derived_model_dir"), str) or not report["derived_model_dir"]:
        problems.append("derived_model_dir")
    if not isinstance(report.get("slurm_job_id"), str) or not report["slurm_job_id"]:
        problems.append("slurm_job_id")
    return problems


def require_identified_preflight() -> tuple[dict, dict]:
    runs = Path(os.environ["P3_RUNS_ROOT"])
    pointer = runs / "uie" / "preflight" / "accepted_job_id.txt"
    if not pointer.is_file():
        fail("J12 preflight is unidentified; refusing to train")
    job_id = pointer.read_text(encoding="utf-8").strip()
    if not job_id or job_id in {".", ".."} or "/" in job_id or "\\" in job_id:
        fail("J12 preflight is unidentified; refusing to train")
    report_path = runs / "uie" / "preflight" / "jobs" / job_id / "preflight_report.json"
    fingerprint_path = report_path.with_name("environment_fingerprint.json")
    if not report_path.is_file() or not fingerprint_path.is_file():
        fail("J12 preflight artifacts are missing; refusing to train")
    report = read_json(report_path)
    if report.get("slurm_job_id") != job_id:
        fail("J12 report belongs to another JOBID; refusing to train")
    if report.get("status") != "PREFLIGHT_PASS":
        fail(f"J12 preflight status is {report.get('status')}; refusing to train")
    problems = _incomplete(report)
    if problems:
        fail("J12 preflight is incomplete; refusing to train: " + "; ".join(problems))
    return report, read_json(fingerprint_path)


def _require_fingerprint_match(saved: dict, current: dict, report: dict) -> None:
    if fingerprint_digest(saved) != fingerprint_digest(current):
        fail(
            "UIE environment fingerprint differs from J12; "
            f"saved {fingerprint_digest(saved)} current {fingerprint_digest(current)}"
        )
    if report.get("fingerprint_sha256") != fingerprint_digest(current):
        fail("J12 report fingerprint digest does not match the recomputed fingerprint")


def _eval_steps(uie_root: Path, train_file: Path) -> int:
    completed = subprocess.run(
        [
            sys.executable,
            str(uie_root / "scripts" / "get_eval_batch_num.py"),
            str(train_file),
            "16",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def _collect(output_dir: Path) -> None:
    trainer_state_path = output_dir / "trainer_state.json"
    if not trainer_state_path.is_file():
        fail(f"trainer_state.json is missing from {output_dir}")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    files = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.stat().st_size > 0 and path.suffix in {".bin", ".safetensors", ".json", ".model"}:
            files[path.relative_to(output_dir).as_posix()] = {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
    if not any(name.endswith(("pytorch_model.bin", "model.safetensors")) for name in files):
        fail("training output has no model weight file")
    write_json(
        Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "training_manifest.json",
        {
            "best_metric": trainer_state.get("best_metric"),
            "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
            "best_model_metric": "eval_overall-F1",
            "files": files,
            "output_dir": str(output_dir),
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "job_name": os.environ.get("SLURM_JOB_NAME"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
            },
            "trainer_state": str(trainer_state_path),
            "training_args": str(output_dir / "training_args.bin"),
        },
    )


def _require_accredited(report: dict) -> None:
    problems = accredited_report_problems(report)
    if problems:
        fail("J13 refuses evidence that is not the accepted J12 run: " + "; ".join(problems))


def _require_probe(runs: Path) -> dict:
    pointer = runs / "uie" / "j13-probe" / "accepted_job_id.txt"
    if not pointer.is_file():
        fail("J13 GPU probe has not passed; refusing to train")
    job_id = pointer.read_text(encoding="utf-8").strip()
    if not job_id or job_id in {".", ".."} or "/" in job_id or "\\" in job_id:
        fail("J13 GPU probe has not passed; refusing to train")
    report_path = runs / "uie" / "j13-probe" / "jobs" / job_id / "probe_report.json"
    if not report_path.is_file():
        fail("J13 GPU probe has not passed; refusing to train")
    report = read_json(report_path)
    if report.get("slurm_job_id") != job_id:
        fail("J13 GPU probe report belongs to another JOBID; refusing to train")
    problems = probe_report_problems(report)
    if problems:
        fail("J13 GPU probe has not passed; refusing to train: " + "; ".join(problems))
    return report


def _acquire_lock(lock_path: Path, job_id: str) -> None:
    require_under_scratch(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        fail("J13 refuses a concurrent training run; publish.lock is already held")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(job_id + "\n")


def _release_lock(lock_path: Path, job_id: str) -> None:
    if not lock_path.is_file():
        return
    if lock_path.read_text(encoding="utf-8").strip() != job_id:
        return
    lock_path.unlink()


def main() -> int:
    report, saved = require_identified_preflight()
    if UIE_J13_TRAINING_AUTHORIZED is not False or not run_authorized():
        fail("J13 training is not authorized")
    _require_accredited(report)
    runs = Path(os.environ["P3_RUNS_ROOT"])
    uie_root = Path(os.environ["P3_UPSTREAM_ROOT"]) / "UIE"
    shot_pointer = runs / "uie" / "shot_dir.txt"
    shot_dir = require_10shot(Path(shot_pointer.read_text(encoding="utf-8").strip()))
    model_path = Path(report["derived_model_dir"])
    if not model_path.is_dir() or any(model_path.glob("*.bin")):
        fail("J13 refuses to train from pytorch_model.bin")
    assert_frozen_experiment(uie_root, shot_dir, model_path, runs / "uie" / "train" / "seed1-10shot")
    require_trainer_runtime()
    _require_probe(runs)
    output_dir = runs / "uie" / "train" / "seed1-10shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        fail(f"training output already exists and is not empty: {output_dir}")
    job_id = os.environ.get("SLURM_JOB_ID") or "local"
    lock_path = runs / "uie" / "train" / "publish.lock"
    _acquire_lock(lock_path, job_id)
    try:
        verify_derived_file(model_path / "model.safetensors")
        current = build_fingerprint(uie_root, shot_dir, model_path)
        _require_fingerprint_match(saved, current, report)
        eval_steps = _eval_steps(uie_root, shot_dir / "train.json")
        argv = adapt_training_argv(
            training_argv(
                uie_root=uie_root,
                train_dir=shot_dir,
                model_path=model_path,
                output_dir=output_dir,
                eval_steps=eval_steps,
            )
        )
        env = os.environ.copy()
        scripts = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = str(uie_root) + os.pathsep + scripts
        if os.environ.get("PYTHONPATH"):
            env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env[LAUNCH_ENV] = "1"
        launcher = Path(__file__).resolve().with_name("uie_trainer_launch.py")
        print("RUN " + " ".join(argv), flush=True)
        completed = subprocess.run(
            [sys.executable, str(launcher), *argv],
            cwd=uie_root,
            env=env,
        )
        if completed.returncode != 0:
            fail(f"UIE trainer exited {completed.returncode}")
        _collect(output_dir)
    finally:
        _release_lock(lock_path, job_id)
    print("PASS J13 seed1/10shot training", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
