"""J13: train only seed1/10shot after the J12 fingerprint matches."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cesvima_common import fail, read_json, sha256_file, write_json
from uie_fingerprint import build_fingerprint, fingerprint_digest
from uie_train_config import require_10shot, training_argv


def _require_preflight(current: dict) -> None:
    report_path = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "preflight" / "preflight_report.json"
    fingerprint_path = (
        Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "preflight" / "environment_fingerprint.json"
    )
    if not report_path.is_file() or not fingerprint_path.is_file():
        fail("J12 preflight artifacts are missing; refusing to train")
    report = read_json(report_path)
    if report.get("status") != "PASS":
        fail(f"J12 preflight status is {report.get('status')}; refusing to train")
    saved = read_json(fingerprint_path)
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


def main() -> int:
    uie_root = Path(os.environ["P3_UPSTREAM_ROOT"]) / "UIE"
    shot_pointer = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "shot_dir.txt"
    shot_dir = require_10shot(Path(shot_pointer.read_text(encoding="utf-8").strip()))
    current = build_fingerprint(uie_root, shot_dir)
    _require_preflight(current)
    snapshot = read_json(Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "base_snapshot.json")
    model_path = Path(snapshot["local_path"])
    output_dir = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "train" / "seed1-10shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        fail(f"training output already exists and is not empty: {output_dir}")
    eval_steps = _eval_steps(uie_root, shot_dir / "train.json")
    argv = training_argv(
        uie_root=uie_root,
        train_dir=shot_dir,
        model_path=model_path,
        output_dir=output_dir,
        eval_steps=eval_steps,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(uie_root)
    env["TOKENIZERS_PARALLELISM"] = "false"
    print("RUN " + " ".join(argv), flush=True)
    completed = subprocess.run([sys.executable, *argv], cwd=uie_root, env=env)
    if completed.returncode != 0:
        fail(f"UIE trainer exited {completed.returncode}")
    _collect(output_dir)
    print("PASS J13 seed1/10shot training", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
