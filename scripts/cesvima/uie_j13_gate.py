"""Per-execution gates for one UIE J13 run.

The module constant UIE_J13_TRAINING_AUTHORIZED stays false and is not a grant.
UIE_J13_RUN_AUTHORIZED is read from the environment of one process. It does not
authorize J12, the conversion job, the GPU probe, or the integration probe.
UIE_J13_INTEGRATION_PROBE_AUTHORIZED does not authorize J13. A missing GPU-probe
report keeps training closed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from cesvima_common import fail, sha256_file
from uie_train_config import training_argv


RUN_ENV = "UIE_J13_RUN_AUTHORIZED"
PROBE_ENV = "UIE_J13_GPU_PROBE_AUTHORIZED"
LAUNCH_ENV = "UIE_J13_LAUNCHED_BY_GATE"
INTEGRATION_ENV = "UIE_J13_INTEGRATION_PROBE_AUTHORIZED"
INTEGRATION_CHILD_ENV = "UIE_J13_INTEGRATION_CHILD"

ACCEPTED_PROBE_JOB_ID = "1363874"
ACCEPTED_PROBE_BATCH16_PEAK_VRAM_BYTES = 14275008512
ACCEPTED_PROBE_GENERATION64_PEAK_VRAM_BYTES = 3223645696
ACCEPTED_PROBE_CHECKPOINT_BYTES = 990185320

ACCEPTED_J12_JOB_ID = "1363190"
ACCEPTED_CONVERSION_JOB_ID = "1363171"
ACCEPTED_FINGERPRINT_SHA256 = (
    "b585f1d9e2fa81cbd6503d47dbb7da64fbf63acf79cbd4a88725843ac6884530"
)
ACCEPTED_SAFETENSORS_SHA256 = (
    "2623672bf5b80a1e9b78f0dd5ccc609792255e58c8845b556ab5822920f3bfd5"
)
ACCEPTED_SAFETENSORS_BYTES = 1187420216

# Items the frozen argv must still contain. save_steps is intentionally
# absent: Transformers 4.57.6 rejects its library default of 500 against the
# computed eval_steps, and the launcher aligns the two at launch.
FROZEN_ARGV_ITEMS = (
    "--do_train",
    "--do_eval",
    "--do_predict",
    "--predict_with_generate",
    "--evaluation_strategy=steps",
    "--save_strategy=steps",
    "--load_best_model_at_end",
    "--metric_for_best_model=eval_overall-F1",
    "--save_total_limit=1",
    "--num_train_epochs=200",
    "--task=meta",
    "--per_device_train_batch_size=16",
    "--per_device_eval_batch_size=64",
    "--learning_rate=1e-4",
    "--source_prefix=meta: ",
    "--lr_scheduler_type=constant",
    "--label_smoothing_factor=0",
    "--warmup_ratio=0.06",
    "--max_source_length=256",
    "--max_target_length=192",
    "--max_prefix_length=-1",
    "--decoding_format=spotasoc",
    "--meta_negative=-1",
    "--meta_positive_rate=1",
    "--ordered_prompt=False",
    "--save_better_checkpoint=True",
    "--spot_noise=0.1",
    "--asoc_noise=0.1",
    "--seed=42",
)
FORBIDDEN_ARGV_PREFIXES = (
    "--fp16",
    "--bf16",
    "--from_checkpoint",
    "--gradient_accumulation_steps",
    "--resume_from_checkpoint",
    "--save_steps",
)


def run_authorized() -> bool:
    return os.environ.get(RUN_ENV) == "1"


def probe_authorized() -> bool:
    return os.environ.get(PROBE_ENV) == "1"


def integration_authorized() -> bool:
    return os.environ.get(INTEGRATION_ENV) == "1"


def launched_by_gate() -> bool:
    return os.environ.get(LAUNCH_ENV) == "1"


def planned_schedule(train_lines: int) -> dict[str, int]:
    """Reproduce scripts/get_eval_batch_num.py for the frozen batch and cadence.

    train_lines is a line count, not a verified example count. The caller
    supplies the count; this function does not open a dataset.
    """

    if train_lines < 1:
        fail("J13 schedule requires a positive train line count")
    steps_per_epoch = math.ceil(train_lines / 16)
    eval_steps = steps_per_epoch * 20
    total_steps = steps_per_epoch * 200
    return {
        "train_lines": train_lines,
        "steps_per_epoch": steps_per_epoch,
        "eval_steps": eval_steps,
        "total_optimizer_steps": total_steps,
        "scheduled_evaluations": total_steps // eval_steps,
        "epochs": 200,
        "train_batch_size": 16,
        "eval_batch_size": 64,
    }


def frozen_argv_problems(argv: list[str]) -> list[str]:
    present = set(argv)
    problems = [item for item in FROZEN_ARGV_ITEMS if item not in present]
    for item in argv:
        for prefix in FORBIDDEN_ARGV_PREFIXES:
            if item == prefix or item.startswith(prefix + "="):
                problems.append(item)
    return problems


def assert_frozen_experiment(uie_root: Path, train_dir: Path, model_path: Path, output_dir: Path) -> list[str]:
    argv = training_argv(
        uie_root=uie_root,
        train_dir=train_dir,
        model_path=model_path,
        output_dir=output_dir,
        eval_steps=1,
    )
    problems = frozen_argv_problems(argv)
    if problems:
        fail(
            "J13 experimental configuration differs from the frozen "
            "CoNLL04 seed1/10shot setup: " + "; ".join(problems)
        )
    return argv


def accredited_report_problems(report: dict) -> list[str]:
    problems: list[str] = []
    if str(report.get("slurm_job_id")) != ACCEPTED_J12_JOB_ID:
        problems.append("slurm_job_id")
    if str(report.get("conversion_job_id")) != ACCEPTED_CONVERSION_JOB_ID:
        problems.append("conversion_job_id")
    if report.get("fingerprint_sha256") != ACCEPTED_FINGERPRINT_SHA256:
        problems.append("fingerprint_sha256")
    weights = report.get("derived_weight_hashes")
    safetensors = weights.get("model.safetensors") if isinstance(weights, dict) else None
    if not isinstance(safetensors, dict):
        problems.append("derived_weight_hashes")
    else:
        if safetensors.get("sha256") != ACCEPTED_SAFETENSORS_SHA256:
            problems.append("safetensors_sha256")
        if safetensors.get("size") != ACCEPTED_SAFETENSORS_BYTES:
            problems.append("safetensors_size")
    return problems


def probe_report_problems(report: dict) -> list[str]:
    problems: list[str] = []
    if report.get("status") != "PROBE_PASS":
        problems.append("status")
    if report.get("batch_16_forward_backward") != "PASS":
        problems.append("batch_16_forward_backward")
    if report.get("eval_batch_64_generation") != "PASS":
        problems.append("eval_batch_64_generation")
    if report.get("train_batch_size") != 16 or report.get("eval_batch_size") != 64:
        problems.append("batch_sizes")
    if report.get("changed_hyperparameters") is not False:
        problems.append("changed_hyperparameters")
    if report.get("fp16") is not False or report.get("bf16") is not False:
        problems.append("numeric_precision")
    if report.get("derived_safetensors_sha256") != ACCEPTED_SAFETENSORS_SHA256:
        problems.append("derived_safetensors_sha256")
    if report.get("fingerprint_sha256") != ACCEPTED_FINGERPRINT_SHA256:
        problems.append("fingerprint_sha256")
    vram = report.get("max_vram_bytes")
    if not isinstance(vram, int) or vram <= 0:
        problems.append("max_vram_bytes")
    if not isinstance(report.get("gpu_name"), str) or not report["gpu_name"]:
        problems.append("gpu_name")
    if not isinstance(report.get("slurm_job_id"), str) or not report["slurm_job_id"]:
        problems.append("slurm_job_id")
    return problems


def verify_derived_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        fail("derived model.safetensors is missing")
    if path.stat().st_size != ACCEPTED_SAFETENSORS_BYTES:
        fail("derived model.safetensors size differs from the accepted conversion")
    if sha256_file(path) != ACCEPTED_SAFETENSORS_SHA256:
        fail("derived model.safetensors hash differs from the accepted conversion")
