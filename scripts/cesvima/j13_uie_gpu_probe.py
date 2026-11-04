"""GPU micropreflight for the frozen UIE J13 configuration.

This process does not call J13 or the upstream training entry point.
It does not change batch size, gradient accumulation, or numeric precision.
A PROBE_PASS report is necessary for a later J13 run and is not that run.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

from cesvima_common import fail, read_json, sha256_file, write_json
from uie_trainer_compat import require_trainer_runtime
from uie_j13_gate import (
    ACCEPTED_SAFETENSORS_BYTES,
    ACCEPTED_SAFETENSORS_SHA256,
    accredited_report_problems,
    planned_schedule,
    probe_authorized,
    run_authorized,
)


def _job_id() -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id or "/" in job_id or "\\" in job_id or job_id in {".", ".."}:
        fail("J13 GPU probe requires SLURM_JOB_ID")
    return job_id


def _line_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def cpu_contract() -> int:
    require_trainer_runtime()
    runs = Path(os.environ["P3_RUNS_ROOT"])
    pointer = runs / "uie" / "shot_dir.txt"
    if not pointer.is_file():
        fail("J13 CPU contract cannot find the shot pointer")
    shot = Path(pointer.read_text(encoding="utf-8").strip())
    counts = {
        name: _line_count(shot / name)
        for name in ("train.json", "val.json", "test.json")
    }
    if counts["train.json"] is None:
        fail(f"J13 CPU contract cannot read {shot / 'train.json'}")
    schedule = planned_schedule(counts["train.json"])
    print(
        json.dumps(
            {
                "status": "CPU_CONTRACT",
                "val_lines": counts["val.json"],
                "test_lines": counts["test.json"],
                **schedule,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _read_j12(runs: Path) -> dict:
    pointer = runs / "uie" / "preflight" / "accepted_job_id.txt"
    if not pointer.is_file():
        fail("J13 GPU probe requires the accepted J12 pointer")
    job_id = pointer.read_text(encoding="utf-8").strip()
    report_path = runs / "uie" / "preflight" / "jobs" / job_id / "preflight_report.json"
    if not report_path.is_file():
        fail("J13 GPU probe cannot read the accepted J12 report")
    report = read_json(report_path)
    if report.get("slurm_job_id") != job_id or report.get("status") != "PREFLIGHT_PASS":
        fail("J13 GPU probe requires a PREFLIGHT_PASS report for the pointer JOBID")
    return report


def _stage(report: dict, key: str, action) -> None:
    import torch

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        action()
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        report[key] = "OOM"
        report[f"{key}_seconds"] = round(time.perf_counter() - started, 6)
        return
    report[key] = "PASS"
    report[f"{key}_seconds"] = round(time.perf_counter() - started, 6)
    report[f"{key}_peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())


def _batch(model, vocab: int, batch: int, source: int, target: int, optimize: bool) -> None:
    import torch

    device = torch.device("cuda")
    input_ids = torch.randint(0, vocab, (batch, source), device=device)
    attention_mask = torch.ones_like(input_ids)
    if optimize:
        labels = torch.randint(0, vocab, (batch, target), device=device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        return
    with torch.inference_mode():
        model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=target)


def gpu_contract() -> int:
    require_trainer_runtime()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not torch.cuda.is_available():
        fail("J13 GPU probe requires a GPU; CPU checks use --cpu-only")
    runs = Path(os.environ["P3_RUNS_ROOT"])
    job_id = _job_id()
    j12 = _read_j12(runs)
    model_dir = Path(str(j12.get("derived_model_dir", "")))
    if not model_dir.is_dir() or any(model_dir.glob("*.bin")):
        fail("J13 GPU probe refuses a model directory that is missing or contains a .bin")
    weights = model_dir / "model.safetensors"
    if not weights.is_file() or weights.is_symlink():
        fail("J13 GPU probe requires model.safetensors")
    measured_sha = sha256_file(weights)
    measured_size = weights.stat().st_size
    evidence = accredited_report_problems(j12)
    if measured_sha != ACCEPTED_SAFETENSORS_SHA256 or measured_size != ACCEPTED_SAFETENSORS_BYTES:
        evidence.append("derived_file")
    if evidence:
        fail("J13 GPU probe evidence does not match the accepted J12 run: " + "; ".join(evidence))

    report: dict = {
        "bf16": False,
        "changed_hyperparameters": False,
        "derived_safetensors_sha256": measured_sha,
        "derived_safetensors_bytes": measured_size,
        "eval_batch_size": 64,
        "fingerprint_sha256": j12.get("fingerprint_sha256"),
        "fp16": False,
        "gpu_name": torch.cuda.get_device_name(0),
        "j12_slurm_job_id": j12.get("slurm_job_id"),
        "slurm_job_id": job_id,
        "train_batch_size": 16,
        "status": "FAIL",
    }
    shot_pointer = runs / "uie" / "shot_dir.txt"
    if shot_pointer.is_file():
        shot = Path(shot_pointer.read_text(encoding="utf-8").strip())
        train_lines = _line_count(shot / "train.json")
        report["train_lines"] = train_lines
        report["val_lines"] = _line_count(shot / "val.json")
        report["test_lines"] = _line_count(shot / "test.json")
        if train_lines:
            report["schedule_if_lines_are_examples"] = planned_schedule(train_lines)

    started = time.perf_counter()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        use_safetensors=True,
        local_files_only=True,
    )
    model.to("cuda")
    model.train()
    report["load_seconds"] = round(time.perf_counter() - started, 6)
    report["vram_after_load_bytes"] = int(torch.cuda.memory_allocated())
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        report["tokenizer_length"] = len(tokenizer)
        report["embedding_rows"] = int(model.get_input_embeddings().weight.shape[0])
    except Exception as exc:
        report["tokenizer_error"] = str(exc)

    vocab = int(model.config.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    report["optimizer"] = "AdamW"
    _stage(report, "batch_1_forward_backward", lambda: _batch(model, vocab, 1, 256, 192, True))
    optimizer.zero_grad(set_to_none=True)
    _stage(report, "batch_16_forward_backward", lambda: _batch(model, vocab, 16, 256, 192, True))
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    _stage(report, "eval_batch_64_generation", lambda: _batch(model, vocab, 64, 256, 192, False))

    probe_dir = runs / "uie" / "j13-probe" / "jobs" / job_id
    if report["batch_16_forward_backward"] == "PASS" and report["eval_batch_64_generation"] == "PASS":
        destination = probe_dir / "checkpoint-probe"
        save_started = time.perf_counter()
        model.save_pretrained(destination, safe_serialization=True)
        saved = destination / "model.safetensors"
        report["checkpoint_save"] = "PASS" if saved.is_file() else "FAIL"
        report["checkpoint_save_seconds"] = round(time.perf_counter() - save_started, 6)
        report["checkpoint_bytes"] = saved.stat().st_size if saved.is_file() else 0
    else:
        report["checkpoint_save"] = "NOT_RUN"

    report["max_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report["ru_maxrss_raw"] = int(usage.ru_maxrss)
    report["ru_maxrss_unit"] = "bytes" if sys.platform == "darwin" else "kilobytes"
    blocked = (
        report["batch_16_forward_backward"] != "PASS"
        or report["eval_batch_64_generation"] != "PASS"
    )
    report["status"] = "BLOCKED_RESOURCE_CONFIG" if blocked else "PROBE_PASS"
    if report["status"] == "PROBE_PASS" and report.get("checkpoint_save") != "PASS":
        report["status"] = "FAIL"
    write_json(probe_dir / "probe_report.json", report)
    if report["status"] == "PROBE_PASS":
        pointer = runs / "uie" / "j13-probe" / "accepted_job_id.txt"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(pointer, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            report["pointer_already_present"] = True
            write_json(probe_dir / "probe_report.json", report)
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(job_id + "\n")
        print("PASS J13 GPU probe", flush=True)
        return 0
    print(f"FAIL: J13 GPU probe status {report['status']}", flush=True)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in ([], ["--cpu-only"]):
        fail("J13 GPU probe accepts no arguments other than --cpu-only")
    if not probe_authorized():
        fail("J13 GPU probe is not authorized")
    if run_authorized() or os.environ.get("UIE_J13_LAUNCHED_BY_GATE") == "1":
        fail("J13 GPU probe refuses a training authorization")
    if args == ["--cpu-only"]:
        return cpu_contract()
    return gpu_contract()


if __name__ == "__main__":
    raise SystemExit(main())
