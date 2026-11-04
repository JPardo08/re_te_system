"""Isolated optimizer step and one-step trainer check for UIE J13.

This process does not call J13 or J14 and does not write their accepted
pointers. A passing report does not authorize the 200-epoch run. The
max_steps and sample caps below apply only to this probe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cesvima_common import fail, read_json, require_under_scratch, sha256_file, write_json
from uie_j13_gate import (
    ACCEPTED_J12_JOB_ID,
    ACCEPTED_PROBE_BATCH16_PEAK_VRAM_BYTES,
    ACCEPTED_PROBE_CHECKPOINT_BYTES,
    ACCEPTED_PROBE_GENERATION64_PEAK_VRAM_BYTES,
    ACCEPTED_PROBE_JOB_ID,
    ACCEPTED_SAFETENSORS_BYTES,
    ACCEPTED_SAFETENSORS_SHA256,
    INTEGRATION_CHILD_ENV,
    INTEGRATION_ENV,
    accredited_report_problems,
    integration_authorized,
    launched_by_gate,
    planned_schedule,
    run_authorized,
)
from uie_safetensors_conversion import tensor_values_sha256
from uie_trainer_compat import adapt_training_argv, install_shims, require_trainer_runtime
from uie_train_config import (
    ASOC_NOISE,
    MAX_PREFIX_LENGTH,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    META_NEGATIVE,
    META_POSITIVE_RATE,
    SPOT_NOISE,
    training_argv,
)


PROBE_MAX_STEPS = 1
PROBE_MAX_TRAIN_SAMPLES = 16
PROBE_MAX_EVAL_SAMPLES = 16
PROBE_EVAL_STEPS = 1
PROTECTED_PREFIXES = ("uie/train", "uie/preflight", "uie/conversion", "uie/j13-probe")


def accepted_probe_vram_note() -> dict[str, str]:
    """Explain the accepted probe's max_vram_bytes without rewriting that job.

    Each stage calls reset_peak_memory_stats before its work. The final
    max_vram_bytes read happens after the generation stage, so it is the peak
    since that reset, not the batch-16 peak. The probe also times CUDA work
    without synchronize and never calls optimizer.step.
    """

    return {
        "allocated": "bytes of live tensors, from memory_allocated",
        "reserved": "bytes held by the caching allocator, from memory_reserved; not recorded by job 1363874",
        "free": "bytes returned by mem_get_info; not recorded by job 1363874",
        "max_vram_bytes": "max_memory_allocated after the generation stage reset its peak counter",
        "batch_16_peak": "max_memory_allocated during random-token forward and backward only",
        "timing": "perf_counter without cuda.synchronize, so the recorded seconds can end before the kernel",
        "optimizer": "AdamW is constructed and zero_grad is called; optimizer.step is not called",
    }


def classify_checkpoint_keys(
    origin_keys: list[str],
    saved_keys: list[str],
    shared_groups: list[list[str]],
) -> dict[str, object]:
    """Separate tied names omitted by a later save from unexplained loss.

    A missing key is explained only when it belongs to a group that still has
    at least one name in the saved file. This does not prove equal values.
    """

    origin = set(origin_keys)
    saved = set(saved_keys)
    missing = sorted(origin - saved)
    extra = sorted(saved - origin)
    explained: list[str] = []
    unexplained: list[str] = []
    for name in missing:
        group = next((item for item in shared_groups if name in item), None)
        if group is not None and any(member in saved for member in group):
            explained.append(name)
        else:
            unexplained.append(name)
    return {
        "equivalence": "not_claimed",
        "explained_missing_keys": explained,
        "extra_keys": extra,
        "missing_keys": missing,
        "unexplained_missing_keys": unexplained,
    }


def limited_training_argv(
    *,
    uie_root: Path,
    train_dir: Path,
    model_path: Path,
    output_dir: Path,
) -> list[str]:
    argv = adapt_training_argv(
        training_argv(
            uie_root=uie_root,
            train_dir=train_dir,
            model_path=model_path,
            output_dir=output_dir,
            eval_steps=PROBE_EVAL_STEPS,
        )
    )
    argv.extend(
        [
            f"--max_steps={PROBE_MAX_STEPS}",
            f"--max_train_samples={PROBE_MAX_TRAIN_SAMPLES}",
            f"--max_val_samples={PROBE_MAX_EVAL_SAMPLES}",
            f"--max_test_samples={PROBE_MAX_EVAL_SAMPLES}",
        ]
    )
    return argv


def _job_id() -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id or "/" in job_id or "\\" in job_id or job_id in {".", ".."}:
        fail("J13 integration probe requires SLURM_JOB_ID")
    return job_id


def assert_output_not_protected(runs: Path, path: Path) -> Path:
    relative = path.resolve().relative_to(runs.resolve()).as_posix()
    if any(relative == prefix or relative.startswith(prefix + "/") for prefix in PROTECTED_PREFIXES):
        fail("J13 integration probe refuses a protected output path")
    return path


def _job_dir(runs: Path, job_id: str) -> Path:
    return assert_output_not_protected(runs, runs / "uie" / "j13-integration" / "jobs" / job_id)


def _read_only_probe(runs: Path) -> dict:
    pointer = runs / "uie" / "j13-probe" / "accepted_job_id.txt"
    if not pointer.is_file():
        fail("J13 integration probe requires the accepted GPU probe pointer")
    job_id = pointer.read_text(encoding="utf-8").strip()
    if job_id != ACCEPTED_PROBE_JOB_ID:
        fail("J13 integration probe requires GPU probe JOBID " + ACCEPTED_PROBE_JOB_ID)
    report_path = runs / "uie" / "j13-probe" / "jobs" / job_id / "probe_report.json"
    before = report_path.read_bytes()
    report = read_json(report_path)
    if report_path.read_bytes() != before:
        fail("J13 integration probe modified the accepted GPU probe report")
    if report.get("slurm_job_id") != job_id or report.get("status") != "PROBE_PASS":
        fail("J13 integration probe requires PROBE_PASS for job " + ACCEPTED_PROBE_JOB_ID)
    expected = {
        "batch_16_forward_backward_peak_vram_bytes": ACCEPTED_PROBE_BATCH16_PEAK_VRAM_BYTES,
        "eval_batch_64_generation_peak_vram_bytes": ACCEPTED_PROBE_GENERATION64_PEAK_VRAM_BYTES,
        "checkpoint_bytes": ACCEPTED_PROBE_CHECKPOINT_BYTES,
    }
    mismatched = [key for key, value in expected.items() if report.get(key) != value]
    if mismatched:
        fail("J13 integration probe evidence does not match job 1363874: " + "; ".join(mismatched))
    return report


def _read_j12(runs: Path) -> dict:
    pointer = runs / "uie" / "preflight" / "accepted_job_id.txt"
    before = pointer.read_bytes() if pointer.is_file() else None
    if before is None:
        fail("J13 integration probe requires the accepted J12 pointer")
    job_id = pointer.read_text(encoding="utf-8").strip()
    if job_id != ACCEPTED_J12_JOB_ID or pointer.read_bytes() != before:
        fail("J13 integration probe requires J12 JOBID " + ACCEPTED_J12_JOB_ID)
    report = read_json(runs / "uie" / "preflight" / "jobs" / job_id / "preflight_report.json")
    problems = accredited_report_problems(report)
    if problems or report.get("status") != "PREFLIGHT_PASS":
        fail("J13 integration probe J12 evidence is incomplete: " + "; ".join(problems))
    return report


def _acquire_lock(lock_path: Path, job_id: str) -> None:
    require_under_scratch(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        fail("J13 integration probe refuses a concurrent integration run")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(job_id + "\n")


def _release_lock(lock_path: Path, job_id: str) -> None:
    if lock_path.is_file() and lock_path.read_text(encoding="utf-8").strip() == job_id:
        lock_path.unlink()


def cpu_contract() -> int:
    runs = Path(os.environ["P3_RUNS_ROOT"])
    probe = _read_only_probe(runs)
    j12 = _read_j12(runs)
    require_trainer_runtime()
    shot = Path((runs / "uie" / "shot_dir.txt").read_text(encoding="utf-8").strip())
    counts = {}
    for name in ("train.json", "val.json", "test.json"):
        path = shot / name
        if not path.is_file():
            fail(f"J13 integration CPU contract cannot read {path}")
        counts[name] = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    schedule = planned_schedule(counts["train.json"])
    print(
        json.dumps(
            {
                "accepted_probe_job": probe.get("slurm_job_id"),
                "authorizes_j13": False,
                "j12_job": j12.get("slurm_job_id"),
                "status": "CPU_CONTRACT",
                "val_records": counts["val.json"],
                "test_records": counts["test.json"],
                **schedule,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _cuda_snapshot(tag: str) -> dict[str, int | str]:
    import torch

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "free_bytes": int(free_bytes),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "tag": tag,
        "total_bytes": int(total_bytes),
    }


def _shared_groups(model) -> list[list[str]]:
    buckets: dict[tuple[int, int], list[str]] = {}
    for name, tensor in model.state_dict().items():
        if not hasattr(tensor, "untyped_storage"):
            continue
        storage = tensor.untyped_storage()
        buckets.setdefault((storage.data_ptr(), storage.nbytes()), []).append(name)
    return [sorted(names) for names in buckets.values() if len(names) > 1]


def _same_tensor(left, right) -> bool:
    if tuple(left.shape) != tuple(right.shape) or str(left.dtype) != str(right.dtype):
        return False
    return tensor_values_sha256(left) == tensor_values_sha256(right)


def _compare_saved_weights(model, origin_file: Path, saved_file: Path) -> dict:
    from safetensors import safe_open

    groups = _shared_groups(model)
    with safe_open(origin_file, framework="pt") as origin, safe_open(saved_file, framework="pt") as saved:
        origin_keys = list(origin.keys())
        saved_keys = list(saved.keys())
        classified = classify_checkpoint_keys(origin_keys, saved_keys, groups)
        compared = []
        mismatched = []
        for name in sorted(set(origin_keys) & set(saved_keys)):
            compared.append(name)
            if not _same_tensor(origin.get_tensor(name), saved.get_tensor(name)):
                mismatched.append(name)
        accounted = []
        unaccounted = []
        for name in classified["explained_missing_keys"]:
            group = next(item for item in groups if name in item)
            survivors = [member for member in group if member in saved_keys and member != name]
            origin_tensor = origin.get_tensor(name)
            if any(_same_tensor(origin_tensor, saved.get_tensor(member)) for member in survivors):
                accounted.append(name)
            else:
                unaccounted.append(name)
    same_keys = not classified["missing_keys"] and not classified["extra_keys"]
    shared_accounted = (
        bool(classified["missing_keys"])
        and not classified["extra_keys"]
        and not classified["unexplained_missing_keys"]
        and not unaccounted
    )
    if not mismatched and compared and same_keys:
        equivalence = "keys_shapes_dtypes_and_values_match"
        meaning = "every shared key matches in shape, dtype and value hash"
    elif not mismatched and compared and shared_accounted:
        equivalence = "accounted_by_shared_storage"
        meaning = (
            "intersecting keys match in shape, dtype and value hash, and every omitted "
            "key matches a survivor from the same loaded storage"
        )
    else:
        equivalence = "not_claimed"
        meaning = "file equality is not claimed"
    return {
        "accounted_missing_keys": accounted,
        "compared_key_count": len(compared),
        "equivalence": equivalence,
        "equivalence_meaning": meaning,
        "extra_keys": classified["extra_keys"],
        "missing_keys": classified["missing_keys"],
        "origin_bytes": origin_file.stat().st_size,
        "origin_key_count": len(origin_keys),
        "saved_bytes": saved_file.stat().st_size,
        "saved_key_count": len(saved_keys),
        "shared_groups": groups,
        "unaccounted_missing_keys": unaccounted,
        "unexplained_missing_keys": classified["unexplained_missing_keys"],
        "value_mismatches_before_training": mismatched,
    }


def _real_training_batch(model, tokenizer, rows: list[dict]):
    from transformers import set_seed

    from uie.extraction import constants
    from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser
    from uie.extraction.record_schema import RecordSchema
    from uie.seq2seq.data_collator.meta_data_collator import (
        DataCollatorForMetaSeq2Seq,
        DynamicSSIGenerator,
    )

    set_seed(42)
    before_rows = int(model.get_input_embeddings().weight.shape[0])
    missing = [
        token
        for token in (
            constants.type_start,
            constants.type_end,
            constants.text_start,
            constants.span_start,
            constants.spot_prompt,
            constants.asoc_prompt,
        )
        if token not in tokenizer.get_vocab()
    ]
    if missing:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": tokenizer.special_tokens_map_extended["additional_special_tokens"]
                + missing
            }
        )
        model.resize_token_embeddings(len(tokenizer))
    after_rows = int(model.get_input_embeddings().weight.shape[0])
    shot = Path((Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "shot_dir.txt").read_text(encoding="utf-8").strip())
    schema = RecordSchema.read_from_file(str(shot / "record.schema"))
    features = []
    for row in rows:
        encoded = tokenizer(row["text"], max_length=int(MAX_SOURCE_LENGTH), truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(row["record"], max_length=int(MAX_TARGET_LENGTH), truncation=True)
        features.append(
            {
                "asocs": row["asoc"],
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels["input_ids"],
                "sample_prompt": True,
                "spot_asoc": row["spot_asoc"],
                "spots": row["spot"],
            }
        )
    collator = DataCollatorForMetaSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        max_length=int(MAX_SOURCE_LENGTH),
        max_prefix_length=int(MAX_PREFIX_LENGTH),
        max_target_length=int(MAX_TARGET_LENGTH),
        negative_sampler=DynamicSSIGenerator(
            tokenizer=tokenizer,
            schema=schema,
            positive_rate=float(META_POSITIVE_RATE),
            negative=int(META_NEGATIVE),
            ordered_prompt=False,
        ),
        spot_asoc_nosier=SpotAsocNoiser(
            spot_noise_ratio=float(SPOT_NOISE),
            asoc_noise_ratio=float(ASOC_NOISE),
            null_span=constants.null_span,
        ),
        decoding_format="spotasoc",
    )
    batch = collator(features)
    return batch, {"embedding_rows_after_resize": after_rows, "embedding_rows_before_resize": before_rows}


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _optimization_step(model, batch, report: dict) -> str:
    import torch

    device = torch.device("cuda")
    optimizer = None
    report["optimizer"] = {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "lr": 1e-4,
        "name": "torch.optim.AdamW",
        "weight_decay": 0.0,
    }
    report["seed"] = 42
    report["bit_exact_with_full_trainer_step"] = False
    report["snapshots"] = [_cuda_snapshot("after_load")]
    moved = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    report["batch_shapes"] = {
        key: list(value.shape) for key, value in moved.items() if hasattr(value, "shape")
    }
    report["snapshots"].append(_cuda_snapshot("after_batch_to_device"))
    torch.cuda.reset_peak_memory_stats()
    try:
        started = time.perf_counter()
        loss = model(**moved).loss
        torch.cuda.synchronize()
        report["forward_seconds"] = round(time.perf_counter() - started, 6)
        report["snapshots"].append(_cuda_snapshot("after_forward"))
        started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        report["backward_seconds"] = round(time.perf_counter() - started, 6)
        report["snapshots"].append(_cuda_snapshot("after_backward"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
        report["snapshots"].append(_cuda_snapshot("after_optimizer_init"))
        started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        report["optimizer_step_seconds"] = round(time.perf_counter() - started, 6)
        report["snapshots"].append(_cuda_snapshot("after_optimizer_step"))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        report["snapshots"].append(_cuda_snapshot("after_zero_grad"))
        report["diagnostic_loss"] = float(loss.detach().cpu())
        report["global_max_allocated_bytes"] = max(item["max_allocated_bytes"] for item in report["snapshots"])
        report["global_max_reserved_bytes"] = max(item["max_reserved_bytes"] for item in report["snapshots"])
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        report["optimization_step"] = "OOM"
        report["oom_error"] = str(exc)
        return "OOM"
    finally:
        del optimizer
    report["optimization_step"] = "PASS"
    return "PASS"


def _load_rows(model_dir: Path, shot: Path) -> tuple[object, object, list[dict]]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from uie_trainer_compat import load_uie_json_dataset

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    dataset = load_uie_json_dataset(
        {
            "train": str(shot / "train.json"),
            "validation": str(shot / "val.json"),
            "test": str(shot / "test.json"),
        },
        Path(os.environ["P3_UPSTREAM_ROOT"]) / "UIE",
    )
    rows = dataset["train"].select(range(PROBE_MAX_TRAIN_SAMPLES)).to_list()
    return model, tokenizer, rows


def run_limited_trainer(job_dir: Path) -> int:
    if os.environ.get(INTEGRATION_CHILD_ENV) != "1":
        fail("J13 integration trainer child is not authorized")
    request = read_json(job_dir / "trainer_request.json")
    uie_root = Path(request["uie_root"])
    output_dir = require_under_scratch(Path(request["output_dir"]))
    try:
        output_dir.resolve().relative_to(job_dir.resolve())
    except ValueError:
        fail("J13 integration trainer refuses an output outside its job directory")
    os.chdir(uie_root)
    sys.path.insert(0, str(uie_root))
    install_shims(uie_root)
    sys.argv = limited_training_argv(
        uie_root=uie_root,
        train_dir=Path(request["shot_dir"]),
        model_path=Path(request["model_dir"]),
        output_dir=output_dir,
    )
    import runpy

    runpy.run_path(str(uie_root / "run_uie_finetune.py"), run_name="__main__")
    return 0


def _metric_present(output_dir: Path) -> str | None:
    state_path = output_dir / "trainer_state.json"
    if not state_path.is_file():
        return None
    text = state_path.read_text(encoding="utf-8")
    eval_file = output_dir / "eval_results_seq2seq.txt"
    if eval_file.is_file():
        text += "\n" + eval_file.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "eval_overall-F1" in line:
            return line.strip()
    if "eval_overall-F1" in text:
        return "eval_overall-F1"
    return None


def gpu_contract() -> int:
    require_trainer_runtime()
    import torch

    if not torch.cuda.is_available():
        fail("J13 integration probe requires a GPU; CPU checks use --cpu-only")
    runs = Path(os.environ["P3_RUNS_ROOT"])
    if (runs / "uie" / "train" / "publish.lock").exists():
        fail("J13 integration probe refuses to run while J13 holds publish.lock")
    job_id = _job_id()
    job_dir = require_under_scratch(_job_dir(runs, job_id))
    lock_path = runs / "uie" / "j13-integration" / "publish.lock"
    _acquire_lock(lock_path, job_id)
    report: dict = {
        "authorizes_j13": False,
        "changed_canonical_config": False,
        "execution_limits": {
            "canonical_eval_steps_for_50_lines": 80,
            "canonical_epochs": 200,
            "eval_steps": PROBE_EVAL_STEPS,
            "max_steps": PROBE_MAX_STEPS,
            "max_test_samples": PROBE_MAX_EVAL_SAMPLES,
            "max_train_samples": PROBE_MAX_TRAIN_SAMPLES,
            "max_val_samples": PROBE_MAX_EVAL_SAMPLES,
            "do_predict_samples": PROBE_MAX_EVAL_SAMPLES,
            "scientific_result": False,
        },
        "numeric_precision": "fp32",
        "slurm_job_id": job_id,
        "status": "FAIL",
        "vram_accounting_of_probe_1363874": accepted_probe_vram_note(),
    }
    try:
        accepted = _read_only_probe(runs)
        j12 = _read_j12(runs)
        report["accepted_probe_max_vram_bytes"] = accepted.get("max_vram_bytes")
        report["j12_slurm_job_id"] = j12.get("slurm_job_id")
        model_dir = Path(str(j12["derived_model_dir"]))
        weights = model_dir / "model.safetensors"
        if sha256_file(weights) != ACCEPTED_SAFETENSORS_SHA256 or weights.stat().st_size != ACCEPTED_SAFETENSORS_BYTES:
            fail("J13 integration probe derived checkpoint does not match the accepted conversion")
        shot = Path((runs / "uie" / "shot_dir.txt").read_text(encoding="utf-8").strip())
        model, tokenizer, rows = _load_rows(model_dir, shot)
        model.to("cuda")
        model.train()
        fresh = require_under_scratch(job_dir / "fresh-save")
        report["fresh_save_dir"] = str(fresh)
        report["fresh_save_is_j13_checkpoint"] = False
        model.save_pretrained(fresh, safe_serialization=True)
        report["checkpoint_comparison_before_training"] = _compare_saved_weights(
            model, weights, fresh / "model.safetensors"
        )
        batch, resize = _real_training_batch(model, tokenizer, rows)
        report.update(resize)
        step = _optimization_step(model, batch, report)
        del model, tokenizer, batch
        torch.cuda.empty_cache()
        if step == "OOM":
            report["status"] = "BLOCKED_RESOURCE_CONFIG"
            write_json(job_dir / "integration_report.json", report)
            print("FAIL: J13 integration optimization step ran out of memory", flush=True)
            return 2
        output_dir = job_dir / "trainer"
        write_json(
            job_dir / "trainer_request.json",
            {
                "model_dir": str(model_dir),
                "output_dir": str(output_dir),
                "shot_dir": str(shot),
                "uie_root": str(Path(os.environ["P3_UPSTREAM_ROOT"]) / "UIE"),
            },
        )
        env = os.environ.copy()
        env[INTEGRATION_CHILD_ENV] = "1"
        env.pop("UIE_J13_RUN_AUTHORIZED", None)
        env.pop("UIE_J13_LAUNCHED_BY_GATE", None)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        scripts = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = str(Path(os.environ["P3_UPSTREAM_ROOT"]) / "UIE") + os.pathsep + scripts
        stdout_path = job_dir / "trainer_stdout.txt"
        stderr_path = job_dir / "trainer_stderr.txt"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--run-limited-trainer", str(job_dir)],
                cwd=Path(os.environ["P3_UPSTREAM_ROOT"]) / "UIE",
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
        report["trainer_exit_code"] = completed.returncode
        report["trainer_stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        if completed.returncode != 0 and "out of memory" in report["trainer_stderr_tail"].lower():
            report["status"] = "BLOCKED_RESOURCE_CONFIG"
            report["trainer_oom"] = True
            write_json(job_dir / "integration_report.json", report)
            print("FAIL: J13 integration trainer ran out of memory", flush=True)
            return 2
        metric = _metric_present(output_dir) if completed.returncode == 0 else None
        report["diagnostic_eval_overall_f1_line"] = metric
        state = output_dir / "trainer_state.json"
        report["trainer_state_present"] = state.is_file()
        saved = output_dir / "model.safetensors"
        report["trainer_checkpoint_present"] = saved.is_file()
        if completed.returncode == 0 and saved.is_file():
            from transformers import AutoModelForSeq2SeqLM

            reloaded = AutoModelForSeq2SeqLM.from_pretrained(
                output_dir,
                local_files_only=True,
                use_safetensors=True,
            )
            report["checkpoint_reread_parameters"] = sum(parameter.numel() for parameter in reloaded.parameters())
            del reloaded
        passed = (
            completed.returncode == 0
            and metric is not None
            and state.is_file()
            and saved.is_file()
            and report.get("checkpoint_reread_parameters", 0) > 0
            and report["authorizes_j13"] is False
            and report["changed_canonical_config"] is False
        )
        report["status"] = "INTEGRATION_PASS" if passed else "FAIL"
        report["authorizes_j13"] = False
        write_json(job_dir / "integration_report.json", report)
    except SystemExit:
        raise
    except Exception as exc:
        report["status"] = "BLOCKED_RESOURCE_CONFIG" if _is_oom(exc) else "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["authorizes_j13"] = False
        try:
            write_json(job_dir / "integration_report.json", report)
        except Exception as write_exc:
            print(f"FAIL: could not write integration report: {write_exc}", flush=True)
        print("FAIL: J13 integration probe status " + report["status"], flush=True)
        return 2 if report["status"] == "BLOCKED_RESOURCE_CONFIG" else 1
    finally:
        _release_lock(lock_path, job_id)
    if report["status"] == "INTEGRATION_PASS":
        print("PASS J13 integration probe", flush=True)
        return 0
    print("FAIL: J13 integration probe status " + report["status"], flush=True)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not integration_authorized():
        fail("J13 integration probe is not authorized")
    if run_authorized() or launched_by_gate():
        fail("J13 integration probe refuses a training authorization")
    if args == ["--cpu-only"]:
        return cpu_contract()
    if len(args) == 2 and args[0] == "--run-limited-trainer":
        return run_limited_trainer(Path(args[1]))
    if args:
        fail("J13 integration probe accepts no arguments other than --cpu-only")
    return gpu_contract()


if __name__ == "__main__":
    raise SystemExit(main())
