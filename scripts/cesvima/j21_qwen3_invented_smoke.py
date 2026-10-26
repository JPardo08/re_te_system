"""Bounded Qwen3 invented-smoke job. It does not score Gold and it does not train.

``--dry-run`` checks the local contract and does not load weights.
``--execute`` is the CESVIMA path: local snapshot only, one fresh directory,
one model load, C0 and C1 on the invented sentences.
``--prepare-c0-c1`` writes the paired plan after a technical smoke pass and
still refuses inference.
``--execute-c0-c1`` is refused.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from cesvima_common import fail, require_cuda_family, require_under_scratch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3 invented smoke. Gold inference stays closed.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-c0-c1", action="store_true")
    parser.add_argument("--execute-c0-c1", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    selected = [args.dry_run, args.execute, args.prepare_c0_c1, args.execute_c0_c1, args.worker]
    if sum(bool(item) for item in selected) > 1:
        fail("choose one of --dry-run, --execute, --prepare-c0-c1, --execute-c0-c1, or --worker")
    if args.worker:
        from re_te_system.controlled_experiment.qwen3 import serve_invented_smoke_worker

        serve_invented_smoke_worker()
        return 0
    if args.execute_c0_c1:
        fail(
            "C0/C1 Gold inference is not authorized. "
            "The phase lock stays closed, the index SHA is not an approved pin, "
            "and a smoke pass does not open the 36-document run."
        )
    if args.prepare_c0_c1:
        return _prepare_c0_c1(args.smoke_report, args.output_dir)
    if args.execute:
        if args.output_dir is None:
            fail("--execute requires --output-dir")
        return _execute(args.output_dir)
    return _dry_run()


def _dry_run() -> int:
    from re_te_system.controlled_experiment.qwen3 import (
        QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
        invented_smoke_authorization,
        invented_smoke_generation_contract,
        load_qwen3_tokenizer,
        paired_c0_c1_scientific_plan,
        prepare_invented_smoke_requests,
        readiness,
        tokenizer_assets_verified,
    )

    report = readiness()
    blockers = [
        "checkpoint_not_verified_on_this_process",
        "gpu_not_verified_on_this_process",
        "gold_inference_not_authorized",
        "index_sha256_not_observed",
        "scientific_execution_not_authorized",
    ]
    prompts: list[dict[str, object]] = []
    if tokenizer_assets_verified():
        requests = prepare_invented_smoke_requests(load_qwen3_tokenizer())
        prompts = [
            {
                "condition": item["condition"],
                "document_id": item["document_id"],
                "fits": item["fits"],
                "rendered_prompt_sha256": item["rendered_prompt_sha256"],
                "text_sha256": item["text_sha256"],
            }
            for item in requests
        ]
        c0 = {item["document_id"]: item["rendered_prompt_sha256"] for item in prompts if item["condition"] == "C0"}
        c1 = {item["document_id"]: item["rendered_prompt_sha256"] for item in prompts if item["condition"] == "C1"}
        if c0 == c1 or not prompts or any(item["fits"] is not True for item in prompts):
            blockers.append("invented_prompt_preflight_failed")
    else:
        blockers.append("tokenizer_assets_not_local")
    plan = paired_c0_c1_scientific_plan()
    if plan["authorization"]["execution_permitted"] is not False:
        blockers.append("scientific_plan_opened_execution")
    coherent = "invented_prompt_preflight_failed" not in blockers and "tokenizer_assets_not_local" not in blockers
    if QWEN3_WEIGHT_EXECUTION_PHASE_LOCK is not True:
        coherent = False
        blockers.append("phase_lock_open")
    payload = {
        "authorization": {
            "configuration_id": invented_smoke_authorization().configuration_id,
            "input_scope": invented_smoke_authorization().input_scope,
        },
        "blockers": blockers,
        "c0_c1_scientific_execution": "NOT_READY",
        "commands": _commands(),
        "decoding": {
            "do_sample": invented_smoke_generation_contract().do_sample,
            "max_new_tokens": invented_smoke_generation_contract().max_new_tokens,
            "temperature": invented_smoke_generation_contract().temperature,
        },
        "execute_on_this_machine": False,
        "gold_inference_authorized": False,
        "invented_prompts": prompts,
        "invented_smoke_job_file": "READY_FOR_HUMAN_REVIEW",
        "invented_smoke_on_this_machine": "NOT_READY",
        "local_contract_coherent": coherent,
        "phase_lock": QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
        "plan_id": plan["plan_id"],
        "plan_execution_permitted": plan["authorization"]["execution_permitted"],
        "readiness": report,
        "weights_executed": False,
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if not coherent:
        return 1
    return 0


def _execute(output_dir: Path) -> int:
    from re_te_system.controlled_experiment.persistence import atomic_write_json, atomic_write_jsonl
    from re_te_system.controlled_experiment.qwen3 import (
        QWEN3_INVENTED_SMOKE_RESULT_ROLE,
        QWEN3_MODEL_ID,
        QWEN3_OBSERVED_DEFAULT_REVISION,
        QWEN3_SMOKE_GENERATION_TIMEOUT_SECONDS,
        QWEN3_SMOKE_LOAD_TIMEOUT_SECONDS,
        QWEN3_TOKENIZER_ASSETS,
        QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
        PersistentJsonWorker,
        claim_fresh_smoke_directory,
        controlled_continuation,
        invented_smoke_technical_pass,
        load_qwen3_tokenizer,
        prepare_invented_smoke_requests,
        verify_qwen3_checkpoint,
    )
    from hf_snapshot import snapshot

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    gpu_name = require_cuda_family("l40s")
    destination = require_under_scratch(output_dir)
    try:
        checkpoint = snapshot(QWEN3_MODEL_ID, QWEN3_OBSERVED_DEFAULT_REVISION, local_files_only=True)
    except SystemExit:
        raise
    except Exception as exc:
        fail(
            "pinned Qwen3 snapshot is not in the local cache. "
            "This smoke does not download weights. Cache "
            f"{QWEN3_MODEL_ID}@{QWEN3_OBSERVED_DEFAULT_REVISION} under SCRATCH and rerun. {exc}"
        )
    verification = verify_qwen3_checkpoint(checkpoint)
    claim_fresh_smoke_directory(destination)
    status_path = destination / "invented_smoke_status.json"
    report_path = destination / "invented_smoke_report.json"
    atomic_write_json(
        status_path,
        {"recovery": "rerun_requires_a_new_directory", "state": "running"},
    )
    atomic_write_json(destination / "checkpoint_inspection.json", dict(verification))
    worker = None
    try:
        tokenizer_dir = _stage_tokenizer(checkpoint, destination / "tokenizer-stage", QWEN3_TOKENIZER_ASSETS)
        tokenizer = load_qwen3_tokenizer(tokenizer_dir)
        requests = prepare_invented_smoke_requests(tokenizer)
        if any(item["fits"] is not True for item in requests):
            raise RuntimeError("invented smoke preflight does not fit")
        worker = PersistentJsonWorker([sys.executable, str(Path(__file__).resolve()), "--worker"])
        loaded = worker.request(
            {"checkpoint_dir": str(checkpoint), "op": "load", "tokenizer_dir": str(tokenizer_dir)},
            QWEN3_SMOKE_LOAD_TIMEOUT_SECONDS,
        )
        attempts = []
        for request in requests:
            generated = worker.request(
                {
                    "condition": request["condition"],
                    "document_id": request["document_id"],
                    "input_ids": request["input_ids"],
                    "model_generate_kwargs": request["model_generate_kwargs"],
                    "op": "generate",
                },
                QWEN3_SMOKE_GENERATION_TIMEOUT_SECONDS,
            )
            trace = controlled_continuation(tokenizer, request["input_ids"], generated["sequence_ids"])
            new_tokens = len(trace["continuation_ids"])
            attempts.append(
                {
                    "condition": request["condition"],
                    "document_id": request["document_id"],
                    "fits": True,
                    "load_count": generated.get("load_count"),
                    "new_tokens": new_tokens,
                    "rendered_prompt_sha256": request["rendered_prompt_sha256"],
                    "termination": generated.get("termination"),
                    "text_sha256": request["text_sha256"],
                    "within_budget": new_tokens <= int(request["model_generate_kwargs"]["max_new_tokens"]),
                }
            )
        technical = invented_smoke_technical_pass(attempts, int(loaded.get("load_count") or 0))
        report = _smoke_report(
            attempts=attempts,
            gpu_name=gpu_name,
            load_count=int(loaded.get("load_count") or 0),
            technical_pass=technical,
            verification=verification,
        )
        atomic_write_jsonl(destination / "attempts.jsonl", attempts)
        atomic_write_json(report_path, report)
        atomic_write_json(
            status_path,
            {
                "recovery": "rerun_requires_a_new_directory",
                "state": "pass" if technical else "fail",
            },
        )
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        if report["gold_inference_authorized"] or report["real_execution_authorized"]:
            return 1
        if QWEN3_WEIGHT_EXECUTION_PHASE_LOCK is not True:
            return 1
        if report["result_role"] != QWEN3_INVENTED_SMOKE_RESULT_ROLE:
            return 1
        return 0 if technical else 1
    except Exception as exc:
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "gold_inference_authorized": False,
            "real_execution_authorized": False,
            "technical_pass": False,
        }
        atomic_write_json(report_path, failure)
        atomic_write_json(
            status_path,
            {"recovery": "rerun_requires_a_new_directory", "state": "fail"},
        )
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if worker is not None:
            worker.close()


def _prepare_c0_c1(smoke_report: Path | None, output_dir: Path | None) -> int:
    from re_te_system.controlled_experiment.persistence import atomic_write_json
    from re_te_system.controlled_experiment.qwen3 import (
        QWEN3_INVENTED_SMOKE_RESULT_ROLE,
        QWEN3_SMOKE_INPUT_SCOPE,
        paired_c0_c1_scientific_plan,
    )

    if smoke_report is None or output_dir is None:
        fail("--prepare-c0-c1 requires --smoke-report and --output-dir")
    if not smoke_report.is_file():
        fail("smoke report is missing")
    observed = json.loads(smoke_report.read_text(encoding="utf-8"))
    required = (
        observed.get("technical_pass") is True
        and observed.get("weights_executed") is True
        and observed.get("gold_inference_authorized") is False
        and observed.get("scientific_result") is False
        and observed.get("real_execution_authorized") is False
        and observed.get("input_scope") == QWEN3_SMOKE_INPUT_SCOPE
        and observed.get("result_role") == QWEN3_INVENTED_SMOKE_RESULT_ROLE
        and observed.get("index_sha256_approved") is False
        and isinstance(observed.get("index_sha256"), str)
        and observed.get("load_count") == 1
    )
    if not required:
        fail("smoke report is not a technical pass with Gold still closed")
    destination = output_dir
    if os.environ.get("SCRATCH_ROOT"):
        destination = require_under_scratch(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plan = paired_c0_c1_scientific_plan()
    plan["observed_smoke"] = {
        "index_sha256": observed["index_sha256"],
        "index_sha256_approved": False,
        "technical_pass": True,
    }
    plan["authorization"]["execution_permitted"] = False
    plan["authorization"]["gold_inference_authorized"] = False
    atomic_write_json(destination / "c0_c1_paired_plan.json", plan)
    atomic_write_json(
        destination / "NOT_AUTHORIZED",
        {
            "execution_permitted": False,
            "gold_inference_authorized": False,
            "reason": "smoke pass does not authorize the 36-document Gold run",
        },
    )
    payload = {
        "c0_c1_scientific_execution": "NOT_READY",
        "execution_permitted": False,
        "gold_inference_authorized": False,
        "plan_id": plan["plan_id"],
        "wrote": str(destination / "c0_c1_paired_plan.json"),
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _smoke_report(
    *,
    attempts: list[dict[str, object]],
    gpu_name: str,
    load_count: int,
    technical_pass: bool,
    verification: dict[str, object],
) -> dict[str, object]:
    from re_te_system.controlled_experiment.qwen3 import (
        QWEN3_INVENTED_SMOKE_RESULT_ROLE,
        QWEN3_OBSERVED_DEFAULT_REVISION,
        QWEN3_PROPOSED_SMOKE_ID,
        QWEN3_SMOKE_INPUT_SCOPE,
        QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
        invented_smoke_generation_contract,
    )

    contract = invented_smoke_generation_contract()
    return {
        "attempts": attempts,
        "configuration_id": QWEN3_PROPOSED_SMOKE_ID,
        "device_policy": contract.device_policy,
        "do_sample": contract.do_sample,
        "dtype_policy": contract.dtype_policy,
        "gold_inference_authorized": False,
        "gpu_name": gpu_name,
        "index_sha256": verification.get("index_sha256"),
        "index_sha256_approved": False,
        "input_scope": QWEN3_SMOKE_INPUT_SCOPE,
        "load_count": load_count,
        "max_new_tokens": contract.max_new_tokens,
        "phase_lock": QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
        "real_execution_authorized": False,
        "result_role": QWEN3_INVENTED_SMOKE_RESULT_ROLE,
        "revision": QWEN3_OBSERVED_DEFAULT_REVISION,
        "scientific_result": False,
        "seed": contract.seed,
        "technical_pass": technical_pass,
        "temperature": None,
        "weights_executed": True,
    }


def _stage_tokenizer(checkpoint: Path, stage: Path, names: dict[str, object]) -> Path:
    stage.mkdir(parents=True, exist_ok=False)
    for name in names:
        source = checkpoint / name
        if not source.is_file():
            fail(f"checkpoint is missing tokenizer file {name}")
        (stage / name).write_bytes(source.read_bytes())
    return stage


def _commands() -> dict[str, str]:
    return {
        "dry_run": "python scripts/cesvima/j21_qwen3_invented_smoke.py --dry-run",
        "execute_c0_c1": "refused",
        "prepare_c0_c1_after_smoke_pass": (
            "python scripts/cesvima/j21_qwen3_invented_smoke.py --prepare-c0-c1 "
            "--smoke-report SMOKE_REPORT --output-dir OUTPUT"
        ),
        "submit_smoke_not_from_cursor": (
            "sbatch /home/y064/SCRATCH/y064644/paper3/isolated/qwen3-j21/"
            "re_te_system/jobs/cesvima/j21_qwen3_invented_smoke.sbatch"
        ),
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
