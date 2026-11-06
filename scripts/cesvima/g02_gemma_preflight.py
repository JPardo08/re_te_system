"""G02: L40S preflight.

--execute loads only after scontrol AllocTRES reports exactly one L40S and
the visible CUDA device matches that family. Grants, offline mode, and the
generation contract are unchanged.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from cesvima_common import fail, require_cuda_family, write_json
from re_te_system.controlled_experiment.gemma4 import (
    DECLARED_NATIVE_CONTEXT_LENGTH,
    INVENTED_PREFLIGHT_TEXT,
    MODEL_ID,
    PROPOSED_DTYPE,
    GemmaAllocationError,
    GemmaChannelError,
    GemmaContextBudgetError,
    GemmaGoldAccessRefused,
    GemmaPreflightNotAuthorized,
    GemmaRevisionNotApproved,
    GemmaWeightFormatError,
    encode_serialized_turn,
    execute_bounded_preflight,
    preflight_decoding_contract,
    preflight_plan,
    read_slurm_job_record,
    refuse_gold_reference,
    serialize_text_turn,
    transport_and_parse,
)


def main(argv: list[str] | None = None, loader=None) -> int:
    parser = argparse.ArgumentParser(description="G02 Gemma GPU preflight. Gold stays closed.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--documents", type=Path)
    args = parser.parse_args(argv)
    try:
        refuse_gold_reference(args.documents)
    except GemmaGoldAccessRefused as exc:
        fail(str(exc))
    if not args.execute:
        report = preflight_plan()
        report["executed"] = False
        write_json(args.output, report)
        fail("G02 execute was not requested; no weights were loaded", code=2)
    if args.snapshot_dir is None:
        fail("G02 --execute requires --snapshot-dir")
    os.environ["GEMMA_SNAPSHOT_DIR"] = str(args.snapshot_dir)
    try:
        allocation = read_slurm_job_record(os.environ.get("SLURM_JOB_ID", ""), _run_scontrol)
        report = execute_bounded_preflight(
            revision=args.revision,
            allocation_text=allocation,
            loader=loader if loader is not None else _load_local_safetensors,
            snapshot_dir=args.snapshot_dir,
        )
    except (
        GemmaAllocationError,
        GemmaPreflightNotAuthorized,
        GemmaRevisionNotApproved,
        GemmaWeightFormatError,
    ) as exc:
        fail(str(exc), code=2)
    report["executed"] = True
    report["generation_request"] = generation_request()
    write_json(args.output, report)
    print("PASS G02", flush=True)
    return 0


def _run_scontrol(argv: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
    }


def generation_request() -> dict[str, object]:
    contract = preflight_decoding_contract()
    return {
        "do_sample": contract.do_sample,
        "fixture": "invented_sentence",
        "max_new_tokens": contract.max_new_tokens,
        "quantization": None,
        "scientific_decode": False,
    }


def _load_local_safetensors(**kwargs: object) -> dict[str, object]:
    if kwargs.get("local_files_only") is not True or kwargs.get("quantization") is not None:
        raise GemmaPreflightNotAuthorized("refusing a networked or quantized load")
    if kwargs.get("model_id") != MODEL_ID or kwargs.get("dtype") != PROPOSED_DTYPE:
        raise GemmaPreflightNotAuthorized("loader identity does not match the instruction-tuned contract")
    require_cuda_family("l40s")
    import torch
    from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration

    snapshot = os.environ["GEMMA_SNAPSHOT_DIR"]
    contract = preflight_decoding_contract()
    serialized = serialize_text_turn(INVENTED_PREFLIGHT_TEXT, enable_thinking=False)
    free_before, total = torch.cuda.mem_get_info()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    official = processor.apply_chat_template(
        [{"role": "user", "content": INVENTED_PREFLIGHT_TEXT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if official != serialized:
        raise GemmaChannelError(
            "the loaded processor template does not match the published 12B reconstruction"
        )
    encoded = encode_serialized_turn(
        processor,
        serialized,
        return_tensors="pt",
    )
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.to("cuda:0")
    model.eval()
    tensors = {key: value.to("cuda:0") for key, value in encoded.items() if hasattr(value, "to")}
    input_len = int(tensors["input_ids"].shape[-1])
    if input_len + int(contract.max_new_tokens) > DECLARED_NATIVE_CONTEXT_LENGTH:
        raise GemmaContextBudgetError("invented preflight does not fit the declared context")
    with torch.inference_mode():
        generated = model.generate(
            **tensors,
            do_sample=False,
            max_new_tokens=contract.max_new_tokens,
        )
    continuation = processor.decode(generated[0][input_len:], skip_special_tokens=False)
    parsed = transport_and_parse(continuation)
    free_after, _total_after = torch.cuda.mem_get_info()
    tokenizer = getattr(processor, "tokenizer", processor)
    return {
        "channel_status": parsed["channel"]["status"],
        "device": "cuda:0",
        "dtype": PROPOSED_DTYPE,
        "eos_token_id_observed_not_pinned": getattr(tokenizer, "eos_token_id", None),
        "free_bytes_after_load": int(free_after),
        "free_bytes_before_load": int(free_before),
        "gold_inference_authorized": False,
        "pad_token_id_observed_not_pinned": getattr(tokenizer, "pad_token_id", None),
        "processor_class": type(processor).__name__,
        "scientific_result": False,
        "template_matches_published_reconstruction": True,
        "total_bytes": int(total),
        "triple_count_not_a_score": len(parsed["triples"]),
        "weights_loaded": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
