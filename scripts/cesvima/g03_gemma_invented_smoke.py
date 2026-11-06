"""G03 invented C0/C1 smoke.

One process loads the accepted local checkpoint once and generates four
invented turns. A technical pass does not authorize G04 or a scientific score.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from cesvima_common import fail, require_cuda_family, write_json
from re_te_system.controlled_experiment.gemma4 import (
    CANDIDATE_REVISION,
    DECLARED_NATIVE_CONTEXT_LENGTH,
    G03_ENV,
    PREFLIGHT_MAX_NEW_TOKENS,
    PROPOSED_DTYPE,
    GemmaAllocationError,
    GemmaChannelError,
    GemmaContextBudgetError,
    GemmaDuplicatePublication,
    GemmaGoldAccessRefused,
    GemmaGoldNotAuthorized,
    GemmaPreflightNotAuthorized,
    GemmaRevisionNotApproved,
    GemmaSilentTruncationRefused,
    GemmaWeightFormatError,
    GemmaWorkerTimeout,
    assert_revision_approved,
    grant_enabled,
    persist_g03,
    read_slurm_job_record,
    run_invented_smoke,
    smoke_requests,
)


def main(
    argv: list[str] | None = None,
    loader=None,
    session_factory=None,
    allocation_text: str | None = None,
    confirm_device=None,
) -> int:
    parser = argparse.ArgumentParser(description="G03 Gemma invented smoke. Gold stays closed.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--revision", default=CANDIDATE_REVISION)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--documents", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.documents is not None:
            raise GemmaGoldAccessRefused("G03 does not read a document corpus")
        if not args.execute:
            if args.output is None:
                fail("G03 plan requires --output")
            write_json(
                args.output,
                {
                    "executed": False,
                    "g04_authorized": False,
                    "requests": smoke_requests(),
                    "scientific_result": False,
                    "stage": "G03",
                    "weights_loaded": False,
                },
            )
            fail("G03 execute was not requested; no weights were loaded", code=2)
        if args.output_dir is None:
            fail("G03 execute requires --output-dir")
        if not grant_enabled(G03_ENV):
            raise GemmaPreflightNotAuthorized("G03 grant is missing")
        assert_revision_approved(args.revision)
        if loader is not None:
            persist_g03(args.output_dir, loader())
        else:
            if args.snapshot_dir is None:
                fail("G03 execute requires --snapshot-dir")
            allocation = (
                allocation_text
                if allocation_text is not None
                else read_slurm_job_record(os.environ.get("SLURM_JOB_ID", ""), _run_scontrol)
            )
            result = run_invented_smoke(
                revision=args.revision,
                snapshot_dir=args.snapshot_dir,
                allocation_text=allocation,
                session_factory=session_factory or (lambda: _load_local_session(args.snapshot_dir)),
                confirm_device=confirm_device or _confirm_l40s,
            )
            persist_g03(args.output_dir, result)
    except (
        GemmaAllocationError,
        GemmaChannelError,
        GemmaContextBudgetError,
        GemmaDuplicatePublication,
        GemmaGoldAccessRefused,
        GemmaGoldNotAuthorized,
        GemmaPreflightNotAuthorized,
        GemmaRevisionNotApproved,
        GemmaSilentTruncationRefused,
        GemmaWeightFormatError,
        GemmaWorkerTimeout,
    ) as exc:
        fail(str(exc), code=2)
    print("PASS G03", flush=True)
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


def _confirm_l40s() -> str:
    return require_cuda_family("l40s")


def _load_local_session(snapshot: Path):
    """Load the accepted snapshot once. No CPU fallback and no second copy."""
    import torch
    from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration

    if PROPOSED_DTYPE != "bfloat16":
        raise GemmaPreflightNotAuthorized("G03 only loads BF16 weights")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise GemmaAllocationError("refusing a CPU fallback or a multi-GPU load")
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    if type(processor).__name__ != "Gemma4UnifiedProcessor":
        raise GemmaPreflightNotAuthorized(
            f"processor is {type(processor).__name__}, not Gemma4UnifiedProcessor"
        )
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.to("cuda:0")
    model.eval()
    return _LocalSession(processor, model)


class _LocalSession:
    def __init__(self, processor: object, model: object) -> None:
        self.processor = processor
        self.model = model
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None)

    def apply_chat_template(self, messages: list, **kwargs: object) -> str:
        return self.processor.apply_chat_template(messages, **kwargs)

    def __call__(self, *, text: str, **kwargs: object):
        return self.processor(text=text, **kwargs)

    def generate(self, encoded: dict, *, do_sample: bool, max_new_tokens: int) -> dict[str, object]:
        if do_sample is not False or max_new_tokens != PREFLIGHT_MAX_NEW_TOKENS:
            raise GemmaPreflightNotAuthorized("G03 refused a decoding policy other than the G02 contract")
        import torch

        tensors = {key: value.to("cuda:0") for key, value in encoded.items() if hasattr(value, "to")}
        input_len = int(tensors["input_ids"].shape[-1])
        if input_len + max_new_tokens > DECLARED_NATIVE_CONTEXT_LENGTH:
            raise GemmaContextBudgetError("invented smoke prompt does not fit the declared context")
        with torch.inference_mode():
            generated = self.model.generate(
                **tensors,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        new_tokens = int(generated.shape[-1]) - input_len
        raw = self.processor.decode(generated[0][input_len:], skip_special_tokens=False)
        return {"raw": raw, "truncated": new_tokens >= max_new_tokens}


if __name__ == "__main__":
    raise SystemExit(main())
