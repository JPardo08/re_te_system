"""G01: explicit Gemma acquisition. The catalog revision is not an approval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cesvima_common import fail, write_json
from re_te_system.controlled_experiment.gemma4 import (
    GemmaAcquireNotAuthorized,
    GemmaDuplicatePublication,
    GemmaGoldAccessRefused,
    GemmaModelIdentityError,
    GemmaRevisionNotApproved,
    acquire_checkpoint,
    acquisition_plan,
    refuse_gold_reference,
)


def main(argv: list[str] | None = None, downloader=None) -> int:
    parser = argparse.ArgumentParser(description="G01 acquires one approved Gemma revision. Tests must not download.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--documents", type=Path)
    args = parser.parse_args(argv)
    try:
        refuse_gold_reference(args.documents)
        plan = acquisition_plan(args.model_id, args.revision)
        if plan["acquire_permitted"] is not True:
            write_json(args.report, plan)
            fail(plan["refusal"] or "acquisition is closed", code=2)
        published = acquire_checkpoint(
            args.model_id,
            args.revision,
            args.destination,
            downloader if downloader is not None else _download_approved_snapshot,
        )
    except (GemmaGoldAccessRefused, GemmaModelIdentityError, GemmaRevisionNotApproved) as exc:
        fail(str(exc), code=2)
    except GemmaAcquireNotAuthorized as exc:
        fail(str(exc), code=2)
    except GemmaDuplicatePublication as exc:
        fail(str(exc), code=2)
    write_json(args.report, published)
    print("PASS G01", flush=True)
    return 0


def _download_approved_snapshot(model_id: str, revision: str) -> Path:
    """Bound only after acquisition_plan has allowed this revision. GPU jobs do not call it."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model_id, revision=revision, local_files_only=False)
    return Path(path)


if __name__ == "__main__":
    raise SystemExit(main())
