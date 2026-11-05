"""G04 campaign entrypoint. It records the refusal and does not read Gold."""

from __future__ import annotations

import argparse
from pathlib import Path

from cesvima_common import fail, write_json
from re_te_system.controlled_experiment.gemma4 import (
    GemmaGoldNotAuthorized,
    launch_g04,
    scientific_plan,
)


def main(argv: list[str] | None = None, loader=None) -> int:
    parser = argparse.ArgumentParser(description="G04 Gemma C0/C1 campaign. Execution is closed.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args(argv)
    if args.annotations is not None or (
        args.documents is not None and "annotations" in args.documents.name
    ):
        fail("G04 cannot take Gold annotations as generation input", code=2)
    plan = scientific_plan()
    evidence = {
        "frozen_decoding": {},
        "g02_report": {},
        "g03_report": {},
        "human_authorization": {},
        "output_safe": False,
        "revision_approved": False,
        "runtime_compatible": False,
        "snapshot_verified": False,
    }
    try:
        launch_g04(evidence, loader=loader)
    except GemmaGoldNotAuthorized as exc:
        write_json(
            args.output,
            {
                "documents_read": False,
                "executed": False,
                "gold_inference_authorized": False,
                "plan": plan,
                "refusal": str(exc),
                "stage": "G04",
            },
        )
        fail(str(exc), code=2)
    fail("G04 returned without a refusal", code=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
