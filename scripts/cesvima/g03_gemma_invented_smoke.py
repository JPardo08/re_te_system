"""G03 invented C0/C1 smoke. A technical pass does not authorize G04."""

from __future__ import annotations

import argparse
from pathlib import Path

from cesvima_common import fail, write_json
from re_te_system.controlled_experiment.gemma4 import (
    CANDIDATE_REVISION,
    G03_ENV,
    GemmaDuplicatePublication,
    GemmaGoldAccessRefused,
    GemmaGoldNotAuthorized,
    GemmaPreflightNotAuthorized,
    GemmaRevisionNotApproved,
    assert_revision_approved,
    grant_enabled,
    persist_g03,
    smoke_requests,
)


def main(argv: list[str] | None = None, loader=None) -> int:
    parser = argparse.ArgumentParser(description="G03 Gemma invented smoke. Gold stays closed.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
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
        if loader is None:
            raise GemmaPreflightNotAuthorized("G03 has no bound loader")
        persist_g03(args.output_dir, loader())
    except (
        GemmaDuplicatePublication,
        GemmaGoldAccessRefused,
        GemmaGoldNotAuthorized,
        GemmaPreflightNotAuthorized,
        GemmaRevisionNotApproved,
    ) as exc:
        fail(str(exc), code=2)
    print("PASS G03", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
