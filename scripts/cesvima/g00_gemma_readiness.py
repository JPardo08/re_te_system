"""G00: Gemma readiness. No download, no GPU reservation, no Gold read."""

from __future__ import annotations

import argparse
from pathlib import Path

from cesvima_common import fail, write_json
from re_te_system.controlled_experiment.gemma4 import (
    G00_ENV,
    GemmaGoldAccessRefused,
    grant_enabled,
    readiness_report,
    refuse_gold_reference,
    technical_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G00 Gemma runtime readiness. Does not download weights.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=Path)
    args = parser.parse_args(argv)
    if not grant_enabled(G00_ENV):
        fail("G00 readiness is not authorized")
    try:
        refuse_gold_reference(args.documents)
    except GemmaGoldAccessRefused as exc:
        fail(str(exc))
    report = readiness_report()
    report["manifest"] = technical_manifest(report)
    write_json(args.output, report)
    if report["ready_for_gpu_reservation"] is not True:
        fail(
            "G00 runtime is not compatible: " + "; ".join(report["runtime_blockers"]),
            code=2,
        )
    if report["scientific_ready"] is not False:
        fail("G00 cannot certify scientific readiness")
    if report["authorizes_later_stages"] != {"G01": False, "G02": False, "G03": False, "G04": False}:
        fail("G00 must not authorize a later stage")
    print("PASS G00 runtime-compatibility-only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
