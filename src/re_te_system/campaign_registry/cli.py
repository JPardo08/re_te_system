"""Local validation command for the campaign registry.

The only subcommand reads JSON and prints a report. It does not submit jobs,
load weights, create environments, or start inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from re_te_system.campaign_registry.contracts import INVALID, STRUCTURALLY_VALID
from re_te_system.campaign_registry.loader import CatalogLoadError
from re_te_system.campaign_registry.report import render_report
from re_te_system.campaign_registry.validator import validate_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="re_te_system.campaign_registry.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--catalog-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command != "validate":
        parser.error("unknown command")
    try:
        report = validate_catalog(args.catalog_root)
    except CatalogLoadError as exc:
        sys.stdout.write(
            render_report(
                {
                    "errors": [{"code": "catalog_load_error", "detail": str(exc), "task_id": None}],
                    "executed_external_commands": [],
                    "overall_status": INVALID,
                    "readiness_status": INVALID,
                    "structural_status": INVALID,
                }
            )
        )
        return 1
    sys.stdout.write(render_report(report))
    if report["structural_status"] != STRUCTURALLY_VALID:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
