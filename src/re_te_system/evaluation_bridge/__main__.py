"""Command line for one finished campaign directory.

This is the bridge entry point. It does not run a model, schedule a job, or
choose an estimand. A report is written for both publishable and blocked
campaigns. The process exits 2 only when the report itself cannot be written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from re_te_system.evaluation_bridge.bridge import evaluate_campaign
from re_te_system.evaluation_bridge.contracts import BridgeError, BridgeRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="re_te_system.evaluation_bridge")
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--benchmark-manifest", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--registry-protocols", type=Path)
    parser.add_argument("--expected-document-id", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected = None if not args.expected_document_id else tuple(args.expected_document_id)
    request = BridgeRequest(
        campaign_dir=args.campaign_dir,
        gold_path=args.gold,
        output_dir=args.output_dir,
        documents_path=args.documents,
        benchmark_manifest_path=args.benchmark_manifest,
        protocol_path=args.protocol,
        registry_protocols_path=args.registry_protocols,
        expected_document_ids=expected,
    )
    try:
        report = evaluate_campaign(request)
    except BridgeError as exc:
        print(f"evaluation bridge error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "blockers": report["blockers"],
                "report": str(Path(args.output_dir) / "campaign_evaluation_report.json"),
                "scientific_result": report["publication_eligibility"]["scientific_result"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
