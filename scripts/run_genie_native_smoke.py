"""Prepare official GenIE native smoke commands. Does not download weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare unconstrained/small/large GenIE smoke commands. "
            "This script never downloads genie_r.ckpt."
        )
    )
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--constraint-profile",
        choices=("unconstrained", "small", "large"),
        default="unconstrained",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--authorize-checkpoint-load", action="store_true")
    parser.add_argument("--execute-mock", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mock_command = [
        sys.executable,
        "-m",
        "re_te_system.cli",
        "run",
        "--extractor",
        "genie-mock",
        "--documents",
        str(Path(args.documents)),
        "--output-root",
        str(Path(args.output_root)),
        "--input-language",
        "en",
        "--genie-constraint-profile",
        args.constraint_profile,
    ]
    real_command = [
        sys.executable,
        "-m",
        "re_te_system.cli",
        "run",
        "--extractor",
        "genie",
        "--documents",
        str(Path(args.documents)),
        "--output-root",
        str(Path(args.output_root)),
        "--input-language",
        "en",
        "--genie-constraint-profile",
        args.constraint_profile,
        "--local-files-only",
    ]
    if args.checkpoint_path:
        real_command.extend(["--genie-checkpoint-path", str(Path(args.checkpoint_path))])
    if args.authorize_checkpoint_load:
        real_command.append("--authorize-checkpoint-load")
    print("REAL_GENIE_NATIVE_SMOKE = BLOCKED_RESOURCE")
    print("P0 does not download genie_r.ckpt and has not verified Lightning loading.")
    print("Prepared mock command:")
    print(" ".join(mock_command))
    print("Prepared real command (not executed; loader unverified):")
    print(" ".join(real_command))
    if args.execute_mock:
        from re_te_system.cli import main as cli_main

        return cli_main(mock_command[3:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
