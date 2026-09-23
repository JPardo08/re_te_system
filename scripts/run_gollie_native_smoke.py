"""Prepare the official GoLLIE-7B native RE smoke command. Does not download weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the future CUDA/FlashAttention command for HiTZ/GoLLIE-7B. "
            "This script does not download or run weights."
        )
    )
    parser.add_argument("--documents", required=True)
    parser.add_argument("--gollie-schema-file", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [
        sys.executable,
        "-m",
        "re_te_system.cli",
        "run",
        "--extractor",
        "gollie",
        "--documents",
        str(Path(args.documents)),
        "--gollie-schema-file",
        str(Path(args.gollie_schema_file)),
        "--output-root",
        str(Path(args.output_root)),
        "--input-language",
        "en",
        "--local-files-only",
        "--gollie-max-new-tokens",
        "128",
    ]
    print(
        "REAL_GOLLIE_NATIVE_SMOKE = BLOCKED_GPU unless an NVIDIA CUDA GPU "
        "and official FlashAttention are available."
    )
    print("Prepared command (not executed):")
    print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
