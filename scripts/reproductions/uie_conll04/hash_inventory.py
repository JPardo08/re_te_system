#!/usr/bin/env python3
"""Create a deterministic SHA-256 inventory for generated reproduction trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--path", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files: dict[str, dict[str, int | str]] = {}
    for relative in args.path:
        target = args.root / relative
        if not target.exists():
            raise FileNotFoundError(target)
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            key = path.relative_to(args.root).as_posix()
            files[key] = {
                "sha256": sha256(path),
                "size": path.stat().st_size,
            }

    payload = {"files": files}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["combined_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
