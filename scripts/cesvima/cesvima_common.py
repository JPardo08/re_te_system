"""Shared helpers for CESVIMA train-first jobs. No model downloads happen here."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


SHA40 = re.compile(r"[0-9a-f]{40}")


def fail(message: str, code: int = 1) -> None:
    print(f"FAIL: {message}", flush=True)
    raise SystemExit(code)


def scratch_root() -> Path:
    raw = os.environ.get("SCRATCH_ROOT")
    if not raw:
        fail("SCRATCH_ROOT is not set")
    return Path(raw).resolve()


def require_under_scratch(path: Path) -> Path:
    resolved = path.resolve()
    root = scratch_root()
    if resolved != root and root not in resolved.parents:
        fail(f"refusing to write outside SCRATCH_ROOT: {resolved}")
    return resolved


def write_json(path: Path, payload: dict) -> None:
    target = require_under_scratch(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {target}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: str, label: str = "revision") -> str:
    if not isinstance(value, str) or SHA40.fullmatch(value) is None:
        fail(f"{label} is not a 40-character lowercase SHA: {value!r}")
    return value


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def require_cuda_family(expected: str) -> str:
    try:
        import torch
    except ImportError as exc:
        fail(f"torch is not importable: {exc}")
    if not torch.cuda.is_available():
        fail("CUDA is not available")
    name = torch.cuda.get_device_name(0)
    lowered = name.lower()
    if "a100" in lowered:
        fail(f"A100 is forbidden; visible device is {name}")
    if expected.lower() not in lowered:
        fail(f"expected CUDA device family {expected}; visible device is {name}")
    print(f"PASS cuda-device {name}", flush=True)
    return name
