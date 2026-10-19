"""Atomic campaign files.

A reader only opens the final name. The temporary file is created in the same
directory and becomes the final name through ``os.replace``. A failed write
leaves the previous complete file in place, or no file if this is the first
attempt, and the exception propagates. A partial temporary file is not a
valid campaign record.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from re_te_system.controlled_experiment.contracts import canonical_json


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, canonical_json(payload) + "\n")


def atomic_write_jsonl(path: Path, rows: list[object]) -> None:
    atomic_write_text(path, "".join(canonical_json(row) + "\n" for row in rows))
