"""J00: verify the existing inference venv. Never creates a venv."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

from cesvima_common import fail


EXPECTED_PYTHON = "3.12.3"
EXPECTED_TORCH = "2.5.1+cu121"
EXPECTED_TORCH_CUDA = "12.1"
EXPECTED_TRANSFORMERS = "4.57.6"
EXPECTED_SENTENCEPIECE = "0.2.2"
CANONICAL_VENV_NAME = "re_te_system_gpu_312"


def main() -> int:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    executable = Path(sys.executable).resolve()
    print(f"python {version} executable {executable}", flush=True)
    if version != EXPECTED_PYTHON:
        fail(f"python {version} != {EXPECTED_PYTHON}")
    infer_venv = Path(os.environ["INFER_VENV"]).resolve()
    if infer_venv.name != CANONICAL_VENV_NAME:
        fail(f"INFER_VENV {infer_venv} is not {CANONICAL_VENV_NAME}")
    if infer_venv not in executable.parents:
        fail(f"interpreter {executable} is outside {infer_venv}")
    if executable.as_posix().endswith("/venvs/re_te_system_312/bin/python") or (
        "venvs/re_te_system_312/" in executable.as_posix()
    ):
        fail("re_te_system_312 is retained but is not the GPU runtime")

    try:
        import torch
    except ImportError as exc:
        fail(f"torch is not importable: {exc}")
    if torch.__version__ != EXPECTED_TORCH:
        fail(f"torch {torch.__version__} != {EXPECTED_TORCH}")
    if str(torch.version.cuda) != EXPECTED_TORCH_CUDA:
        fail(f"torch CUDA {torch.version.cuda} != {EXPECTED_TORCH_CUDA}")

    try:
        import transformers
    except ImportError as exc:
        fail(f"transformers is not importable: {exc}")
    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        fail(f"transformers {transformers.__version__} != {EXPECTED_TRANSFORMERS}")

    try:
        import sentencepiece
    except ImportError as exc:
        fail(f"sentencepiece is not importable: {exc}")
    if sentencepiece.__version__ != EXPECTED_SENTENCEPIECE:
        fail(f"sentencepiece {sentencepiece.__version__} != {EXPECTED_SENTENCEPIECE}")

    try:
        import re_te_system
    except ImportError as exc:
        fail(f"re_te_system is not importable: {exc}")
    imported_from = Path(re_te_system.__file__).resolve().parent
    expected_package = (Path(os.environ["SYSTEM_ROOT"]).resolve() / "src" / "re_te_system")
    if imported_from != expected_package:
        fail(f"re_te_system imported from {imported_from}, expected {expected_package}")

    try:
        distribution = importlib.metadata.distribution("re-te-system")
    except importlib.metadata.PackageNotFoundError as exc:
        fail(f"re-te-system is not installed in this interpreter: {exc}")
    direct = distribution.read_text("direct_url.json")
    if not direct:
        fail("re-te-system has no direct_url.json; expected an editable install")
    record = json.loads(direct)
    editable = bool(record.get("dir_info", {}).get("editable"))
    if not editable:
        fail(f"re-te-system is not an editable install: {record}")
    url = str(record.get("url", ""))
    system_root = Path(os.environ["SYSTEM_ROOT"]).resolve()
    if system_root.as_posix() not in url:
        fail(f"editable install url {url} does not point at {system_root}")

    print("PASS J00 infer venv", flush=True)
    print(
        json.dumps(
            {
                "editable_url": url,
                "python": version,
                "sentencepiece": sentencepiece.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
