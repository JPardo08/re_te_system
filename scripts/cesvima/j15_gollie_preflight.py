"""J15: GoLLIE preflight without downloading 7B weights."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path

from cesvima_common import fail, require_cuda_family, write_json


GOLLIE_REPO = "HiTZ/GoLLIE-7B"
GOLLIE_REVISION = "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9"
MODELING_FILE = "modeling_flash_llama.py"
FREE_BYTES = 40 * 1024 * 1024 * 1024


def _black_ok() -> str:
    version = importlib.metadata.version("black")
    parts = tuple(int(piece) for piece in version.split(".")[:2])
    if parts < (24, 8) or parts >= (25, 0):
        fail(f"black {version} is outside >=24.8,<25")
    return version


def _flash_kernel() -> None:
    import torch
    from flash_attn import flash_attn_func

    query = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    output = flash_attn_func(query, query, query, dropout_p=0.0, causal=True)
    if not torch.isfinite(output).all():
        fail("FlashAttention kernel returned non-finite values")


def _bfloat16_matmul() -> None:
    import torch

    if not torch.cuda.is_bf16_supported():
        fail("CUDA device does not support bfloat16")
    values = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    product = values @ values
    if not torch.isfinite(product).all():
        fail("bfloat16 matmul returned non-finite values")


def _import_modeling() -> str:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=GOLLIE_REPO,
            filename=MODELING_FILE,
            revision=GOLLIE_REVISION,
        )
    )
    if path.stat().st_size > 5_000_000:
        fail(f"{MODELING_FILE} is unexpectedly large ({path.stat().st_size} bytes)")
    if path.name != MODELING_FILE:
        fail(f"downloaded {path.name} instead of {MODELING_FILE}")
    spec = importlib.util.spec_from_file_location("gollie_modeling_flash_llama", path)
    if spec is None or spec.loader is None:
        fail(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(path)


def main() -> int:
    import torch

    device_name = require_cuda_family("L40S")
    cuda = str(torch.version.cuda)
    if not cuda.startswith("12.1"):
        fail(f"torch CUDA {cuda} is not the cu121 build")
    if torch.__version__ != "2.5.1+cu121":
        fail(f"torch {torch.__version__} != 2.5.1+cu121")
    _bfloat16_matmul()
    black_version = _black_ok()
    _flash_kernel()
    modeling_path = _import_modeling()
    usage = shutil.disk_usage(os.environ["SCRATCH_ROOT"])
    if usage.free < FREE_BYTES:
        fail(f"SCRATCH free space {usage.free} bytes is below {FREE_BYTES}")
    report = Path(os.environ["P3_RUNS_ROOT"]) / "gollie" / "preflight_report.json"
    write_json(
        report,
        {
            "black": black_version,
            "cuda_device_name": device_name,
            "free_bytes": usage.free,
            "modeling_file": modeling_path,
            "status": "PASS",
            "torch": torch.__version__,
            "torch_cuda": cuda,
            "weights_downloaded": False,
        },
    )
    print("PASS J15 GoLLIE preflight", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
