"""J15: GoLLIE preflight without downloading 7B weights."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import urllib.request
from pathlib import Path

from cesvima_common import fail, require_cuda_family
from gollie_environment import (
    build_preflight_report,
    collect_runtime_abi,
    merge_abi_observations,
    probe_installed_environment,
    write_accepted_pointer,
    write_job_report,
)


GOLLIE_REPO = "HiTZ/GoLLIE-7B"
GOLLIE_REVISION = "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9"
MODELING_FILE = "modeling_flash_llama.py"
MODELING_BLOB_OID = "1330bc1ca5c6769f0880555350cfd33ea5ea4fbd"
MODELING_SIZE = 46523
MODELING_URL = (
    "https://huggingface.co/HiTZ/GoLLIE-7B/resolve/"
    "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9/modeling_flash_llama.py"
)
FREE_BYTES = 40 * 1024 * 1024 * 1024


class ModelingIdentityError(ValueError):
    """The local modeling module is not the pinned GoLLIE revision."""


def git_blob_oid(payload: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(payload) + payload).hexdigest()


def modeling_cache_path(cache: Path) -> Path:
    folder = "models--" + GOLLIE_REPO.replace("/", "--")
    return cache / folder / "snapshots" / GOLLIE_REVISION / MODELING_FILE


def audit_modeling_file(path: Path) -> None:
    """Check the pinned blob without importing or executing it."""
    if path.is_symlink() or not path.is_file():
        raise ModelingIdentityError(f"{MODELING_FILE} is absent at {path}")
    payload = path.read_bytes()
    if len(payload) != MODELING_SIZE:
        raise ModelingIdentityError(f"modeling file size {len(payload)} != {MODELING_SIZE}")
    oid = git_blob_oid(payload)
    if oid != MODELING_BLOB_OID:
        raise ModelingIdentityError(f"modeling git blob {oid} != {MODELING_BLOB_OID}")
    if b"\0" in payload:
        raise ModelingIdentityError("modeling file is not the text module")
    try:
        compile(payload.decode("utf-8"), MODELING_FILE, "exec")
    except (SyntaxError, UnicodeError) as exc:
        raise ModelingIdentityError(f"modeling file does not compile: {exc}") from exc


def ensure_modeling_file(cache: str | Path, *, authorized: bool, fetcher=None) -> str:
    """Place only modeling_flash_llama.py. Weights and the rest of the repo stay untouched."""
    path = modeling_cache_path(Path(cache))
    if path.is_symlink():
        raise ModelingIdentityError("refusing a symlinked modeling file")
    if path.is_file():
        audit_modeling_file(path)
        return "PRESENT"
    if authorized is not True:
        raise ModelingIdentityError("REFUSING_DOWNLOAD modeling file and weights")
    if fetcher is None:
        request = urllib.request.Request(MODELING_URL, headers={"User-Agent": "re_te_system-gollie-modeling"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    else:
        payload = fetcher(MODELING_URL)
    if not isinstance(payload, bytes):
        raise ModelingIdentityError("modeling download did not return bytes")
    temporary = path.with_name("." + path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(payload)
    try:
        if len(payload) != MODELING_SIZE or git_blob_oid(payload) != MODELING_BLOB_OID:
            raise ModelingIdentityError("downloaded modeling file does not match the pinned blob")
        try:
            compile(payload.decode("utf-8"), MODELING_FILE, "exec")
        except (SyntaxError, UnicodeError) as exc:
            raise ModelingIdentityError(f"modeling file does not compile: {exc}") from exc
        if path.exists():
            raise ModelingIdentityError("refusing to overwrite the modeling file")
        os.link(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    audit_modeling_file(path)
    return "FETCHED"


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


def _job_id() -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id or job_id in {".", ".."} or "/" in job_id or "\\" in job_id:
        fail("J15 requires a single-path-segment SLURM_JOB_ID")
    return job_id


def _device_family() -> str | None:
    """Name the visible GPU without launching FlashAttention."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    if "l40s" in name.lower():
        return "L40S"
    return name


def _observe_abi(assessment: dict[str, object]) -> dict[str, object]:
    raw = assessment.get("installed_extension_path")
    family = _device_family()
    if not isinstance(raw, str) or not raw:
        return merge_abi_observations(
            assessment,
            {
                "abi_observed_on_gpu_node": False,
                "dynamic_dependencies_resolved": None,
                "glibc_requirement_satisfied": None,
                "import_succeeded": None,
            },
        )
    return merge_abi_observations(
        assessment,
        collect_runtime_abi(Path(raw), device_family=family),
    )


def _kernel_observation() -> dict[str, object]:
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
    return {
        "black": black_version,
        "cuda_device_name": device_name,
        "free_bytes": usage.free,
        "modeling_file": modeling_path,
        "torch": torch.__version__,
        "torch_cuda": cuda,
    }


def main() -> int:
    job_id = _job_id()
    runs = Path(os.environ["P3_RUNS_ROOT"])
    node = os.environ.get("SLURMD_NODENAME") or None
    assessment = probe_installed_environment()
    if assessment["accredited_for_reuse"] is not True:
        report = build_preflight_report(
            assessment,
            job_id=job_id,
            node=node,
            kernel_finite=None,
        )
        write_job_report(runs, job_id, report)
        print(f"BLOCKED_BINARY_IDENTITY {assessment['binary_identity']}", flush=True)
        return 2
    assessment = _observe_abi(assessment)
    if assessment["abi_pass"] is not True:
        report = build_preflight_report(
            assessment,
            job_id=job_id,
            node=node,
            kernel_finite=None,
        )
        write_job_report(runs, job_id, report)
        print(f"BLOCKED_ABI abi_verdict={assessment['abi_verdict']}", flush=True)
        return 2
    observed = _kernel_observation()
    report = build_preflight_report(
        assessment,
        job_id=job_id,
        node=node,
        kernel_finite=True,
        kernel_device_family="L40S",
    )
    report.update(observed)
    write_job_report(runs, job_id, report)
    if report["preflight_pass"] is not True:
        print("KERNEL_OBSERVED J15 preflight_pass=false", flush=True)
        return 2
    write_accepted_pointer(runs, job_id)
    print("KERNEL_OBSERVED J15 preflight_pass=true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
