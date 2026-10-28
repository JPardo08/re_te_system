"""Eager launch shim for the pinned FlashAttention RoPE kernel.

FlashAttention 2.8.3.post2 launches ``rotary_kernel`` through
``torch.library.wrap_triton``. PyTorch 2.5.1 does not provide that attribute.
Later PyTorch uses it to trace kernels; with wrapping disabled, the same
function returns the original ``JITFunction`` so ``kernel[grid](*args)`` is the
Triton launch. This module installs that identity only while one eager
``generate()`` call runs, then removes it.

The attribute is process-global for that call. The shim returns the original
kernel and otherwise raises, so it does not launch a different kernel or edit
arguments. A non-blocking lock rejects a second entry. ``finally`` restores the
previous binding, including when generation raises.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator


ACCREDITED_TORCH = "2.5.1+cu121"
ACCREDITED_TRITON = "3.1.0"
ACCREDITED_FLASH_ATTN = "2.8.3.post2"
ACCREDITED_TRANSFORMERS = "4.57.6"
ROPE_ADAPTER_ID = "gollie_flash_rope_wrap_triton_v1"
_LOCK = Lock()


def rope_compatibility_record() -> dict[str, object]:
    """Operational record. It is not part of the scientific run identity."""
    return {
        "adapter": ROPE_ADAPTER_ID,
        "flash_attn": ACCREDITED_FLASH_ATTN,
        "global_effect": "torch.library.wrap_triton is visible only during one eager generate call",
        "kernel": "flash_attn.ops.triton.rotary.rotary_kernel",
        "shim": "identity_for_triton_jit_function",
        "torch": ACCREDITED_TORCH,
        "triton": ACCREDITED_TRITON,
        "transformers": ACCREDITED_TRANSFORMERS,
    }


def _library_module() -> Any:
    import torch

    return torch.library


def assert_gollie_rope_runtime() -> None:
    """Refuse every runtime other than the accredited eager stack."""
    import torch

    found_torch = torch.__version__
    if found_torch != ACCREDITED_TORCH:
        raise RuntimeError(
            f"GoLLIE RoPE adapter requires torch {ACCREDITED_TORCH}; found {found_torch}"
        )
    try:
        import triton
    except ImportError as exc:
        raise RuntimeError(
            f"GoLLIE RoPE adapter requires triton {ACCREDITED_TRITON}"
        ) from exc
    found_triton = getattr(triton, "__version__", None)
    if found_triton != ACCREDITED_TRITON:
        raise RuntimeError(
            f"GoLLIE RoPE adapter requires triton {ACCREDITED_TRITON}; found {found_triton}"
        )
    try:
        import flash_attn
    except ImportError as exc:
        raise RuntimeError(
            f"GoLLIE RoPE adapter requires flash_attn {ACCREDITED_FLASH_ATTN}"
        ) from exc
    found_flash = getattr(flash_attn, "__version__", None)
    if found_flash != ACCREDITED_FLASH_ATTN:
        raise RuntimeError(
            "GoLLIE RoPE adapter requires flash_attn "
            f"{ACCREDITED_FLASH_ATTN}; found {found_flash}"
        )


def identity_wrap_triton(kernel: Any, /) -> Any:
    """Return the original Triton kernel so ``kernel[grid](*args)`` is unchanged."""
    kind = type(kernel)
    if kind.__module__ != "triton.runtime.jit" or kind.__name__ != "JITFunction":
        raise RuntimeError(
            "GoLLIE RoPE adapter only returns the original Triton JITFunction"
        )
    return kernel


def _refuse_non_eager(model: Any) -> None:
    import torch

    if getattr(model, "training", False):
        raise RuntimeError("GoLLIE RoPE adapter refuses training mode")
    if torch.is_grad_enabled():
        raise RuntimeError("GoLLIE RoPE adapter refuses grad-enabled execution")
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if not callable(is_compiling):
        raise RuntimeError("GoLLIE RoPE adapter cannot confirm eager compilation state")
    if is_compiling():
        raise RuntimeError("GoLLIE RoPE adapter refuses torch.compile")
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_compiling = getattr(dynamo, "is_compiling", None)
    if not callable(dynamo_compiling):
        raise RuntimeError("GoLLIE RoPE adapter cannot confirm that Dynamo is inactive")
    if dynamo_compiling():
        raise RuntimeError("GoLLIE RoPE adapter refuses Dynamo tracing")
    jit = getattr(torch, "jit", None)
    is_tracing = getattr(jit, "is_tracing", None)
    is_scripting = getattr(jit, "is_scripting", None)
    if not callable(is_tracing) or not callable(is_scripting):
        raise RuntimeError("GoLLIE RoPE adapter cannot confirm that TorchScript is inactive")
    if is_tracing() or is_scripting():
        raise RuntimeError("GoLLIE RoPE adapter refuses TorchScript")
    is_exporting = getattr(compiler, "is_exporting", None)
    if callable(is_exporting) and is_exporting():
        raise RuntimeError("GoLLIE RoPE adapter refuses export")


@contextmanager
def flash_rope_eager_launch(model: Any) -> Iterator[None]:
    """Install the identity shim for one eager generation call."""
    assert_gollie_rope_runtime()
    _refuse_non_eager(model)
    library = _library_module()
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("GoLLIE RoPE adapter is already active")
    installed = False
    try:
        if hasattr(library, "wrap_triton"):
            raise RuntimeError(
                "GoLLIE RoPE adapter refuses to replace an existing torch.library.wrap_triton"
            )
        library.wrap_triton = identity_wrap_triton
        installed = True
        yield
    finally:
        try:
            if installed and hasattr(library, "wrap_triton"):
                del library.wrap_triton
        finally:
            _LOCK.release()
