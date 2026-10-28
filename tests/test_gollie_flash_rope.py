"""Flash RoPE eager shim. The numerical kernel comparison waits for an L40S."""

from __future__ import annotations

from types import SimpleNamespace
import threading
from typing import Any

import pytest
import torch

from re_te_system.contracts import InputRecord
from re_te_system.extractors.gollie_flash_rope import (
    ACCREDITED_FLASH_ATTN,
    ACCREDITED_TORCH,
    ACCREDITED_TRITON,
    ACCREDITED_TRANSFORMERS,
    ROPE_ADAPTER_ID,
    flash_rope_eager_launch,
    identity_wrap_triton,
    rope_compatibility_record,
)
from re_te_system.extractors import gollie_flash_rope
from re_te_system.extractors.mock import MockExtractor
from re_te_system.runner.run import run_pipeline


DATASET = {"hash": "abc", "id": "test", "version": "1"}
SOURCE = {"documents_sha256": "abc", "manifest_sha256": "def"}
GENERATION = {
    "do_sample": False,
    "max_new_tokens": None,
    "num_beams": 1,
    "seed": None,
}
GRID = (2, 1, 1)
LAUNCH_ARGS = ("output", "x", "cos", "sin", 0)
LAUNCH_KWARGS = {"BLOCK_M": 8, "BLOCK_H": 2}


class JITFunction:
    def __init__(self) -> None:
        self.launches: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def __getitem__(self, grid: Any) -> Any:
        def launch(*args: Any, **kwargs: Any) -> str:
            self.launches.append((grid, args, kwargs))
            return "original-kernel"

        return launch


JITFunction.__module__ = "triton.runtime.jit"


def _accredited(monkeypatch: pytest.MonkeyPatch, library: Any) -> None:
    monkeypatch.setattr(gollie_flash_rope, "assert_gollie_rope_runtime", lambda: None)
    monkeypatch.setattr(gollie_flash_rope, "_library_module", lambda: library)
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)


def test_rope_pins_match_the_isolated_runtime() -> None:
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts" / "cesvima"
    sys.path.insert(0, str(scripts))
    from flash_attn_pin import (  # noqa: WPS433
        EXPECTED_TORCH,
        EXPECTED_TRANSFORMERS,
        EXPECTED_TRITON,
        FLASH_ATTN_VERSION,
    )

    assert ACCREDITED_TORCH == EXPECTED_TORCH
    assert ACCREDITED_TRITON == EXPECTED_TRITON
    assert ACCREDITED_FLASH_ATTN == FLASH_ATTN_VERSION
    assert ACCREDITED_TRANSFORMERS == EXPECTED_TRANSFORMERS


def test_local_runtime_is_refused_without_touching_torch_library() -> None:
    original = torch.library.wrap_triton
    with pytest.raises(RuntimeError, match="2.5.1\\+cu121"):
        with flash_rope_eager_launch(SimpleNamespace(training=False)):
            pass
    assert torch.library.wrap_triton is original


def test_existing_wrap_triton_is_not_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    library = SimpleNamespace(wrap_triton=sentinel)
    _accredited(monkeypatch, library)
    with pytest.raises(RuntimeError, match="existing"):
        with flash_rope_eager_launch(SimpleNamespace(training=False)):
            pass
    assert library.wrap_triton is sentinel


def test_shim_launches_the_same_kernel_and_restores_after_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = SimpleNamespace()
    _accredited(monkeypatch, library)
    kernel = JITFunction()
    with pytest.raises(RuntimeError, match="boom"):
        with flash_rope_eager_launch(SimpleNamespace(training=False)):
            assert library.wrap_triton is identity_wrap_triton
            launched = library.wrap_triton(kernel)[GRID](*LAUNCH_ARGS, **LAUNCH_KWARGS)
            assert launched == "original-kernel"
            assert kernel.launches == [(GRID, LAUNCH_ARGS, LAUNCH_KWARGS)]
            with pytest.raises(RuntimeError, match="JITFunction"):
                library.wrap_triton(lambda: None)
            raise RuntimeError("boom")
    assert not hasattr(library, "wrap_triton")


def test_reentry_and_training_and_compile_leave_the_library_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = SimpleNamespace()
    _accredited(monkeypatch, library)
    with flash_rope_eager_launch(SimpleNamespace(training=False)):
        with pytest.raises(RuntimeError, match="already active"):
            with flash_rope_eager_launch(SimpleNamespace(training=False)):
                pass
        assert library.wrap_triton is identity_wrap_triton
    assert not hasattr(library, "wrap_triton")

    with pytest.raises(RuntimeError, match="training"):
        with flash_rope_eager_launch(SimpleNamespace(training=True)):
            pass
    assert not hasattr(library, "wrap_triton")

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="torch.compile"):
        with flash_rope_eager_launch(SimpleNamespace(training=False)):
            pass
    assert not hasattr(library, "wrap_triton")


def test_a_second_thread_cannot_enter_while_the_shim_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = SimpleNamespace()
    _accredited(monkeypatch, library)
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def other() -> None:
        entered.wait(timeout=2)
        try:
            with flash_rope_eager_launch(SimpleNamespace(training=False)):
                pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            release.set()

    worker = threading.Thread(target=other)
    worker.start()
    with flash_rope_eager_launch(SimpleNamespace(training=False)):
        entered.set()
        assert release.wait(timeout=2)
    worker.join(timeout=2)
    assert len(errors) == 1
    assert "already active" in str(errors[0])
    assert not hasattr(library, "wrap_triton")


def test_rope_record_stays_outside_the_scientific_run_id(tmp_path) -> None:
    class Recorded(MockExtractor):
        rope_compatibility = rope_compatibility_record()

    plain = run_pipeline(
        inputs=[InputRecord("gollie-1", "gollie-1", "Ana visited Mary.")],
        extractor=MockExtractor(),
        output_root=tmp_path / "plain",
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GENERATION,
        code_commit="test-commit",
    )
    recorded = run_pipeline(
        inputs=[InputRecord("gollie-1", "gollie-1", "Ana visited Mary.")],
        extractor=Recorded(),
        output_root=tmp_path / "recorded",
        dataset=DATASET,
        benchmark_source=SOURCE,
        generation=GENERATION,
        code_commit="test-commit",
    )
    assert plain.name == recorded.name
    import json

    manifest = json.loads((recorded / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["operational_metadata"]["rope_compatibility"]
    assert record["adapter"] == ROPE_ADAPTER_ID
    assert record["torch"] == ACCREDITED_TORCH
    assert record["triton"] == ACCREDITED_TRITON
    assert record["flash_attn"] == ACCREDITED_FLASH_ATTN
    assert record["transformers"] == ACCREDITED_TRANSFORMERS
    assert record["kernel"] == "flash_attn.ops.triton.rotary.rotary_kernel"
    assert "rope_compatibility" not in manifest["model"]


def _reference_rotary(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Non-interleaved rotation from the pinned ``rotary_kernel``.

    The first half is ``x0 * cos - x1 * sin`` and the second half is
    ``x0 * sin + x1 * cos``. ``cos`` and ``sin`` have shape
    ``(seqlen, rotary_dim / 2)``.
    """
    half = cos.shape[-1]
    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    left = values[..., :half]
    right = values[..., half : half * 2]
    output = values.clone()
    output[..., :half] = left * cos_b - right * sin_b
    output[..., half : half * 2] = left * sin_b + right * cos_b
    return output


def test_reference_formula_matches_the_pinned_half_rotation() -> None:
    values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    observed = _reference_rotary(values, torch.tensor([[0.0]]), torch.tensor([[1.0]]))
    assert observed.tolist() == [[[[-2.0, 1.0, 3.0, 4.0]]]]


def test_l40s_flash_rope_kernel_matches_the_reference_formula() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is absent; the L40S numerical comparison has not run")
    gollie_flash_rope.assert_gollie_rope_runtime()
    from flash_attn.ops.triton.rotary import apply_rotary

    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)
    values = torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.float32, generator=generator)
    cos = torch.randn(4, 4, device="cuda", dtype=torch.float32, generator=generator)
    sin = torch.randn(4, 4, device="cuda", dtype=torch.float32, generator=generator)
    reference = _reference_rotary(values, cos, sin)
    with torch.inference_mode():
        with flash_rope_eager_launch(SimpleNamespace(training=False)):
            observed = apply_rotary(values, cos, sin, interleaved=False, inplace=False)
    assert observed.shape == values.shape
    assert torch.allclose(observed, reference, rtol=1e-4, atol=1e-4)
    bf_values = values.to(dtype=torch.bfloat16)
    bf_cos = cos.to(dtype=torch.bfloat16)
    bf_sin = sin.to(dtype=torch.bfloat16)
    bf_reference = _reference_rotary(bf_values.float(), bf_cos.float(), bf_sin.float()).to(
        dtype=torch.bfloat16
    )
    with torch.inference_mode():
        with flash_rope_eager_launch(SimpleNamespace(training=False)):
            bf_observed = apply_rotary(
                bf_values, bf_cos, bf_sin, interleaved=False, inplace=False
            )
    assert torch.allclose(bf_observed, bf_reference, rtol=2e-2, atol=2e-2)
