"""Offline J15 discovery and ABI observations. No wheel, kernel, or install."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cesvima"))

import gollie_environment as gollie_env  # noqa: E402
import j15_gollie_preflight as j15  # noqa: E402
from flash_attn_pin import (  # noqa: E402
    EXPECTED_BLACK,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    FLASH_ATTN_EXTENSION,
    FLASH_ATTN_RECORDED_GLIBC_MAX,
    FLASH_ATTN_VERSION,
    FLASH_ATTN_WHEEL_SHA256,
    INSTALLED_EXTENSION_SHA256,
)


EXTENSION = FLASH_ATTN_EXTENSION
LDD_RESOLVED = "libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x1)\n"
LDD_MISSING = "libmissing.so.1 => not found\n"
SYMBOLS_214 = "Name: GLIBC_2.14\nName: GLIBC_2.17\n"
SYMBOLS_232 = "Name: GLIBC_2.32\n"


def _site(venv: Path) -> Path:
    return venv / "lib" / "python3.12" / "site-packages"


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_discovery_accepts_only_the_cesvima_site_packages_file(tmp_path: Path) -> None:
    venv = tmp_path / "gollie-l40s"
    exact = _site(venv) / EXTENSION
    _write(exact, b"cesvima-extension")
    other_python = venv / "lib" / "python3.11" / "site-packages" / EXTENSION
    _write(other_python, b"other-python")
    assert gollie_env._find_extension(venv) == exact


def test_nested_copy_and_two_candidates_are_rejected(tmp_path: Path) -> None:
    venv = tmp_path / "gollie-l40s"
    nested = _site(venv) / "flash_attn" / EXTENSION
    _write(nested, b"nested-only")
    assert gollie_env._find_extension(venv) is None
    exact = _site(venv) / EXTENSION
    _write(exact, b"also-present")
    assert gollie_env._find_extension(venv) is None


def test_ldd_and_glibc_symbols_are_observations_not_the_static_wheel_maximum() -> None:
    assert gollie_env.dynamic_dependencies_resolved(LDD_RESOLVED) is True
    assert gollie_env.dynamic_dependencies_resolved(LDD_MISSING) is False
    assert gollie_env.dynamic_dependencies_resolved("") is None
    assert gollie_env.glibc_requirement_satisfied(SYMBOLS_214, "glibc 2.35") is True
    assert gollie_env.glibc_requirement_satisfied(SYMBOLS_232, "2.31") is False
    assert gollie_env.glibc_requirement_satisfied(FLASH_ATTN_RECORDED_GLIBC_MAX, "2.35") is None
    assert gollie_env.glibc_requirement_satisfied(None, "2.35") is None
    absent = gollie_env.collect_runtime_abi(
        Path("/absent/flash_attn_2_cuda.so"),
        device_family="L40S",
    )
    assert absent["observed_on_gpu_node"] is False
    assert absent["static_glibc_used"] is False
    assert absent["import_succeeded"] is None


def test_runtime_abi_is_not_collected_off_the_l40s(tmp_path: Path) -> None:
    extension = _site(tmp_path) / EXTENSION
    _write(extension, b"not-executed")
    observed = gollie_env.collect_runtime_abi(extension, device_family=None)
    assert observed["observed_on_gpu_node"] is False
    assert observed["dynamic_dependencies_resolved"] is None
    assert observed["glibc_requirement_satisfied"] is None
    assert observed["import_succeeded"] is None
    assert observed["static_glibc_used"] is False


def test_j15_observes_abi_before_the_kernel_and_keeps_j16_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    runs = scratch / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    monkeypatch.setenv("SLURM_JOB_ID", "1350008")
    monkeypatch.setenv("SLURMD_NODENAME", "r1n1")
    order: list[str] = []
    extension = _site(tmp_path / "isolated") / EXTENSION
    _write(extension, b"isolated-accredited")
    assessment = gollie_env.assess_gollie_installation(
        torch_version=EXPECTED_TORCH,
        transformers_version=EXPECTED_TRANSFORMERS,
        flash_attn_version=FLASH_ATTN_VERSION,
        black_version=EXPECTED_BLACK,
        installed_extension_sha256="ab" * 32,
        wheel_sha256=FLASH_ATTN_WHEEL_SHA256,
        wheel_extension_sha256="ab" * 32,
    )
    assessment["installed_extension_path"] = str(extension)
    assert assessment["abi_pass"] is False
    monkeypatch.setattr(j15, "probe_installed_environment", lambda: assessment)

    def family() -> str:
        order.append("device")
        return "L40S"

    def collect(path: Path, *, device_family: str | None) -> dict[str, object]:
        order.append("abi")
        assert path == extension
        assert device_family == "L40S"
        return {
            "dynamic_dependencies_resolved": gollie_env.dynamic_dependencies_resolved(LDD_RESOLVED),
            "glibc_requirement_satisfied": gollie_env.glibc_requirement_satisfied(SYMBOLS_214, "glibc 2.35"),
            "import_succeeded": True,
            "observed_on_gpu_node": True,
            "static_glibc_used": False,
        }

    def kernel() -> dict[str, object]:
        order.append("kernel")
        return {"cuda_device_name": "NVIDIA L40S"}

    monkeypatch.setattr(j15, "_device_family", family)
    monkeypatch.setattr(j15, "collect_runtime_abi", collect)
    monkeypatch.setattr(j15, "_kernel_observation", kernel)
    assert j15.main() == 0
    assert order == ["device", "abi", "kernel"]
    report = json.loads(
        (runs / "gollie" / "preflight" / "jobs" / "1350008" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["abi_pass"] is True
    assert report["import_succeeded"] is True
    assert report["installed_extension_sha256"] == "ab" * 32
    assert report["recorded_evidence"]["installed_extension_sha256"] == INSTALLED_EXTENSION_SHA256
    assert report["recorded_evidence"]["binary_identity"] == "BINARY_MISMATCH"
    assert report["static_glibc_is_abi_pass"] is False
    assert gollie_env.download_authorized() is False


def test_mismatch_does_not_observe_abi_or_run_the_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    runs = scratch / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    monkeypatch.setenv("SLURM_JOB_ID", "1350009")
    assessment = gollie_env.assess_gollie_installation(
        torch_version=EXPECTED_TORCH,
        transformers_version=EXPECTED_TRANSFORMERS,
        flash_attn_version=FLASH_ATTN_VERSION,
        black_version=EXPECTED_BLACK,
        installed_extension_sha256=INSTALLED_EXTENSION_SHA256,
        wheel_sha256=FLASH_ATTN_WHEEL_SHA256,
        wheel_extension_sha256="ab" * 32,
    )
    monkeypatch.setattr(j15, "probe_installed_environment", lambda: assessment)

    def forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("ABI observation or kernel ran on a mismatched binary")

    monkeypatch.setattr(j15, "_device_family", forbidden)
    monkeypatch.setattr(j15, "collect_runtime_abi", forbidden)
    monkeypatch.setattr(j15, "_kernel_observation", forbidden)
    assert j15.main() == 2
    report = json.loads(
        (runs / "gollie" / "preflight" / "jobs" / "1350009" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "BLOCKED_BINARY_IDENTITY"
    assert report["binary_identity"] == "BINARY_MISMATCH"
    assert report["abi_pass"] is False


def test_identity_probe_does_not_run_dynamic_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv = tmp_path / "gollie-l40s"
    extension = _site(venv) / EXTENSION
    _write(extension, b"identity-only")
    monkeypatch.setenv("GOLLIE_VENV", str(venv))
    monkeypatch.delenv("GOLLIE_FLASH_ATTN_WHEEL", raising=False)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("identity probe ran a dynamic tool")

    monkeypatch.setattr(gollie_env.subprocess, "run", forbidden)
    monkeypatch.setattr(gollie_env, "_import_extension_without_kernel", forbidden)
    monkeypatch.setattr(gollie_env, "_probe_versions", lambda: {})
    assessment = gollie_env.probe_installed_environment()
    assert assessment["installed_extension_path"] == str(extension)
    assert assessment["binary_identity"] == "NOT_COMPARED"
    assert assessment["abi_pass"] is False
