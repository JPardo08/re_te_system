"""Offline GoLLIE transition checks for EXP.4.5.1.

Synthetic hashes and monkeypatched probes only. No wheel install, no kernel
launch, and no weight download.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cesvima"))

import gollie_environment as gollie_env  # noqa: E402
import j15_gollie_preflight as j15  # noqa: E402
import j16_gollie_download as j16  # noqa: E402
from flash_attn_pin import (  # noqa: E402
    EXPECTED_BLACK,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    FLASH_ATTN_VERSION,
    FLASH_ATTN_WHEEL_SHA256,
    GOLLIE_J16_DOWNLOAD_AUTHORIZED,
    INSTALLED_EXTENSION_AT,
    INSTALLED_EXTENSION_GLIBC_MAX,
    INSTALLED_EXTENSION_SHA256,
    STATIC_GLIBC_IS_ABI_PASS,
    WHEEL_BUILD_FINISHED_AT,
    WHEEL_L40S_KERNEL_VALIDATED,
    recorded_flash_attention_evidence,
)


FUTURE = "ab" * 32


def _versions(**overrides: str) -> dict[str, str]:
    values = {
        "torch_version": EXPECTED_TORCH,
        "transformers_version": EXPECTED_TRANSFORMERS,
        "flash_attn_version": FLASH_ATTN_VERSION,
        "black_version": EXPECTED_BLACK,
    }
    values.update(overrides)
    return values


def _scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    scratch = tmp_path / "scratch"
    runs = scratch / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    monkeypatch.setenv("SLURM_JOB_ID", "1350001")
    return runs


def _assess(installed: str | None, wheel_member: str | None, **extra: object) -> dict[str, object]:
    return gollie_env.assess_gollie_installation(
        installed_extension_sha256=installed,
        wheel_extension_sha256=wheel_member,
        wheel_sha256=FLASH_ATTN_WHEEL_SHA256,
        **_versions(),
        **extra,
    )


def test_recorded_install_is_not_abi_pass() -> None:
    evidence = recorded_flash_attention_evidence()
    assert evidence["installed_extension_sha256"] == INSTALLED_EXTENSION_SHA256
    assert evidence["installed_extension_sha256"] == (
        "1fed48745de90c41071d324ff0bbdf17c3c054c679279e5bc30c640b74007a03"
    )
    assert evidence["installed_at"] == INSTALLED_EXTENSION_AT == "2026-10-26T14:50:00Z"
    assert evidence["wheel_finished_at"] == WHEEL_BUILD_FINISHED_AT == "2026-10-26T19:36:00Z"
    assert INSTALLED_EXTENSION_AT < WHEEL_BUILD_FINISHED_AT
    assert evidence["installed_glibc_max"] == INSTALLED_EXTENSION_GLIBC_MAX == "2.32"
    assert evidence["wheel_static_glibc_max"] == "2.14"
    assert evidence["declared_version"] == FLASH_ATTN_VERSION
    assert evidence["same_declared_version"] is True
    assert evidence["binary_identity"] == "BINARY_MISMATCH"
    assert evidence["wheel_sha256"] == FLASH_ATTN_WHEEL_SHA256
    assert evidence["wheel_build_pass"] is True
    assert evidence["wheel_l40s_kernel_validated"] is WHEEL_L40S_KERNEL_VALIDATED is False
    assert evidence["abi_pass"] is False
    assert evidence["abi_failed"] is False
    assert evidence["abi_verdict"] == "not_decided"
    assert evidence["static_glibc_is_abi_pass"] is STATIC_GLIBC_IS_ABI_PASS is False
    assert gollie_env.current_artifacts_allow_preflight() is False


def test_false_or_distinct_identity_blocks_preflight() -> None:
    distinct = _assess("a" * 64, "b" * 64)
    assert distinct["binary_identity"] == "BINARY_MISMATCH"
    assert distinct["abi_pass"] is False
    assert distinct["abi_failed"] is False
    assert distinct["accredited_for_reuse"] is False
    claimed = _assess(
        INSTALLED_EXTENSION_SHA256,
        INSTALLED_EXTENSION_SHA256,
        dynamic_dependencies_resolved=True,
        glibc_requirement_satisfied=True,
        abi_observed_on_gpu_node=True,
    )
    assert claimed["known_precorrection_binary"] is True
    assert claimed["binary_identity"] == "BINARY_MISMATCH"
    assert claimed["abi_pass"] is False
    assert claimed["abi_failed"] is False
    assert claimed["installed_glibc_max"] == "2.32"
    report = gollie_env.build_preflight_report(
        claimed,
        job_id="1350001",
        node="r1n1",
        kernel_finite=True,
        kernel_device_family="L40S",
    )
    assert report["status"] == "BLOCKED_BINARY_IDENTITY"
    assert report["preflight_pass"] is False
    assert report["abi_pass"] is False
    missing = _assess(FUTURE, None)
    assert missing["binary_identity"] == "NOT_COMPARED"
    assert missing["accredited_for_reuse"] is False


def test_static_glibc_and_unobserved_abi_do_not_pass() -> None:
    matched = _assess(FUTURE, FUTURE)
    assert matched["binary_identity"] == "MATCH"
    assert matched["accredited_for_reuse"] is True
    assert matched["glibc_requirement_satisfied"] is None
    assert matched["abi_pass"] is False
    assert matched["static_glibc_is_abi_pass"] is False
    report = gollie_env.build_preflight_report(
        matched,
        job_id="1350001",
        node="r1n1",
        kernel_finite=True,
        kernel_device_family="L40S",
    )
    assert report["status"] == "BLOCKED_ABI"
    assert report["preflight_pass"] is False
    negative = _assess(
        FUTURE,
        FUTURE,
        dynamic_dependencies_resolved=True,
        glibc_requirement_satisfied=False,
        abi_observed_on_gpu_node=True,
    )
    assert negative["abi_failed"] is True
    assert negative["abi_pass"] is False
    blocked = gollie_env.build_preflight_report(
        negative,
        job_id="1350001",
        node="r1n1",
        kernel_finite=True,
        kernel_device_family="L40S",
    )
    assert blocked["status"] == "BLOCKED_ABI"
    assert blocked["abi_failed"] is True


def test_simulated_match_abi_and_kernel_pass_without_opening_j16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assessment = _assess(
        FUTURE,
        FUTURE,
        dynamic_dependencies_resolved=True,
        glibc_requirement_satisfied=True,
        import_succeeded=True,
        abi_observed_on_gpu_node=True,
    )
    assert assessment["abi_pass"] is True
    assert assessment["abi_verdict"] == "pass"
    assert assessment["abi_failed"] is False
    report = gollie_env.build_preflight_report(
        assessment,
        job_id="1350001",
        node="r1n1",
        kernel_finite=True,
        kernel_device_family="L40S",
    )
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["preflight_pass"] is True
    assert report["abi_pass"] is True
    assert report["build_pass"] is True
    assert report["recorded_wheel_build_pass"] is True
    assert report["recorded_prior_l40s_validation"] is False
    assert report["slurm_job_id"] == "1350001"
    assert report["installed_extension_sha256"] == FUTURE
    assert report["wheel_extension_sha256"] == FUTURE
    assert report["wheel_sha256"] == FLASH_ATTN_WHEEL_SHA256
    assert report["flash_attn"] == FLASH_ATTN_VERSION
    assert report["kernel_device_family"] == "L40S"
    assert report["kernel_finite"] is True
    assert report["dynamic_dependencies_resolved"] is True
    assert report["glibc_requirement_satisfied"] is True
    assert report["import_succeeded"] is True
    assert report["weights_downloaded"] is False
    assert gollie_env.j16_rejection_reason(report, "1350001") is None
    assert gollie_env.j16_rejection_reason(report, "1350002") == (
        "J15 report belongs to another JOBID; refusing to download"
    )
    assert gollie_env.download_authorized() is False
    assert GOLLIE_J16_DOWNLOAD_AUTHORIZED is False
    wrong_device = gollie_env.build_preflight_report(
        assessment,
        job_id="1350001",
        node="r1n1",
        kernel_finite=True,
        kernel_device_family="V100",
    )
    assert wrong_device["status"] == "BLOCKED_PREFLIGHT"
    assert wrong_device["preflight_pass"] is False

    runs = _scratch(tmp_path, monkeypatch)
    legacy = runs / "gollie" / "preflight_report.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"status":"PASS"}\n', encoding="utf-8")
    monkeypatch.setenv("SLURMD_NODENAME", "r1n1")
    assessment["installed_extension_path"] = str(
        tmp_path / "lib" / "python3.12" / "site-packages" / "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
    )
    monkeypatch.setattr(j15, "probe_installed_environment", lambda: assessment)
    monkeypatch.setattr(j15, "_device_family", lambda: "L40S")
    monkeypatch.setattr(
        j15,
        "collect_runtime_abi",
        lambda _extension, *, device_family: {
            "dynamic_dependencies_resolved": True,
            "glibc_requirement_satisfied": True,
            "import_succeeded": True,
            "observed_on_gpu_node": device_family == "L40S",
            "static_glibc_used": False,
        },
    )

    def kernel() -> dict[str, object]:
        return {"cuda_device_name": "NVIDIA L40S"}

    monkeypatch.setattr(j15, "_kernel_observation", kernel)
    assert j15.main() == 0
    written = json.loads(
        (runs / "gollie" / "preflight" / "jobs" / "1350001" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["status"] == "PREFLIGHT_PASS"
    assert written["slurm_job_id"] == "1350001"
    assert (runs / "gollie" / "preflight" / "accepted_job_id.txt").read_text(encoding="utf-8") == (
        "1350001\n"
    )
    assert legacy.read_text(encoding="utf-8") == '{"status":"PASS"}\n'

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("weights were downloaded")

    monkeypatch.setattr(j16, "recover_snapshot", explode)
    with pytest.raises(SystemExit):
        j16.main()
    assert "J16 weight download is not authorized" in capsys.readouterr().out


def test_j15_does_not_run_the_kernel_for_the_recorded_or_unobserved_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _scratch(tmp_path, monkeypatch)

    def kernel() -> dict[str, object]:
        raise AssertionError("kernel ran")

    monkeypatch.setattr(j15, "_kernel_observation", kernel)
    monkeypatch.setattr(j15, "probe_installed_environment", lambda: _assess(INSTALLED_EXTENSION_SHA256, FUTURE))
    assert j15.main() == 2
    first = json.loads(
        (runs / "gollie" / "preflight" / "jobs" / "1350001" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert first["status"] == "BLOCKED_BINARY_IDENTITY"
    assert first["abi_pass"] is False
    assert first["abi_failed"] is False
    assert not (runs / "gollie" / "preflight" / "accepted_job_id.txt").exists()

    monkeypatch.setenv("SLURM_JOB_ID", "1350002")
    monkeypatch.setattr(j15, "probe_installed_environment", lambda: _assess(FUTURE, FUTURE))
    assert j15.main() == 2
    second = json.loads(
        (runs / "gollie" / "preflight" / "jobs" / "1350002" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert second["status"] == "BLOCKED_ABI"
    assert second["binary_identity"] == "MATCH"
    assert second["abi_pass"] is False
    assert not (runs / "gollie" / "preflight" / "accepted_job_id.txt").exists()


def test_legacy_gollie_report_is_not_authority(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    legacy = runs / "gollie" / "preflight_report.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"status": "PREFLIGHT_PASS", "weights_downloaded": False}), encoding="utf-8")
    with pytest.raises(SystemExit):
        j16.load_accepted_report(runs)
    assert "unidentified" in capsys.readouterr().out
