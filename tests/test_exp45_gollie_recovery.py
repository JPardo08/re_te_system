"""Offline GoLLIE checks for the EXP.4.5 recovery contract."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
sys.path.insert(0, str(SCRIPTS))

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
    GOLLIE_J17_SMOKE_AUTHORIZED,
    OBSERVED_BINARY_IDENTITY,
    interpret_binary_identity,
)


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
    monkeypatch.setenv("SLURM_JOB_ID", "1349123")
    return runs


def test_binary_mismatch_is_not_an_abi_failure_and_blocks_reuse() -> None:
    assert OBSERVED_BINARY_IDENTITY == "BINARY_MISMATCH"
    observed = interpret_binary_identity(OBSERVED_BINARY_IDENTITY)
    assert observed["abi_failed"] is False
    assert observed["abi_verdict"] == "not_decided"
    assert observed["accredited_for_reuse"] is False
    assert observed["automatic_reinstall"] is False
    assert observed["compilation"] is False
    assessment = gollie_env.assess_gollie_installation(
        installed_extension_sha256="a" * 64,
        wheel_extension_sha256="b" * 64,
        wheel_sha256=FLASH_ATTN_WHEEL_SHA256,
        **_versions(),
    )
    assert assessment["versions_ok"] is True
    assert assessment["binary_identity"] == "BINARY_MISMATCH"
    assert assessment["accredited_for_reuse"] is False
    assert assessment["installed_package_is_verified_wheel"] is False
    assert assessment["abi_failed"] is False
    assert assessment["abi_pass"] is False
    plan = gollie_env.isolated_recovery_plan(assessment)
    assert plan["authorized"] is False
    assert plan["automatic_reinstall"] is False
    assert "pip install flash-attn" in plan["blocked_actions"]
    with pytest.raises(gollie_env.RecoveryNotAuthorized):
        gollie_env.execute_isolated_recovery()


def test_unpinned_wheel_is_not_a_match_and_version_drift_fails_closed() -> None:
    same = "c" * 64
    unpinned = gollie_env.assess_gollie_installation(
        installed_extension_sha256=same,
        wheel_extension_sha256=same,
        wheel_sha256="d" * 64,
        **_versions(),
    )
    assert unpinned["binary_identity"] == "NOT_COMPARED"
    assert unpinned["accredited_for_reuse"] is False
    drifted = gollie_env.assess_gollie_installation(
        installed_extension_sha256=same,
        wheel_extension_sha256=same,
        wheel_sha256=FLASH_ATTN_WHEEL_SHA256,
        **_versions(flash_attn_version="2.8.2"),
    )
    assert drifted["versions_ok"] is False
    assert drifted["accredited_for_reuse"] is False


def test_match_still_does_not_record_abi_pass() -> None:
    same = "e" * 64
    assessment = gollie_env.assess_gollie_installation(
        installed_extension_sha256=same,
        wheel_extension_sha256=same,
        wheel_sha256=FLASH_ATTN_WHEEL_SHA256,
        **_versions(),
    )
    assert assessment["accredited_for_reuse"] is True
    report = gollie_env.build_preflight_report(
        assessment,
        job_id="1349032",
        node="r1n1",
        kernel_finite=True,
    )
    assert report["status"] == "BLOCKED_ABI"
    assert report["preflight_pass"] is False
    assert report["abi_pass"] is False
    assert report["abi_verdict"] == "not_decided"
    assert report["wheel_sha256"] == FLASH_ATTN_WHEEL_SHA256
    assert gollie_env.j16_rejection_reason(report, "1349032") == (
        "J15 status is BLOCKED_ABI; refusing to download"
    )
    assert gollie_env.j16_rejection_reason(report, "other") == (
        "J15 report belongs to another JOBID; refusing to download"
    )


def test_j15_writes_a_blocked_report_without_running_the_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _scratch(tmp_path, monkeypatch)
    legacy = runs / "gollie" / "preflight_report.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"status":"PASS"}\n', encoding="utf-8")
    assessment = gollie_env.assess_gollie_installation(
        installed_extension_sha256="a" * 64,
        wheel_extension_sha256="b" * 64,
        wheel_sha256=FLASH_ATTN_WHEEL_SHA256,
        **_versions(),
    )
    monkeypatch.setattr(j15, "probe_installed_environment", lambda: assessment)

    def kernel() -> dict[str, object]:
        raise AssertionError("kernel ran")

    monkeypatch.setattr(j15, "_kernel_observation", kernel)
    assert j15.main() == 2
    assert legacy.read_text(encoding="utf-8") == '{"status":"PASS"}\n'
    report = json.loads(
        (runs / "gollie" / "preflight" / "jobs" / "1349123" / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "BLOCKED_BINARY_IDENTITY"
    assert report["binary_identity"] == "BINARY_MISMATCH"
    assert report["abi_failed"] is False
    assert report["slurm_job_id"] == "1349123"
    assert not (runs / "gollie" / "preflight" / "accepted_job_id.txt").exists()


def test_j16_rejects_legacy_and_stays_unauthorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    legacy = runs / "gollie" / "preflight_report.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps({"status": "PASS", "weights_downloaded": False}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        j16.load_accepted_report(runs)
    assert "unidentified" in capsys.readouterr().out

    job = "1349032"
    report = {
        "abi_pass": True,
        "abi_verdict": "pass",
        "binary_identity": "MATCH",
        "build_pass": True,
        "node": "r1n1",
        "preflight_pass": True,
        "slurm_job_id": job,
        "status": "PREFLIGHT_PASS",
        "dynamic_dependencies_resolved": True,
        "flash_attn": FLASH_ATTN_VERSION,
        "glibc_requirement_satisfied": True,
        "import_succeeded": True,
        "installed_extension_sha256": "f" * 64,
        "kernel_device_family": "L40S",
        "kernel_finite": True,
        "weights_downloaded": False,
        "wheel_extension_sha256": "f" * 64,
        "wheel_sha256": FLASH_ATTN_WHEEL_SHA256,
    }
    report_path = runs / "gollie" / "preflight" / "jobs" / job / "preflight_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (runs / "gollie" / "preflight" / "accepted_job_id.txt").write_text(job + "\n", encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("weights were downloaded")

    monkeypatch.setattr(j16, "cache_or_download", explode)
    with pytest.raises(SystemExit):
        j16.main()
    assert "J16 weight download is not authorized" in capsys.readouterr().out
    assert GOLLIE_J16_DOWNLOAD_AUTHORIZED is False
    assert GOLLIE_J17_SMOKE_AUTHORIZED is False


def test_bootstrap_does_not_compile_and_j17_stays_closed(tmp_path: Path) -> None:
    bootstrap = (SCRIPTS / "bootstrap_gollie_venv.sh").read_text(encoding="utf-8")
    executable = [
        line.strip()
        for line in bootstrap.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("pip install" in line or line.strip().startswith("pip ") for line in executable)
    assert all("flash-attn" not in line for line in executable)
    j17 = (SCRIPTS / "j17_gollie_native_smoke.sh").read_text(encoding="utf-8")
    assert 'GOLLIE_J17_SMOKE_AUTHORIZED:-' in j17
    env = (SCRIPTS / "env.sh").read_text(encoding="utf-8")
    assert "GOLLIE_J17_SMOKE_AUTHORIZED=1" not in env
    missing = subprocess.run(
        ["bash", str(SCRIPTS / "bootstrap_gollie_venv.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "GOLLIE_VENV": str(tmp_path / "missing")},
    )
    assert missing.returncode == 1
    forbidden = subprocess.run(
        ["bash", str(SCRIPTS / "bootstrap_gollie_venv.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "GOLLIE_BOOTSTRAP_ALLOW_COMPILE": "1", "GOLLIE_VENV": str(tmp_path)},
    )
    assert forbidden.returncode == 1
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/bash\nexit 2\n", encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    assessed = subprocess.run(
        ["bash", str(SCRIPTS / "bootstrap_gollie_venv.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "GOLLIE_VENV": str(python.parent.parent),
            "SYSTEM_ROOT": str(ROOT),
        },
    )
    assert assessed.returncode == 2

