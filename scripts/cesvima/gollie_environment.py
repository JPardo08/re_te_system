"""Validate the existing gollie-l40s environment. Never install or compile."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

from cesvima_common import fail, require_under_scratch, sha256_file, write_json
from flash_attn_pin import (
    EXPECTED_BLACK,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    FLASH_ATTN_EXTENSION,
    FLASH_ATTN_RECORDED_GLIBC_MAX,
    FLASH_ATTN_VERSION,
    FLASH_ATTN_WHEEL_SHA256,
    GOLLIE_J16_DOWNLOAD_AUTHORIZED,
    INSTALLED_EXTENSION_GLIBC_MAX,
    INSTALLED_EXTENSION_SHA256,
    ISOLATED_RECOVERY_AUTHORIZED,
    RECORDED_WHEEL_BUILD_PASS,
    STATIC_GLIBC_IS_ABI_PASS,
    WHEEL_L40S_KERNEL_VALIDATED,
    interpret_binary_identity,
    recorded_flash_attention_evidence,
)


class RecoveryNotAuthorized(RuntimeError):
    """Isolated wheel recovery exists only as a plan."""


def evaluate_abi(
    *,
    binary_identity: str,
    installed_extension_sha256: str | None,
    dynamic_dependencies_resolved: bool | None,
    glibc_requirement_satisfied: bool | None,
    observed_on_gpu_node: bool,
) -> dict[str, object]:
    """ABI_PASS requires a matched wheel binary observed on the GPU node.

    A missing observation stays not_decided. The pre-correction extension and
    a binary that is not MATCH never become ABI_PASS. Static GLIBC text is
    not an observation. An explicit negative observation of a MATCH is a
    failure of that binary, not of the historical mismatch.
    """
    if installed_extension_sha256 == INSTALLED_EXTENSION_SHA256 or binary_identity != "MATCH":
        return {
            "abi_failed": False,
            "abi_pass": False,
            "abi_verdict": "not_decided",
            "reason": "binary identity is not the pinned wheel",
        }
    if (
        not observed_on_gpu_node
        or dynamic_dependencies_resolved is None
        or glibc_requirement_satisfied is None
    ):
        return {
            "abi_failed": False,
            "abi_pass": False,
            "abi_verdict": "not_decided",
            "reason": "ABI was not observed on the GPU node",
        }
    if dynamic_dependencies_resolved is True and glibc_requirement_satisfied is True:
        return {
            "abi_failed": False,
            "abi_pass": True,
            "abi_verdict": "pass",
            "reason": None,
        }
    return {
        "abi_failed": True,
        "abi_pass": False,
        "abi_verdict": "fail",
        "reason": "dynamic dependencies or GLIBC are not satisfied on the GPU node",
    }


def assess_gollie_installation(
    *,
    torch_version: str | None,
    transformers_version: str | None,
    flash_attn_version: str | None,
    black_version: str | None,
    installed_extension_sha256: str | None,
    wheel_sha256: str | None,
    wheel_extension_sha256: str | None,
    dynamic_dependencies_resolved: bool | None = None,
    glibc_requirement_satisfied: bool | None = None,
    abi_observed_on_gpu_node: bool = False,
) -> dict[str, object]:
    """Compare versions and extension bytes. Do not treat a version string as the wheel."""
    versions_ok = (
        torch_version == EXPECTED_TORCH
        and transformers_version == EXPECTED_TRANSFORMERS
        and flash_attn_version == FLASH_ATTN_VERSION
        and black_version == EXPECTED_BLACK
    )
    wheel_checked = wheel_sha256 is not None
    wheel_ok = wheel_sha256 == FLASH_ATTN_WHEEL_SHA256
    known_precorrection = installed_extension_sha256 == INSTALLED_EXTENSION_SHA256
    if known_precorrection:
        identity = "BINARY_MISMATCH"
    elif not wheel_ok or installed_extension_sha256 is None or wheel_extension_sha256 is None:
        identity = "NOT_COMPARED"
    elif installed_extension_sha256 == wheel_extension_sha256:
        identity = "MATCH"
    else:
        identity = "BINARY_MISMATCH"
    interpreted = interpret_binary_identity(identity)
    abi = evaluate_abi(
        binary_identity=identity,
        installed_extension_sha256=installed_extension_sha256,
        dynamic_dependencies_resolved=dynamic_dependencies_resolved,
        glibc_requirement_satisfied=glibc_requirement_satisfied,
        observed_on_gpu_node=abi_observed_on_gpu_node,
    )
    accredited = bool(
        versions_ok and interpreted["accredited_for_reuse"] and wheel_ok and not known_precorrection
    )
    return {
        "abi_failed": abi["abi_failed"],
        "abi_observed_on_gpu_node": abi_observed_on_gpu_node,
        "abi_pass": abi["abi_pass"],
        "abi_reason": abi["reason"],
        "abi_verdict": abi["abi_verdict"],
        "accredited_for_reuse": accredited,
        "automatic_reinstall": False,
        "binary_identity": identity,
        "black": black_version,
        "build_pass_for_this_install": False,
        "compilation": False,
        "dynamic_dependencies_resolved": dynamic_dependencies_resolved,
        "flash_attn": flash_attn_version,
        "glibc_requirement_satisfied": glibc_requirement_satisfied,
        "installed_extension_sha256": installed_extension_sha256,
        "installed_glibc_max": INSTALLED_EXTENSION_GLIBC_MAX if known_precorrection else None,
        "installed_package_is_verified_wheel": accredited,
        "isolated_recovery_authorized": False,
        "known_precorrection_binary": known_precorrection,
        "recorded_wheel_build_pass": RECORDED_WHEEL_BUILD_PASS,
        "static_glibc_is_abi_pass": STATIC_GLIBC_IS_ABI_PASS,
        "torch": torch_version,
        "transformers": transformers_version,
        "versions_ok": versions_ok,
        "wheel_extension_sha256": wheel_extension_sha256,
        "wheel_file_checked": wheel_checked,
        "wheel_sha256": wheel_sha256,
        "wheel_sha256_ok": wheel_ok,
        "wheel_static_glibc_max": FLASH_ATTN_RECORDED_GLIBC_MAX,
    }


def isolated_recovery_plan(assessment: dict[str, object]) -> dict[str, object]:
    """Describe a future isolated recovery. This function does not execute it."""
    return {
        "assessment_binary_identity": assessment.get("binary_identity"),
        "authorized": False,
        "automatic_reinstall": False,
        "blocked_actions": [
            "pip install flash-attn",
            "pip install a different flash-attn version",
            "build flash-attn from source",
            "overwrite gollie-l40s in place",
        ],
        "compilation": False,
        "target": "a new venv, after a separate authorization, using only the pinned wheel",
        "wheel_sha256": FLASH_ATTN_WHEEL_SHA256,
    }


def execute_isolated_recovery() -> None:
    if ISOLATED_RECOVERY_AUTHORIZED is not True:
        raise RecoveryNotAuthorized(
            "isolated recovery is not authorized; refusing to reinstall or compile"
        )
    raise RecoveryNotAuthorized("recovery execution is not implemented")


def wheel_extension_sha256(wheel: Path) -> str | None:
    """Hash the extension inside the wheel. Return None when the wheel is not the pin."""
    if sha256_file(wheel) != FLASH_ATTN_WHEEL_SHA256:
        return None
    with zipfile.ZipFile(wheel) as archive:
        matches = [
            name
            for name in archive.namelist()
            if Path(name).name == FLASH_ATTN_EXTENSION
        ]
        if len(matches) != 1:
            return None
        payload = archive.read(matches[0])
    return hashlib.sha256(payload).hexdigest()


def job_report_dir(runs_root: Path, job_id: str) -> Path:
    _require_job_id(job_id)
    return runs_root / "gollie" / "preflight" / "jobs" / job_id


def _require_job_id(job_id: str) -> None:
    if not job_id or job_id in {".", ".."} or "/" in job_id or "\\" in job_id:
        fail(f"SLURM_JOB_ID is not a single path segment: {job_id!r}")


def write_job_report(runs_root: Path, job_id: str, payload: dict[str, object]) -> Path:
    path = job_report_dir(runs_root, job_id) / "preflight_report.json"
    target = require_under_scratch(path)
    if target.exists():
        fail(f"refusing to overwrite existing evidence: {target}")
    legacy = require_under_scratch(runs_root / "gollie" / "preflight_report.json")
    if target == legacy:
        fail("refusing to write the legacy GoLLIE preflight path")
    write_json(target, payload)
    return target


def build_preflight_report(
    assessment: dict[str, object],
    *,
    job_id: str,
    node: str | None,
    kernel_finite: bool | None,
    kernel_device_family: str | None = None,
) -> dict[str, object]:
    identity = str(assessment["binary_identity"])
    installed = assessment.get("installed_extension_sha256")
    wheel_member = assessment.get("wheel_extension_sha256")
    abi = evaluate_abi(
        binary_identity=identity,
        installed_extension_sha256=installed if isinstance(installed, str) else None,
        dynamic_dependencies_resolved=_optional_bool(assessment.get("dynamic_dependencies_resolved")),
        glibc_requirement_satisfied=_optional_bool(assessment.get("glibc_requirement_satisfied")),
        observed_on_gpu_node=assessment.get("abi_observed_on_gpu_node") is True,
    )
    hashes_match = (
        isinstance(installed, str)
        and installed == wheel_member
        and installed != INSTALLED_EXTENSION_SHA256
    )
    preflight_pass = bool(
        assessment.get("accredited_for_reuse") is True
        and assessment.get("wheel_sha256_ok") is True
        and identity == "MATCH"
        and hashes_match
        and abi["abi_pass"] is True
        and kernel_finite is True
        and node
        and kernel_device_family == "L40S"
        and job_id
    )
    if preflight_pass:
        status = "PREFLIGHT_PASS"
    elif identity != "MATCH":
        status = "BLOCKED_BINARY_IDENTITY"
    elif abi["abi_pass"] is not True:
        status = "BLOCKED_ABI"
    else:
        status = "BLOCKED_PREFLIGHT"
    return {
        "abi_failed": abi["abi_failed"],
        "abi_pass": abi["abi_pass"],
        "abi_reason": abi["reason"],
        "abi_verdict": abi["abi_verdict"],
        "binary_identity": identity,
        "build_pass": bool(preflight_pass and assessment.get("wheel_sha256_ok") is True),
        "dynamic_dependencies_resolved": assessment.get("dynamic_dependencies_resolved"),
        "extension_filename": FLASH_ATTN_EXTENSION,
        "flash_attn": assessment.get("flash_attn"),
        "glibc_requirement_satisfied": assessment.get("glibc_requirement_satisfied"),
        "installed_extension_sha256": installed,
        "installed_package_is_verified_wheel": identity == "MATCH" and bool(assessment.get("versions_ok")),
        "kernel_device_family": kernel_device_family,
        "kernel_finite": kernel_finite,
        "node": node,
        "preflight_pass": preflight_pass,
        "recorded_evidence": recorded_flash_attention_evidence(),
        "recorded_prior_l40s_validation": WHEEL_L40S_KERNEL_VALIDATED,
        "recorded_wheel_build_pass": RECORDED_WHEEL_BUILD_PASS,
        "slurm_job_id": job_id,
        "static_glibc_is_abi_pass": STATIC_GLIBC_IS_ABI_PASS,
        "status": status,
        "weights_downloaded": False,
        "wheel_extension_sha256": wheel_member,
        "wheel_sha256": assessment.get("wheel_sha256"),
        "wheel_static_glibc_max": FLASH_ATTN_RECORDED_GLIBC_MAX,
    }


def _optional_bool(value: object) -> bool | None:
    if value is True or value is False:
        return value
    return None


def current_artifacts_allow_preflight() -> bool:
    """The measured gollie-l40s extension cannot reach PREFLIGHT_PASS.

    Caller-supplied observations are ignored. The recorded hash, the absent
    L40S kernel result, and ABI_PASS=false are the current artifacts.
    """
    assessment = assess_gollie_installation(
        torch_version=EXPECTED_TORCH,
        transformers_version=EXPECTED_TRANSFORMERS,
        flash_attn_version=FLASH_ATTN_VERSION,
        black_version=EXPECTED_BLACK,
        installed_extension_sha256=INSTALLED_EXTENSION_SHA256,
        wheel_sha256=FLASH_ATTN_WHEEL_SHA256,
        wheel_extension_sha256=None,
    )
    report = build_preflight_report(
        assessment,
        job_id="recorded-evidence",
        node=None,
        kernel_finite=None,
    )
    return bool(report["preflight_pass"] is True and assessment["abi_pass"] is True)


def j16_rejection_reason(report: dict[str, object], pointer_job_id: str) -> str | None:
    """Return a rejection, or None when the report itself would be acceptable.

    Download authorization is a separate gate. None here does not authorize J16.
    """
    job_id = report.get("slurm_job_id")
    if not isinstance(job_id, str) or not job_id:
        return "J15 preflight is unidentified; refusing to download"
    if job_id != pointer_job_id:
        return "J15 report belongs to another JOBID; refusing to download"
    if report.get("status") != "PREFLIGHT_PASS" or report.get("preflight_pass") is not True:
        return f"J15 status is {report.get('status')}; refusing to download"
    if report.get("binary_identity") != "MATCH":
        return "J15 binary identity is not MATCH; refusing to download"
    if report.get("wheel_sha256") != FLASH_ATTN_WHEEL_SHA256:
        return "J15 wheel SHA-256 is not the pinned wheel; refusing to download"
    if report.get("abi_pass") is not True or report.get("abi_verdict") != "pass":
        return "ABI_PASS is not recorded; refusing to download"
    if report.get("build_pass") is not True:
        return "BUILD_PASS is not recorded for this execution; refusing to download"
    if report.get("weights_downloaded") is not False:
        return "J15 report does not record weights_downloaded=false"
    if report.get("node") in (None, ""):
        return "J15 preflight is incomplete; refusing to download"
    installed = report.get("installed_extension_sha256")
    wheel_member = report.get("wheel_extension_sha256")
    if installed == INSTALLED_EXTENSION_SHA256:
        return "J15 report records the pre-correction extension; refusing to download"
    if not isinstance(installed, str) or installed != wheel_member:
        return "J15 extension identity is not the pinned wheel member; refusing to download"
    if report.get("kernel_finite") is not True or report.get("kernel_device_family") != "L40S":
        return "J15 kernel was not observed on L40S; refusing to download"
    if (
        report.get("dynamic_dependencies_resolved") is not True
        or report.get("glibc_requirement_satisfied") is not True
    ):
        return "ABI observations are incomplete; refusing to download"
    if report.get("flash_attn") != FLASH_ATTN_VERSION:
        return "J15 flash-attn version is not the pinned version; refusing to download"
    return None


def download_authorized() -> bool:
    return GOLLIE_J16_DOWNLOAD_AUTHORIZED is True


def legacy_report_path(runs_root: Path) -> Path:
    return runs_root / "gollie" / "preflight_report.json"


def accepted_pointer(runs_root: Path) -> Path:
    return runs_root / "gollie" / "preflight" / "accepted_job_id.txt"


def write_accepted_pointer(runs_root: Path, job_id: str) -> Path:
    _require_job_id(job_id)
    path = require_under_scratch(accepted_pointer(runs_root))
    if path.exists():
        fail(f"refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(job_id + "\n", encoding="utf-8")
    return path


def read_pointer_job_id(runs_root: Path) -> str:
    pointer = accepted_pointer(runs_root)
    if not pointer.is_file():
        fail("J15 preflight is unidentified; refusing to download")
    return pointer.read_text(encoding="utf-8").strip()


def probe_installed_environment() -> dict[str, object]:
    """Read the current interpreter and the wheel path. Do not import flash_attn."""
    venv = Path(os.environ["GOLLIE_VENV"])
    wheel_raw = os.environ.get("GOLLIE_FLASH_ATTN_WHEEL")
    return assess_from_files(
        versions=_probe_versions(),
        installed_extension=_find_extension(venv),
        wheel=Path(wheel_raw) if wheel_raw else None,
    )


def cli_exit_code(assessment: dict[str, object]) -> int:
    if assessment["accredited_for_reuse"] is True:
        return 0
    return 2


def assess_from_files(
    *,
    versions: dict[str, str | None],
    installed_extension: Path | None,
    wheel: Path | None,
) -> dict[str, object]:
    installed_hash = sha256_file(installed_extension) if installed_extension and installed_extension.is_file() else None
    wheel_hash = sha256_file(wheel) if wheel and wheel.is_file() else None
    member_hash = wheel_extension_sha256(wheel) if wheel and wheel.is_file() and wheel_hash == FLASH_ATTN_WHEEL_SHA256 else None
    return assess_gollie_installation(
        torch_version=versions.get("torch"),
        transformers_version=versions.get("transformers"),
        flash_attn_version=versions.get("flash_attn"),
        black_version=versions.get("black"),
        installed_extension_sha256=installed_hash,
        wheel_sha256=wheel_hash,
        wheel_extension_sha256=member_hash,
    )


def _probe_versions() -> dict[str, str | None]:
    import importlib.metadata

    found: dict[str, str | None] = {}
    for key, distribution in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("flash_attn", "flash-attn"),
        ("black", "black"),
    ):
        try:
            found[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            found[key] = None
    return found


def _find_extension(venv: Path) -> Path | None:
    matches = list(venv.glob(f"lib/python*/site-packages/flash_attn/{FLASH_ATTN_EXTENSION}"))
    if len(matches) != 1:
        return None
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["--assess"]:
        fail("usage: gollie_environment.py --assess")
    if os.environ.get("GOLLIE_BOOTSTRAP_ALLOW_COMPILE"):
        fail("FlashAttention compilation is forbidden")
    venv = Path(os.environ["GOLLIE_VENV"])
    wheel_raw = os.environ.get("GOLLIE_FLASH_ATTN_WHEEL")
    wheel = Path(wheel_raw) if wheel_raw else None
    assessment = assess_from_files(
        versions=_probe_versions(),
        installed_extension=_find_extension(venv),
        wheel=wheel,
    )
    sys.stdout.write(json.dumps(assessment, sort_keys=True) + "\n")
    return cli_exit_code(assessment)


if __name__ == "__main__":
    raise SystemExit(main())
