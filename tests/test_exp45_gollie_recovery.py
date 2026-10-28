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

    monkeypatch.setattr(j16, "recover_snapshot", explode)
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


def _accepted_j15(runs: Path, job: str = "1351016") -> None:
    report = {
        "abi_pass": True,
        "abi_verdict": "pass",
        "binary_identity": "MATCH",
        "build_pass": True,
        "dynamic_dependencies_resolved": True,
        "flash_attn": FLASH_ATTN_VERSION,
        "glibc_requirement_satisfied": True,
        "import_succeeded": True,
        "installed_extension_sha256": "f" * 64,
        "kernel_device_family": "L40S",
        "kernel_finite": True,
        "node": "r1n1",
        "preflight_pass": True,
        "slurm_job_id": job,
        "status": "PREFLIGHT_PASS",
        "weights_downloaded": False,
        "wheel_extension_sha256": "f" * 64,
        "wheel_sha256": FLASH_ATTN_WHEEL_SHA256,
    }
    path = runs / "gollie" / "preflight" / "jobs" / job / "preflight_report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    (runs / "gollie" / "preflight" / "accepted_job_id.txt").write_text(job + "\n", encoding="utf-8")


def _isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    scratch = tmp_path / "scratch"
    runs = scratch / "runs"
    venv = scratch / "venvs" / "gollie-isolated"
    original = scratch / "venvs" / "gollie-l40s"
    cache = scratch / "hf-cache" / "hub"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(Path(sys.executable))
    original.mkdir(parents=True)
    cache.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("GOLLIE_VENV", str(venv))
    monkeypatch.setenv("GOLLIE_ORIGINAL_VENV", str(original))
    monkeypatch.setenv("SLURM_JOB_ID", "1351100")
    monkeypatch.delenv("GOLLIE_J16_DOWNLOAD_AUTHORIZED", raising=False)
    monkeypatch.setattr(
        j16.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": j16.SCRATCH_FREE_BYTES})(),
    )
    return scratch, runs


def _complete_snapshot(cache: Path) -> Path:
    snapshot_dir = (
        cache
        / "models--HiTZ--GoLLIE-7B"
        / "snapshots"
        / j16.GOLLIE_REVISION
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot_dir / "modeling_flash_llama.py").write_bytes(b"x" * j16.MODELING_SIZE)
    (snapshot_dir / "model.safetensors").write_bytes(b"weights")
    return snapshot_dir


def test_j16_uses_the_isolated_venv_and_records_a_complete_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch, runs = _isolated_runtime(tmp_path, monkeypatch)
    _accepted_j15(runs)
    snapshot_dir = _complete_snapshot(scratch / "hf-cache" / "hub")
    calls: list[str] = []

    def local_hit() -> Path:
        calls.append("local")
        return snapshot_dir

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("weights were downloaded")

    monkeypatch.setattr(j16, "_local_complete_snapshot", local_hit)
    monkeypatch.setattr(j16, "probe_repository_access", explode)
    monkeypatch.setattr(j16, "recover_snapshot", explode)
    monkeypatch.setenv("GOLLIE_J16_DOWNLOAD_AUTHORIZED", "1")
    assert j16.main() == 0
    assert calls == ["local"]
    pointer = runs / "gollie" / "download" / "accepted_job_id.txt"
    report = json.loads(
        (runs / "gollie" / "download" / "jobs" / "1351100" / "download_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert pointer.read_text(encoding="utf-8") == "1351100\n"
    assert report["status"] == "DOWNLOAD_PASS"
    assert report["checkpoint_complete"] is True
    assert report["revision"] == j16.GOLLIE_REVISION
    assert report["weight_files"] == ["model.safetensors"]
    assert report["venv"].endswith("/venvs/gollie-isolated")
    assert GOLLIE_J16_DOWNLOAD_AUTHORIZED is False
    with pytest.raises(SystemExit):
        j16.main()
    assert j16.assert_j17_prerequisites(runs)["status"] == "DOWNLOAD_PASS"


def test_j16_checks_access_before_a_missing_snapshot_and_rejects_bad_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scratch, runs = _isolated_runtime(tmp_path, monkeypatch)
    _accepted_j15(runs)
    snapshot_dir = _complete_snapshot(scratch / "hf-cache" / "hub")
    order: list[str] = []

    monkeypatch.setattr(j16, "_local_complete_snapshot", lambda: None)
    monkeypatch.setattr(
        j16,
        "probe_repository_access",
        lambda repo, revision: order.append(f"access:{repo}@{revision}"),
    )

    def fetch() -> Path:
        order.append("download")
        return snapshot_dir

    monkeypatch.setattr(j16, "recover_snapshot", fetch)
    monkeypatch.setenv("GOLLIE_J16_DOWNLOAD_AUTHORIZED", "1")
    assert j16.main() == 0
    assert order == [f"access:{j16.GOLLIE_REPO}@{j16.GOLLIE_REVISION}", "download"]

    _scratch, original_runs = _isolated_runtime(tmp_path / "original", monkeypatch)
    _accepted_j15(original_runs)
    monkeypatch.setenv("GOLLIE_VENV", str(tmp_path / "original" / "scratch" / "venvs" / "gollie-l40s"))
    monkeypatch.setenv("GOLLIE_J16_DOWNLOAD_AUTHORIZED", "1")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("weights were downloaded")

    monkeypatch.setattr(j16, "recover_snapshot", explode)
    monkeypatch.setattr(j16, "probe_repository_access", explode)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        j16.main()
    assert "gollie-isolated" in capsys.readouterr().out

    space, space_runs = _isolated_runtime(tmp_path / "space", monkeypatch)
    _accepted_j15(space_runs)
    monkeypatch.setenv("GOLLIE_J16_DOWNLOAD_AUTHORIZED", "1")
    monkeypatch.setattr(
        j16.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": j16.SCRATCH_FREE_BYTES - 1})(),
    )
    monkeypatch.setattr(j16, "_local_complete_snapshot", lambda: None)
    with pytest.raises(SystemExit):
        j16.main()
    assert "free space" in capsys.readouterr().out

    status = j16._access_status(type("HttpError", (), {"response": type("Response", (), {"status_code": 401})()})())
    assert status == 401
    incomplete = space / "hf-cache" / "hub" / "partial"
    incomplete.mkdir(parents=True)
    (incomplete / "config.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        j16.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": j16.SCRATCH_FREE_BYTES})(),
    )
    with pytest.raises(SystemExit):
        j16.assert_snapshot_complete(incomplete)
    assert "modeling module" in capsys.readouterr().out
    monkeypatch.setenv("GOLLIE_VENV", str(space / "venvs" / "gollie-l40s"))
    with pytest.raises(SystemExit):
        j16.assert_j17_prerequisites(space_runs)
    assert "gollie-isolated" in capsys.readouterr().out


def _revision_snapshot(cache: Path) -> Path:
    """The directory J15 leaves behind: the modeling file and nothing else."""
    path = cache / "models--HiTZ--GoLLIE-7B" / "snapshots" / j16.GOLLIE_REVISION
    path.mkdir(parents=True)
    (path / j16.MODELING_FILE).write_bytes(b"x" * j16.MODELING_SIZE)
    return path


def _grant_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOLLIE_J16_DOWNLOAD_AUTHORIZED", "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


def _download_report(runs: Path) -> dict[str, object]:
    return json.loads(
        (runs / "gollie" / "download" / "jobs" / "1351100" / "download_report.json").read_text(
            encoding="utf-8"
        )
    )


def test_j16_completes_a_snapshot_that_only_has_the_modeling_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch, runs = _isolated_runtime(tmp_path, monkeypatch)
    _accepted_j15(runs)
    original = scratch / "venvs" / "gollie-l40s" / "untouched"
    original.write_text("keep\n", encoding="utf-8")
    partial = _revision_snapshot(scratch / "hf-cache" / "hub")
    modes: list[bool] = []

    def hub(repo: str, revision: str, *, local_files_only: bool) -> Path:
        assert repo == j16.GOLLIE_REPO
        assert revision == j16.GOLLIE_REVISION
        modes.append(local_files_only)
        if not local_files_only:
            (partial / "config.json").write_text("{}\n", encoding="utf-8")
            (partial / "model.safetensors").write_bytes(b"weights")
        return partial

    monkeypatch.setattr(j16, "snapshot", hub)
    monkeypatch.setattr(
        j16,
        "probe_repository_access",
        lambda repo, revision: modes.append(repo == j16.GOLLIE_REPO and revision == j16.GOLLIE_REVISION),
    )
    _grant_download(monkeypatch)
    assert j16.main() == 0
    assert modes == [True, True, False]
    report = _download_report(runs)
    assert report["status"] == "DOWNLOAD_PASS"
    assert report["checkpoint_complete"] is True
    assert report["weight_files"] == ["model.safetensors"]
    pointer = runs / "gollie" / "download" / "accepted_job_id.txt"
    assert pointer.read_text(encoding="utf-8") == "1351100\n"
    recorded = json.loads((runs / "gollie" / "snapshot.json").read_text(encoding="utf-8"))
    assert recorded["source"] == "downloaded"
    assert original.read_text(encoding="utf-8") == "keep\n"
    before = pointer.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        j16.main()
    assert pointer.read_text(encoding="utf-8") == before
    assert j16.assert_j17_prerequisites(runs)["status"] == "DOWNLOAD_PASS"

    indexed, indexed_runs = _isolated_runtime(tmp_path / "indexed", monkeypatch)
    _accepted_j15(indexed_runs)
    shard_dir = _revision_snapshot(indexed / "hf-cache" / "hub")
    (shard_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (shard_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    shard_modes: list[bool] = []

    def hub_index(repo: str, revision: str, *, local_files_only: bool) -> Path:
        assert repo == j16.GOLLIE_REPO and revision == j16.GOLLIE_REVISION
        shard_modes.append(local_files_only)
        if not local_files_only:
            (shard_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard")
        return shard_dir

    monkeypatch.setattr(j16, "snapshot", hub_index)
    monkeypatch.setattr(j16, "probe_repository_access", lambda _repo, _revision: None)
    _grant_download(monkeypatch)
    assert j16.main() == 0
    assert shard_modes == [True, False]
    assert _download_report(indexed_runs)["weight_files"] == ["model-00001-of-00002.safetensors"]

    cached, cached_runs = _isolated_runtime(tmp_path / "cached", monkeypatch)
    _accepted_j15(cached_runs)
    repo = cached / "hf-cache" / "hub" / "models--HiTZ--GoLLIE-7B"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "config").write_text("{}\n", encoding="utf-8")
    (repo / "blobs" / "weights").write_bytes(b"weights")
    complete = repo / "snapshots" / j16.GOLLIE_REVISION
    complete.mkdir(parents=True)
    (complete / "config.json").symlink_to(Path("../../blobs/config"))
    (complete / "model.safetensors").symlink_to(Path("../../blobs/weights"))
    (complete / j16.MODELING_FILE).write_bytes(b"x" * j16.MODELING_SIZE)
    cache_modes: list[bool] = []

    def hub_cache(repo_id: str, revision: str, *, local_files_only: bool) -> Path:
        assert repo_id == j16.GOLLIE_REPO and revision == j16.GOLLIE_REVISION
        cache_modes.append(local_files_only)
        if not local_files_only:
            raise AssertionError("complete snapshot was downloaded")
        return complete

    monkeypatch.setattr(j16, "snapshot", hub_cache)
    monkeypatch.setattr(
        j16,
        "probe_repository_access",
        lambda *_args: (_ for _ in ()).throw(AssertionError("metadata was probed")),
    )
    _grant_download(monkeypatch)
    assert j16.main() == 0
    assert cache_modes == [True]
    cached_record = json.loads((cached_runs / "gollie" / "snapshot.json").read_text(encoding="utf-8"))
    assert cached_record["source"] == "cache"
    assert _download_report(cached_runs)["weight_files"] == ["model.safetensors"]


def test_j16_keeps_a_partial_or_unsafe_snapshot_from_passing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_probe = j16.probe_repository_access
    scratch, runs = _isolated_runtime(tmp_path, monkeypatch)
    _accepted_j15(runs)
    original = scratch / "venvs" / "gollie-l40s" / "untouched"
    original.write_text("keep\n", encoding="utf-8")
    partial = _revision_snapshot(scratch / "hf-cache" / "hub")
    modes: list[bool] = []

    def unchanged(repo: str, revision: str, *, local_files_only: bool) -> Path:
        assert repo == j16.GOLLIE_REPO and revision == j16.GOLLIE_REVISION
        modes.append(local_files_only)
        return partial

    monkeypatch.setattr(j16, "snapshot", unchanged)
    monkeypatch.setattr(j16, "probe_repository_access", lambda _repo, _revision: None)
    _grant_download(monkeypatch)
    with pytest.raises(SystemExit):
        j16.main()
    assert "snapshot config.json is absent" in capsys.readouterr().out
    assert modes == [True, False]
    assert not (runs / "gollie" / "download" / "accepted_job_id.txt").exists()
    with pytest.raises(SystemExit):
        j16.assert_j17_prerequisites(runs)
    assert "unidentified" in capsys.readouterr().out
    assert original.read_text(encoding="utf-8") == "keep\n"

    offline, offline_runs = _isolated_runtime(tmp_path / "offline", monkeypatch)
    _accepted_j15(offline_runs)
    offline_partial = _revision_snapshot(offline / "hf-cache" / "hub")
    offline_modes: list[bool] = []

    def offline_hub(repo: str, revision: str, *, local_files_only: bool) -> Path:
        assert repo == j16.GOLLIE_REPO and revision == j16.GOLLIE_REVISION
        offline_modes.append(local_files_only)
        if not local_files_only:
            raise AssertionError("offline snapshot was downloaded")
        return offline_partial

    monkeypatch.setattr(j16, "snapshot", offline_hub)
    monkeypatch.setattr(j16, "probe_repository_access", real_probe)
    monkeypatch.setenv("GOLLIE_J16_DOWNLOAD_AUTHORIZED", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(SystemExit):
        j16.main()
    assert "HF_HUB_OFFLINE" in capsys.readouterr().out
    assert offline_modes == [True]
    assert not (offline_runs / "gollie" / "download" / "accepted_job_id.txt").exists()

    def refused(message: str, root: Path, build) -> None:
        refused_scratch, refused_runs = _isolated_runtime(root, monkeypatch)
        _accepted_j15(refused_runs)
        returned = build(refused_scratch / "hf-cache" / "hub")
        called: list[bool] = []

        def hub(repo: str, revision: str, *, local_files_only: bool) -> Path:
            assert repo == j16.GOLLIE_REPO and revision == j16.GOLLIE_REVISION
            called.append(local_files_only)
            if not local_files_only:
                raise AssertionError("unsafe snapshot was recovered")
            return returned

        monkeypatch.setattr(j16, "snapshot", hub)
        monkeypatch.setattr(
            j16,
            "probe_repository_access",
            lambda *_args: (_ for _ in ()).throw(AssertionError("unsafe snapshot probed the hub")),
        )
        _grant_download(monkeypatch)
        with pytest.raises(SystemExit):
            j16.main()
        assert message in capsys.readouterr().out
        assert called == [True]
        assert not (refused_runs / "gollie" / "download" / "accepted_job_id.txt").exists()
        assert (refused_scratch / "venvs" / "gollie-l40s").is_dir()

    def build_outside(_cache: Path) -> Path:
        outside = tmp_path / "outside" / j16.GOLLIE_REVISION
        outside.mkdir(parents=True)
        return outside

    def build_wrong(cache: Path) -> Path:
        wrong = cache / "wrong-revision"
        wrong.mkdir()
        return wrong

    def build_malformed(cache: Path) -> Path:
        malformed = _revision_snapshot(cache)
        (malformed / "config.json").write_text("{}\n", encoding="utf-8")
        (malformed / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer": "../model.safetensors"}}),
            encoding="utf-8",
        )
        return malformed

    def build_link(cache: Path) -> Path:
        linked = _revision_snapshot(cache)
        (linked / "config.json").symlink_to(tmp_path / "outside-config")
        return linked

    def build_short(cache: Path) -> Path:
        corrupt = _revision_snapshot(cache)
        (corrupt / "config.json").write_text("{}\n", encoding="utf-8")
        (corrupt / "model.safetensors").write_bytes(b"weights")
        (corrupt / j16.MODELING_FILE).write_bytes(b"short")
        return corrupt

    refused("outside the scratch", tmp_path / "outside-run", build_outside)
    refused("snapshot revision", tmp_path / "revision-run", build_wrong)
    refused("weight index is malformed", tmp_path / "index-run", build_malformed)
    refused("improper symlink", tmp_path / "link-run", build_link)
    refused("not the pinned file", tmp_path / "size-run", build_short)

    exploded, exploded_runs = _isolated_runtime(tmp_path / "exploded", monkeypatch)
    _accepted_j15(exploded_runs)
    _revision_snapshot(exploded / "hf-cache" / "hub")

    def explode(repo: str, revision: str, *, local_files_only: bool) -> Path:
        assert repo == j16.GOLLIE_REPO and revision == j16.GOLLIE_REVISION
        assert local_files_only is True
        raise RuntimeError("hub exploded")

    monkeypatch.setattr(j16, "snapshot", explode)
    monkeypatch.setattr(
        j16,
        "probe_repository_access",
        lambda *_args: (_ for _ in ()).throw(AssertionError("an unexpected error became a download")),
    )
    _grant_download(monkeypatch)
    with pytest.raises(RuntimeError, match="hub exploded"):
        j16.main()
    assert not (exploded_runs / "gollie" / "download" / "accepted_job_id.txt").exists()


def test_j16_and_j17_sbatch_select_the_isolated_venv_without_a_global_grant() -> None:
    j16_text = (ROOT / "jobs" / "cesvima" / "j16_gollie_download.sbatch").read_text(encoding="utf-8")
    j17_text = (ROOT / "jobs" / "cesvima" / "j17_gollie_native_smoke.sbatch").read_text(encoding="utf-8")
    smoke = (SCRIPTS / "j17_gollie_native_smoke.sh").read_text(encoding="utf-8")
    for text in (j16_text, j17_text):
        source = text.index("source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh")
        assert source < text.index('export GOLLIE_VENV="${GOLLIE_ISOLATED_VENV}"')
        assert "gollie-l40s" in text
        assert "submit_train_first.sh" not in text
    assert "GOLLIE_J16_DOWNLOAD_AUTHORIZED=1" not in j16_text
    assert "GOLLIE_J17_SMOKE_AUTHORIZED=1" not in j17_text
    assert 'saved_download="${GOLLIE_J16_DOWNLOAD_AUTHORIZED:-}"' in j16_text
    assert source_before_restore(j16_text, 'export GOLLIE_J16_DOWNLOAD_AUTHORIZED="${saved_download}"')
    assert "unset HF_HUB_OFFLINE" in j16_text
    assert "HF_HUB_OFFLINE=1" in j17_text
    assert "official_re_documents.jsonl" in smoke
    assert "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9" in smoke
    assert "assert_j17_prerequisites" in smoke
    assert GOLLIE_J16_DOWNLOAD_AUTHORIZED is False
    assert GOLLIE_J17_SMOKE_AUTHORIZED is False
    pin = (SCRIPTS / "flash_attn_pin.py").read_text(encoding="utf-8")
    assert "GOLLIE_J16_DOWNLOAD_AUTHORIZED = False" in pin
    assert "GOLLIE_J17_SMOKE_AUTHORIZED = False" in pin


def source_before_restore(text: str, restore: str) -> bool:
    return text.index("source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh") < text.index(restore)
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

