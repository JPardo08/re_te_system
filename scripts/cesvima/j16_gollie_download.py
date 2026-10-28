"""J16: download GoLLIE-7B only after the accepted J15 report and an explicit grant.

The module flag GOLLIE_J16_DOWNLOAD_AUTHORIZED stays false. This process
downloads only when the environment variable of the same name is 1, the
accepted preflight passes, and the interpreter is gollie-isolated. J17 reads
the per-JOBID record written after the snapshot integrity check.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from cesvima_common import fail, read_json, require_under_scratch, write_json
from gollie_environment import _require_job_id, accepted_pointer, j16_rejection_reason, job_report_dir
from hf_snapshot import snapshot


GOLLIE_REPO = "HiTZ/GoLLIE-7B"
GOLLIE_REVISION = "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9"
MODELING_FILE = "modeling_flash_llama.py"
MODELING_SIZE = 46523
# Same operational SCRATCH floor already used by J15. It is not a measured
# size of the 7B checkpoint.
SCRATCH_FREE_BYTES = 40 * 1024 * 1024 * 1024


def load_accepted_report(runs_root: Path) -> dict:
    pointer = accepted_pointer(runs_root)
    if not pointer.is_file():
        fail("J15 preflight is unidentified; refusing to download")
    job_id = pointer.read_text(encoding="utf-8").strip()
    report_path = job_report_dir(runs_root, job_id) / "preflight_report.json"
    if not report_path.is_file():
        fail("J15 preflight report is missing")
    report = read_json(report_path)
    reason = j16_rejection_reason(report, job_id)
    if reason is not None:
        fail(reason)
    return report


def download_runtime_authorized() -> bool:
    """Explicit job grant. The module constant is not this grant."""
    return os.environ.get("GOLLIE_J16_DOWNLOAD_AUTHORIZED") == "1"


def isolated_venv() -> Path:
    raw = os.environ.get("GOLLIE_VENV", "")
    if not raw:
        fail("GOLLIE_VENV is required")
    venv = Path(raw)
    if venv.is_symlink() or venv.name != "gollie-isolated" or venv.parent.name != "venvs":
        fail("refusing to use a venv other than gollie-isolated")
    original = os.environ.get("GOLLIE_ORIGINAL_VENV", "")
    if original and Path(original).name == "gollie-l40s" and venv.resolve() == Path(original).resolve():
        fail("isolated venv collapses onto gollie-l40s")
    if "gollie-l40s" in venv.parts:
        fail("refusing to use gollie-l40s")
    python = venv / "bin" / "python"
    if not python.is_file():
        fail("isolated venv python is absent")
    return venv


def download_job_dir(runs_root: Path, job_id: str) -> Path:
    _require_job_id(job_id)
    return runs_root / "gollie" / "download" / "jobs" / job_id


def download_pointer(runs_root: Path) -> Path:
    return runs_root / "gollie" / "download" / "accepted_job_id.txt"


def _inside(root: Path, path: Path) -> bool:
    resolved = path.resolve()
    base = root.resolve()
    return resolved == base or base in resolved.parents


def assert_cache_writable() -> Path:
    raw = os.environ.get("HF_HUB_CACHE", "")
    scratch_raw = os.environ.get("SCRATCH_ROOT", "")
    if not raw or not scratch_raw:
        fail("HF_HUB_CACHE and SCRATCH_ROOT are required")
    cache = Path(raw)
    scratch = Path(scratch_raw)
    if cache.is_symlink() or not _inside(scratch, cache):
        fail("HF_HUB_CACHE is outside SCRATCH_ROOT")
    parent = cache if cache.is_dir() else cache.parent
    if parent.is_symlink() or not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        fail("HF_HUB_CACHE is not writable")
    return cache.resolve()


def assert_download_space() -> None:
    scratch = Path(os.environ["SCRATCH_ROOT"])
    usage = shutil.disk_usage(scratch)
    if usage.free < SCRATCH_FREE_BYTES:
        fail(f"SCRATCH free space {usage.free} bytes is below {SCRATCH_FREE_BYTES}")


def _access_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def probe_repository_access(repo_id: str, revision: str) -> None:
    """Read repository metadata. This does not download weight files."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        fail("HF_HUB_OFFLINE=1 blocks the weight repository access check")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        fail(f"huggingface_hub is not importable: {exc}")
    try:
        info = HfApi().model_info(repo_id, revision=revision)
    except Exception as exc:
        status = _access_status(exc)
        if status in {401, 403}:
            fail(f"Hugging Face access to {repo_id}@{revision} returned {status}")
        fail(f"could not read repository metadata for {repo_id}@{revision}: {exc}")
    observed = getattr(info, "sha", None)
    if observed != revision:
        fail(f"repository metadata revision {observed} is not {revision}")


def _repo_cache(cache: Path) -> Path:
    return (cache / ("models--" + GOLLIE_REPO.replace("/", "--"))).resolve()


def _local_entry_not_found() -> type[BaseException]:
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        fail(f"huggingface_hub is not importable: {exc}")
    return LocalEntryNotFoundError


def _accepted_blob_target(path: Path, cache: Path, scratch: Path) -> Path:
    """Hugging Face stores snapshot names as symlinks to blobs in the same repo cache."""
    try:
        target = path.resolve()
    except OSError:
        fail(f"{path.name} is an improper symlink")
    repo = _repo_cache(cache)
    if (
        not target.is_file()
        or not _inside(repo, target)
        or not _inside(cache, target)
        or not _inside(scratch, target)
    ):
        fail(f"{path.name} is an improper symlink")
    return target


def _content_gap(
    path: Path,
    *,
    snapshot: Path,
    cache: Path,
    scratch: Path,
    absent: str,
) -> str | None:
    """Return absent when a safe entry is missing or empty. fail() on an unsafe entry."""
    if path.is_symlink():
        target = _accepted_blob_target(path, cache, scratch)
        if target.stat().st_size == 0:
            return absent
        return None
    if not path.is_file() or path.stat().st_size == 0:
        return absent
    resolved = path.resolve()
    if not _inside(snapshot, resolved) or not _inside(scratch, resolved):
        fail(f"{path.name} escapes the snapshot")
    return None


def _weight_entries(snapshot_dir: Path, cache: Path, scratch: Path) -> tuple[list[Path] | None, str | None]:
    index = snapshot_dir / "model.safetensors.index.json"
    if index.is_symlink() or index.is_file():
        if index.is_symlink():
            _accepted_blob_target(index, cache, scratch)
        elif not _inside(snapshot_dir, index.resolve()) or not _inside(scratch, index.resolve()):
            fail("weight index escapes the snapshot")
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            fail("weight index is malformed")
        if not isinstance(payload, dict) or not isinstance(payload.get("weight_map"), dict):
            fail("weight index is malformed")
        mapping = payload["weight_map"]
        if not mapping:
            fail("weight index is empty")
        names: list[str] = []
        for name in mapping.values():
            if not isinstance(name, str) or name != Path(name).name or name in {".", ".."}:
                fail("weight index is malformed")
            names.append(name)
        paths = [snapshot_dir / name for name in sorted(set(names))]
    else:
        paths = [
            path
            for path in snapshot_dir.iterdir()
            if path.is_file()
            and (path.name.endswith(".safetensors") or path.name.startswith("pytorch_model"))
        ]
        if not paths:
            return None, "snapshot has no weight files"
    for path in paths:
        gap = _content_gap(
            path,
            snapshot=snapshot_dir,
            cache=cache,
            scratch=scratch,
            absent=f"weight file is absent or empty: {path.name}",
        )
        if gap is not None:
            return None, gap
    return paths, None


def inspect_snapshot(snapshot_dir: Path) -> tuple[dict[str, object] | None, str | None]:
    """Return the weight record, or a recoverable absence reason.

    Paths outside SCRATCH, improper symlinks, a wrong tree, and a malformed
    index call fail() and must not be treated as a cache miss.
    """
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        fail("snapshot is not a real directory")
    cache = assert_cache_writable()
    scratch = Path(os.environ["SCRATCH_ROOT"]).resolve()
    resolved = snapshot_dir.resolve()
    if not _inside(cache, resolved) or not _inside(scratch, resolved):
        fail("snapshot is outside the scratch Hugging Face cache")
    config_gap = _content_gap(
        snapshot_dir / "config.json",
        snapshot=resolved,
        cache=cache,
        scratch=scratch,
        absent="snapshot config.json is absent",
    )
    if config_gap is not None:
        return None, config_gap
    modeling = snapshot_dir / MODELING_FILE
    modeling_gap = _content_gap(
        modeling,
        snapshot=resolved,
        cache=cache,
        scratch=scratch,
        absent="snapshot modeling module is absent",
    )
    if modeling_gap is not None:
        return None, modeling_gap
    if modeling.stat().st_size != MODELING_SIZE:
        fail("snapshot modeling module is not the pinned file")
    weights, weight_gap = _weight_entries(resolved, cache, scratch)
    if weight_gap is not None or weights is None:
        return None, weight_gap or "snapshot has no weight files"
    names = []
    total = 0
    for path in weights:
        names.append(path.name)
        total += path.stat().st_size
    return {"bytes": total, "weight_files": names}, None


def assert_snapshot_complete(snapshot_dir: Path) -> dict[str, object]:
    record, reason = inspect_snapshot(snapshot_dir)
    if reason is not None or record is None:
        fail(reason or "snapshot is incomplete")
    return record


def _require_revision_directory(path: Path) -> None:
    if path.name != GOLLIE_REVISION:
        fail(f"snapshot revision {path.name} is not {GOLLIE_REVISION}")


def _local_complete_snapshot() -> Path | None:
    """Return a complete local snapshot.

    A missing directory and a safe but incomplete directory both return None so
    the caller can recover them. Any other error, including an unsafe tree, is
    not converted into a cache miss.
    """
    missing = _local_entry_not_found()
    try:
        path = snapshot(GOLLIE_REPO, GOLLIE_REVISION, local_files_only=True)
    except missing:
        return None
    _require_revision_directory(path)
    _record, reason = inspect_snapshot(path)
    if reason is not None:
        return None
    return path


def recover_snapshot() -> Path:
    """Download the pinned revision.

    cache_or_download() returns an existing snapshot directory without checking
    its files, which is how the partial J15 cache skipped the download. This
    call asks the hub for the revision with local_files_only disabled. The
    caller still has to validate the result before writing DOWNLOAD_PASS.
    """
    missing = _local_entry_not_found()
    try:
        path = snapshot(GOLLIE_REPO, GOLLIE_REVISION, local_files_only=False)
    except missing as exc:
        fail(f"could not complete the GoLLIE snapshot: {exc}")
    _require_revision_directory(path)
    return path


def _write_download_record(runs: Path, job_id: str, payload: dict[str, object]) -> None:
    report = require_under_scratch(download_job_dir(runs, job_id) / "download_report.json")
    pointer = require_under_scratch(download_pointer(runs))
    if report.exists() or pointer.exists():
        fail(f"refusing to overwrite existing evidence: {report if report.exists() else pointer}")
    write_json(report, payload)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(job_id + "\n", encoding="utf-8")


def assert_j17_prerequisites(runs_root: Path) -> dict[str, object]:
    """J17 stays closed until this J16 record says the checkpoint is complete."""
    isolated_venv()
    pointer = download_pointer(runs_root)
    if not pointer.is_file():
        fail("J16 download is unidentified; refusing the native smoke")
    job_id = pointer.read_text(encoding="utf-8").strip()
    report_path = download_job_dir(runs_root, job_id) / "download_report.json"
    if not report_path.is_file():
        fail("J16 download report is missing")
    report = read_json(report_path)
    if report.get("slurm_job_id") != job_id:
        fail("J16 download report belongs to another JOBID")
    if report.get("status") != "DOWNLOAD_PASS" or report.get("checkpoint_complete") is not True:
        fail("J16 download is not a complete checkpoint")
    if report.get("repository") != GOLLIE_REPO or report.get("revision") != GOLLIE_REVISION:
        fail("J16 download revision is not the pinned GoLLIE checkpoint")
    local = report.get("local_path")
    if not isinstance(local, str) or not local:
        fail("J16 download path is absent")
    checked = assert_snapshot_complete(Path(local))
    if checked["weight_files"] != report.get("weight_files"):
        fail("J16 download record does not match the snapshot")
    return report


def main() -> int:
    runs = Path(os.environ["P3_RUNS_ROOT"])
    load_accepted_report(runs)
    if not download_runtime_authorized():
        fail("J16 weight download is not authorized")
    if download_pointer(runs).exists():
        fail("refusing to overwrite existing evidence: download pointer")
    venv = isolated_venv()
    assert_cache_writable()
    assert_download_space()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    _require_job_id(job_id)
    local = _local_complete_snapshot()
    source = "cache"
    if local is None:
        probe_repository_access(GOLLIE_REPO, GOLLIE_REVISION)
        local = recover_snapshot()
        source = "downloaded"
    checked = assert_snapshot_complete(local)
    write_json(
        runs / "gollie" / "snapshot.json",
        {
            "local_path": str(local.resolve()),
            "repository": GOLLIE_REPO,
            "revision": GOLLIE_REVISION,
            "source": source,
        },
    )
    _write_download_record(
        runs,
        job_id,
        {
            "bytes": checked["bytes"],
            "checkpoint_complete": True,
            "local_path": str(local.resolve()),
            "repository": GOLLIE_REPO,
            "revision": GOLLIE_REVISION,
            "slurm_job_id": job_id,
            "status": "DOWNLOAD_PASS",
            "venv": str(venv),
            "weight_files": checked["weight_files"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
