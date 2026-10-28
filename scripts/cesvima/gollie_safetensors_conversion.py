"""Convert the indexed GoLLIE pytorch shards to Safetensors.

The GPU runtime stays on torch 2.5.1 and must not open the .bin shards.
This process runs in a separate CPU interpreter at torch >= 2.6. It reads
only the files named by pytorch_model.bin.index.json, using
torch.load(..., map_location="cpu", weights_only=True). The origin snapshot
and the Hugging Face cache are not written.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from cesvima_common import fail, read_json, require_under_scratch, sha256_file, write_json
from gollie_environment import _require_job_id
from j16_gollie_download import GOLLIE_REPO, GOLLIE_REVISION, MODELING_FILE, MODELING_SIZE, download_job_dir, download_pointer


PYTORCH_INDEX = "pytorch_model.bin.index.json"
SAFETENSORS_INDEX = "model.safetensors.index.json"
# Accredited total size of the two shards named by the pytorch index.
INDEXED_SHARD_BYTES = 13_477_726_551
MINIMUM_TORCH = (2, 6)
REQUIRED_COMPANIONS = (MODELING_FILE, "config.json", "tokenizer_config.json")


def torch_version_at_least(version: str, minimum: tuple[int, int] = MINIMUM_TORCH) -> bool:
    numeric = version.split("+", 1)[0]
    pieces: list[int] = []
    for piece in numeric.split("."):
        if not piece.isdigit():
            break
        pieces.append(int(piece))
    while len(pieces) < 2:
        pieces.append(0)
    return tuple(pieces[:2]) >= minimum


def conversion_pointer(runs_root: Path) -> Path:
    return runs_root / "gollie" / "conversion" / "accepted_job_id.txt"


def conversion_report_path(runs_root: Path, job_id: str) -> Path:
    _require_job_id(job_id)
    return runs_root / "gollie" / "conversion" / "jobs" / job_id / "conversion_report.json"


def derived_snapshot_dir(models_root: Path) -> Path:
    return models_root / "gollie" / f"hitz-gollie-7b-{GOLLIE_REVISION}-safetensors"


def _inside(root: Path, path: Path) -> bool:
    resolved = path.resolve()
    base = root.resolve()
    return resolved == base or base in resolved.parents


def _runtime_prefix_name() -> str:
    return Path(sys.prefix).name


def _running_torch_version() -> str:
    import torch

    return str(torch.__version__)


def assert_cpu_runtime() -> str:
    prefix = _runtime_prefix_name()
    if prefix in {"gollie-isolated", "gollie-l40s"}:
        fail("refusing to load pytorch bins in the GPU runtime")
    version = _running_torch_version()
    if not torch_version_at_least(version):
        fail(f"PyTorch {version} is below 2.6; refusing weights_only load")
    return version


def _load_shard(path: Path) -> dict:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        fail(f"shard {path.name} is not a weights_only state dict: {exc}")
    if not isinstance(payload, dict) or not payload:
        fail(f"shard {path.name} is not a tensor state dict")
    for name, value in payload.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            fail(f"shard entry is not a tensor: {name}")
    return payload


def _storage_key(tensor: object) -> tuple[int, int]:
    storage = tensor.untyped_storage()  # type: ignore[attr-defined]
    return (storage.data_ptr(), storage.nbytes())


def _covers_storage(tensor: object) -> bool:
    if tensor.numel() == 0:  # type: ignore[attr-defined]
        return False
    storage = tensor.untyped_storage()  # type: ignore[attr-defined]
    return tensor.data_ptr() == storage.data_ptr() and tensor.numel() * tensor.element_size() == storage.nbytes()  # type: ignore[attr-defined]


def materialize_independent_tensors(tensors: dict) -> tuple[dict, list[list[str]]]:
    """Clone shared complete tensors. Fail when a partial view cannot be preserved."""
    import torch

    groups: dict[tuple[int, int], list[str]] = {}
    for name, tensor in tensors.items():
        if tensor.layout != torch.strided or tensor.numel() == 0:
            fail(f"tensor {name} is not a dense strided tensor")
        groups.setdefault(_storage_key(tensor), []).append(name)
    shared: list[list[str]] = []
    for names in groups.values():
        if len(names) < 2:
            continue
        members = [tensors[name] for name in names]
        if not all(_covers_storage(item) for item in members):
            fail("shared storage cannot be preserved: " + ", ".join(sorted(names)))
        reference = members[0]
        for item in members[1:]:
            if item.dtype != reference.dtype or item.shape != reference.shape or not torch.equal(item, reference):
                fail("shared storage cannot be preserved: " + ", ".join(sorted(names)))
        shared.append(sorted(names))
    independent = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in tensors.items()
    }
    return independent, shared


def _safetensors_name(bin_name: str) -> str:
    if not bin_name.endswith(".bin") or bin_name != Path(bin_name).name:
        fail(f"indexed shard name is not a local bin file: {bin_name}")
    return bin_name[: -len(".bin")] + ".safetensors"


def _read_pytorch_index(origin: Path, scratch: Path) -> dict[str, str]:
    index_path = origin / PYTORCH_INDEX
    if index_path.is_symlink():
        target = index_path.resolve()
        if not target.is_file() or not _inside(scratch, target):
            fail("pytorch index escapes SCRATCH")
    elif not index_path.is_file() or not _inside(origin, index_path.resolve()):
        fail("pytorch index is absent")
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        fail("pytorch index is malformed")
    mapping = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        fail("pytorch index is empty")
    cleaned: dict[str, str] = {}
    for tensor_name, filename in mapping.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            fail("pytorch index is malformed")
        if not isinstance(filename, str):
            fail("pytorch index is malformed")
        _safetensors_name(filename)
        cleaned[tensor_name] = filename
    return cleaned


def _repo_cache(origin: Path) -> Path:
    return origin.parent.parent


def _shard_path(origin: Path, scratch: Path, filename: str) -> Path:
    path = origin / filename
    if path.is_symlink():
        target = path.resolve()
        if (
            not target.is_file()
            or target.stat().st_size == 0
            or not _inside(scratch, target)
            or not _inside(_repo_cache(origin), target)
        ):
            fail(f"indexed shard escapes SCRATCH: {filename}")
        return path
    if not path.is_file() or path.stat().st_size == 0 or not _inside(origin, path.resolve()):
        fail(f"indexed shard is absent: {filename}")
    return path


def _is_weight_name(name: str) -> bool:
    return name.endswith((".bin", ".safetensors", ".pt", ".pth")) or name in {
        PYTORCH_INDEX,
        SAFETENSORS_INDEX,
    }


def _file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _assert_destination(origin: Path, destination: Path, scratch: Path) -> None:
    if destination.exists() or destination.is_symlink():
        fail("refusing to overwrite a derived snapshot")
    parent = destination.parent
    if not _inside(scratch, parent):
        fail("derived snapshot is outside SCRATCH")
    cache = os.environ.get("HF_HUB_CACHE", "")
    if cache and _inside(Path(cache), destination):
        fail("refusing to write inside the Hugging Face cache")
    if _inside(origin, destination) or destination.resolve() == origin.resolve():
        fail("refusing to write inside the origin snapshot")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or partial.is_symlink():
        fail("refusing to resume a partial derived snapshot")


def _assert_space(scratch: Path, indexed_bytes: int) -> None:
    free = shutil.disk_usage(scratch).free
    required = indexed_bytes * 2
    if free < required:
        fail(f"SCRATCH free space {free} bytes is below {required}")


def _origin_from_j16(runs: Path) -> Path:
    pointer = download_pointer(runs)
    if not pointer.is_file():
        fail("J16 download is unidentified; refusing the conversion")
    job_id = pointer.read_text(encoding="utf-8").strip()
    report = read_json(download_job_dir(runs, job_id) / "download_report.json")
    if report.get("status") != "DOWNLOAD_PASS" or report.get("revision") != GOLLIE_REVISION:
        fail("J16 download is not the pinned GoLLIE checkpoint")
    if report.get("repository") != GOLLIE_REPO:
        fail("J16 download is not the pinned GoLLIE checkpoint")
    local = report.get("local_path")
    if not isinstance(local, str) or not local:
        fail("J16 download path is absent")
    origin = Path(local)
    scratch = Path(os.environ["SCRATCH_ROOT"]).resolve()
    if origin.is_symlink() or not origin.is_dir() or not _inside(scratch, origin):
        fail("J16 snapshot is outside SCRATCH")
    return origin


def convert_indexed_checkpoint(
    origin: Path,
    destination: Path,
    *,
    expected_bytes: int,
    scratch: Path,
) -> dict[str, object]:
    """Write a new Safetensors snapshot. The origin directory is only read."""
    import safetensors
    from safetensors.torch import load_file, save_file

    if origin.is_symlink() or not origin.is_dir():
        fail("origin snapshot is not a real directory")
    _assert_destination(origin, destination, scratch)
    weight_map = _read_pytorch_index(origin, scratch)
    by_shard: dict[str, list[str]] = {}
    for tensor_name, filename in weight_map.items():
        by_shard.setdefault(filename, []).append(tensor_name)
    shards = {filename: _shard_path(origin, scratch, filename) for filename in sorted(by_shard)}
    indexed_bytes = sum(path.stat().st_size for path in shards.values())
    if indexed_bytes != expected_bytes:
        fail(f"indexed shard bytes {indexed_bytes} are not {expected_bytes}")
    _assert_space(scratch, indexed_bytes)
    input_files = {PYTORCH_INDEX: _file_record(origin / PYTORCH_INDEX)}
    input_files.update({filename: _file_record(path) for filename, path in shards.items()})
    before = {name: record["sha256"] for name, record in input_files.items()}

    partial = destination.with_name(destination.name + ".partial")
    created = False
    try:
        partial.mkdir(parents=True)
        created = True
        ignored: list[str] = []
        copied: dict[str, dict[str, object]] = {}
        for entry in sorted(origin.iterdir(), key=lambda item: item.name):
            if entry.name == "__pycache__":
                if entry.is_symlink():
                    fail("__pycache__ is a symlink")
                if not entry.is_dir():
                    fail("__pycache__ has an unexpected structure")
                if not _inside(origin, entry.resolve()):
                    fail("__pycache__ escapes the origin snapshot")
                continue
            if entry.name.endswith(".incomplete"):
                fail("origin snapshot has an incomplete download")
            if entry.is_dir():
                fail(f"unexpected directory in the origin snapshot: {entry.name}")
            if entry.name == PYTORCH_INDEX or entry.name in shards:
                continue
            if _is_weight_name(entry.name):
                ignored.append(entry.name)
                continue
            if entry.is_symlink():
                target = entry.resolve()
                if (
                    not target.is_file()
                    or not _inside(scratch, target)
                    or not _inside(_repo_cache(origin), target)
                ):
                    fail(f"companion escapes SCRATCH: {entry.name}")
                payload = target.read_bytes()
            elif entry.is_file() and _inside(origin, entry.resolve()):
                payload = entry.read_bytes()
            else:
                fail(f"companion escapes the origin snapshot: {entry.name}")
            target_path = partial / entry.name
            target_path.write_bytes(payload)
            copied[entry.name] = _file_record(target_path)
        for name in REQUIRED_COMPANIONS:
            record = copied.get(name)
            if record is None:
                fail(f"required companion is absent: {name}")
        modeling = copied[MODELING_FILE]
        if modeling["bytes"] != MODELING_SIZE:
            fail("snapshot modeling module is not the pinned file")

        output_files: dict[str, dict[str, object]] = {}
        derived_map: dict[str, str] = {}
        inventory: list[dict[str, object]] = []
        shared_groups: list[list[str]] = []
        tensor_bytes = 0
        for filename, path in shards.items():
            loaded = _load_shard(path)
            expected = set(by_shard[filename])
            if set(loaded) != expected:
                fail(f"shard does not match the index: {filename}")
            independent, shared = materialize_independent_tensors(loaded)
            shared_groups.extend(shared)
            safetensors_name = _safetensors_name(filename)
            save_file(
                independent,
                str(partial / safetensors_name),
                metadata={"format": "pt"},
            )
            reloaded = load_file(str(partial / safetensors_name))
            if set(reloaded) != set(independent):
                fail(f"reloaded shard does not match the origin: {safetensors_name}")
            for tensor_name, original in independent.items():
                current = reloaded[tensor_name]
                if not original.dtype == current.dtype or original.shape != current.shape:
                    fail(f"reloaded tensor does not match the origin: {tensor_name}")
                import torch

                if not torch.equal(current, original):
                    fail(f"reloaded tensor does not match the origin: {tensor_name}")
                derived_map[tensor_name] = safetensors_name
                tensor_bytes += original.numel() * original.element_size()
                inventory.append(
                    {
                        "dtype": str(original.dtype),
                        "name": tensor_name,
                        "shape": list(original.shape),
                        "shard": safetensors_name,
                    }
                )
            output_files[safetensors_name] = _file_record(partial / safetensors_name)
        if set(derived_map) != set(weight_map):
            fail("derived index does not match the pytorch index")
        index_payload = {"metadata": {"total_size": tensor_bytes}, "weight_map": derived_map}
        index_path = partial / SAFETENSORS_INDEX
        index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_files[SAFETENSORS_INDEX] = _file_record(index_path)
        after = {name: sha256_file(origin / name) for name in before}
        if after != before:
            fail("origin snapshot changed during conversion")
        os.replace(partial, destination)
        created = False
    except BaseException:
        if created and partial.exists():
            shutil.rmtree(partial)
        raise
    return {
        "checkpoint_verified": True,
        "copied_files": copied,
        "derived_index": {"weight_map": derived_map},
        "derived_path": str(destination.resolve()),
        "ignored_weight_files": ignored,
        "indexed_bytes": indexed_bytes,
        "input_files": input_files,
        "map_location": "cpu",
        "origin_path": str(origin.resolve()),
        "origin_unchanged": True,
        "output_files": output_files,
        "repository": GOLLIE_REPO,
        "revision": GOLLIE_REVISION,
        "safetensors_version": safetensors.__version__,
        "shared_storage_groups": shared_groups,
        "tensors": inventory,
        "torch_version": _running_torch_version(),
        "use_safetensors": True,
        "verified": True,
        "weights_only": True,
    }


def _write_conversion_record(runs: Path, job_id: str, payload: dict[str, object]) -> None:
    report = require_under_scratch(conversion_report_path(runs, job_id))
    pointer = require_under_scratch(conversion_pointer(runs))
    if report.exists() or pointer.exists():
        fail(f"refusing to overwrite existing evidence: {report if report.exists() else pointer}")
    write_json(report, payload)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(job_id + "\n", encoding="utf-8")


def assert_j17_conversion(runs_root: Path) -> dict[str, object]:
    """J17 stays closed until this record proves the Safetensors snapshot."""
    pointer = conversion_pointer(runs_root)
    if not pointer.is_file():
        fail("GoLLIE conversion is unidentified; refusing the native smoke")
    job_id = pointer.read_text(encoding="utf-8").strip()
    report_path = conversion_report_path(runs_root, job_id)
    if not report_path.is_file():
        fail("GoLLIE conversion report is missing")
    report = read_json(report_path)
    if report.get("slurm_job_id") != job_id:
        fail("GoLLIE conversion report belongs to another JOBID")
    if report.get("status") != "CONVERSION_PASS" or report.get("verified") is not True:
        fail("GoLLIE conversion is not a verified safetensors snapshot")
    if report.get("weights_only") is not True or report.get("use_safetensors") is not True:
        fail("GoLLIE conversion did not keep the safetensors load")
    if report.get("repository") != GOLLIE_REPO or report.get("revision") != GOLLIE_REVISION:
        fail("GoLLIE conversion revision is not the pinned checkpoint")
    if report.get("indexed_bytes") != INDEXED_SHARD_BYTES or report.get("origin_unchanged") is not True:
        fail("GoLLIE conversion does not match the indexed shards")
    derived = report.get("derived_path")
    if not isinstance(derived, str) or not derived:
        fail("GoLLIE conversion path is absent")
    snapshot = Path(derived)
    scratch = Path(os.environ["SCRATCH_ROOT"]).resolve()
    if snapshot.is_symlink() or not snapshot.is_dir() or not _inside(scratch, snapshot):
        fail("derived snapshot is outside SCRATCH")
    from re_te_system.extractors.gollie import assert_safetensors_snapshot

    try:
        assert_safetensors_snapshot(snapshot)
    except ValueError as exc:
        fail(str(exc))
    files = report.get("output_files")
    copied = report.get("copied_files")
    if not isinstance(files, dict) or not isinstance(copied, dict):
        fail("GoLLIE conversion file inventory is absent")
    for name, record in {**copied, **files}.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            fail("GoLLIE conversion file inventory is malformed")
        path = snapshot / name
        if path.is_symlink() or not path.is_file():
            fail(f"derived file is absent: {name}")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            fail(f"derived file hash does not match the conversion record: {name}")
    return report


def main() -> int:
    if os.environ.get("GOLLIE_SAFETENSORS_CONVERT_AUTHORIZED") != "1":
        fail("GoLLIE safetensors conversion is not authorized")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    _require_job_id(job_id)
    version = assert_cpu_runtime()
    runs = Path(os.environ["P3_RUNS_ROOT"])
    if conversion_pointer(runs).exists():
        fail("refusing to overwrite existing evidence: conversion pointer")
    origin = _origin_from_j16(runs)
    scratch = Path(os.environ["SCRATCH_ROOT"])
    destination = derived_snapshot_dir(Path(os.environ["P3_MODELS_ROOT"]))
    payload = convert_indexed_checkpoint(
        origin,
        destination,
        expected_bytes=INDEXED_SHARD_BYTES,
        scratch=scratch,
    )
    payload["slurm_job_id"] = job_id
    payload["status"] = "CONVERSION_PASS"
    payload["torch_version"] = version
    payload["venv"] = sys.prefix
    _write_conversion_record(runs, job_id, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
