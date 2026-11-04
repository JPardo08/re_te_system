"""Gated conversion of the pinned UIE checkpoint to one Safetensors file.

The origin pickle stays in the Hugging Face snapshot. Deserialization is
allowed only for an authorized CPU interpreter at PyTorch >= 2.6, and only
through torch.load(..., map_location="cpu", weights_only=True). PyTorch
2.5.1 and the uie-train environment are refused. Transformers' torch.load
gate is not modified. The module constants in uie_origin_pin stay false:
UIE_SAFETENSORS_CONVERT_AUTHORIZED and UIE_J12_DERIVED_LOAD_AUTHORIZED are
independent environment grants, and neither one grants J13.

Tensor content hash: SHA-256 of the raw little-endian bytes of a
C-contiguous CPU tensor, exactly numel * element_size, with no header.
dtype and shape are stored beside that digest. They are not folded into it.
A shared storage is recorded before any clone. Safetensors receives
independent clones, so a later physical alias is not claimed unless the
reloaded tensors still share storage.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from cesvima_common import canonical_json, fail, read_json, require_under_scratch, scratch_root, sha256_file
from uie_origin_pin import (
    CONVERSION_ENABLED,
    ORIGIN_BIN_FILENAME,
    ORIGIN_BIN_SHA256,
    ORIGIN_BIN_SIZE,
    ORIGIN_REVISION,
    deserialization_authorized,
    origin_identity_matches,
    verify_origin_bin,
)


class ConversionDisabled(RuntimeError):
    """The conversion is unauthorized or the checkpoint cannot be published."""


UIE_REPOSITORY = "luyaojie/uie-base-en"
CONVERT_ENV = "UIE_SAFETENSORS_CONVERT_AUTHORIZED"
J12_LOAD_ENV = "UIE_J12_DERIVED_LOAD_AUTHORIZED"
FORBIDDEN_RUNTIME = "uie-train"
MINIMUM_DESERIALIZATION_TORCH = (2, 6)
INTENDED_LOAD = "torch.load(path, map_location='cpu', weights_only=True)"
REQUIRED_COMPANIONS = ("config.json", "tokenizer_config.json")
# Inventory observed on the J11 snapshot. Only REQUIRED_COMPANIONS are
# mandatory. The other non-weight names are copied when they are present.
# pytorch_model.bin is never copied into the derived directory.
J11_SNAPSHOT_FILES = (
    "pytorch_model.bin",
    "config.json",
    "tokenizer_config.json",
    "spiece.model",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "README.md",
    ".gitattributes",
)
# `quota` without an explicit unit prints 1 KiB blocks. Any other shape is
# left unobserved instead of being treated as a quota pass.
QUOTA_BLOCK_BYTES = 1024
TENSOR_CONTENT_HASH = {
    "algorithm": "sha256",
    "byte_order": "little",
    "contiguous": "C",
    "device": "cpu",
    "dtype_and_shape": "recorded separately from the digest",
    "payload": "raw element bytes of numel * element_size, without a header",
}
CONVERSION_REQUIREMENTS = {
    "enabled": CONVERSION_ENABLED,
    "deserialization_authorized": False,
    "independent_environment": True,
    "minimum_torch": "2.6",
    "forbidden_runtime": "2.5.1",
    "weights_only": True,
    "transformers_security_check": "must_remain_enabled",
    "mutates_origin_snapshot": False,
    "intended_load": INTENDED_LOAD,
    "conversion_authorization_env": CONVERT_ENV,
    "j12_load_authorization_env": J12_LOAD_ENV,
}


def conversion_job_authorized() -> bool:
    """CPU conversion grant. This does not read the module constants."""
    return os.environ.get(CONVERT_ENV) == "1"


def j12_load_authorized() -> bool:
    """J12 load grant. Conversion authorization does not imply this one."""
    return os.environ.get(J12_LOAD_ENV) == "1"


def torch_version_can_deserialize_origin(version: str) -> bool:
    """Version predicate only. A true result is not a load authorization."""
    numeric = version.split("+", 1)[0]
    pieces = []
    for piece in numeric.split("."):
        if not piece.isdigit():
            break
        pieces.append(int(piece))
    while len(pieces) < 3:
        pieces.append(0)
    return tuple(pieces[:3]) >= MINIMUM_DESERIALIZATION_TORCH


def _runtime_prefix_name() -> str:
    return Path(sys.prefix).name


def _running_torch_version() -> str:
    import torch

    return str(torch.__version__)


def assert_conversion_runtime(torch_version: str) -> None:
    if not conversion_job_authorized():
        raise ConversionDisabled(
            "origin SHA-256 match does not authorize deserialization; "
            "conversion is disabled"
        )
    if _runtime_prefix_name() == FORBIDDEN_RUNTIME:
        raise ConversionDisabled("refusing to load pytorch_model.bin in uie-train")
    if torch_version.split("+", 1)[0].startswith("2.5.1") or not torch_version_can_deserialize_origin(
        torch_version
    ):
        raise ConversionDisabled("PyTorch 2.5.1 must not open pytorch_model.bin")


def _job_id_ok(job_id: str) -> bool:
    return bool(job_id) and job_id not in {".", ".."} and "/" not in job_id and "\\" not in job_id


def _require_job_id(job_id: str) -> None:
    if not _job_id_ok(job_id):
        fail(f"SLURM_JOB_ID is not a single path segment: {job_id!r}")


def conversion_pointer(runs_root: Path) -> Path:
    return runs_root / "uie" / "conversion" / "accepted_job_id.txt"


def conversion_report_path(runs_root: Path, job_id: str) -> Path:
    _require_job_id(job_id)
    return runs_root / "uie" / "conversion" / "jobs" / job_id / "conversion_report.json"


def derived_snapshot_dir(models_root: Path) -> Path:
    return models_root / "uie" / f"luyaojie-uie-base-en-{ORIGIN_REVISION}-safetensors"


def _inside(root: Path, path: Path) -> bool:
    resolved = path.resolve()
    base = root.resolve()
    return resolved == base or base in resolved.parents


def _writes_inside_origin(origin_bin: Path, destination: Path) -> bool:
    return destination.resolve() == origin_bin.resolve() or destination.resolve().is_relative_to(
        origin_bin.resolve().parent
    )


def convert_origin_to_safetensors(
    origin_bin: Path,
    destination: Path,
    *,
    loader: object = None,
) -> dict[str, object]:
    """Convert the pinned bin, or refuse before any loader runs.

    An injected loader is ignored. The only read of the pickle is the
    weights_only load inside this module, and only after the pin matches.
    """
    del loader
    if _writes_inside_origin(origin_bin, destination):
        raise ConversionDisabled("refusing to write inside the origin snapshot")
    if origin_bin.name != ORIGIN_BIN_FILENAME:
        raise ConversionDisabled("conversion input must be the preserved pytorch_model.bin")
    assert_conversion_runtime(_running_torch_version())
    if verify_origin_bin(origin_bin).get("matches_pin") is not True:
        raise ConversionDisabled("origin pytorch_model.bin is not the pinned checkpoint")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    _require_job_id(job_id)
    return convert_pinned_origin(origin_bin.parent, destination, scratch_root(), job_id)


def tensor_values_sha256(tensor: object) -> str:
    """SHA-256 of the contiguous little-endian CPU bytes. No header."""
    import torch

    if sys.byteorder != "little":
        raise ConversionDisabled("tensor content hash requires a little-endian host")
    if not isinstance(tensor, torch.Tensor):
        raise ConversionDisabled("tensor content hash requires a tensor")
    cpu = tensor.detach().to(device="cpu").contiguous()
    if cpu.layout != torch.strided:
        raise ConversionDisabled("tensor content hash requires a dense strided tensor")
    nbytes = cpu.numel() * cpu.element_size()
    storage = cpu.untyped_storage()
    if storage.nbytes() != nbytes:
        raise ConversionDisabled("tensor content hash requires exact storage coverage")
    payload = bytes(storage) if nbytes else b""
    return hashlib.sha256(payload).hexdigest()


def _storage_key(tensor: object) -> tuple[int, int]:
    storage = tensor.untyped_storage()  # type: ignore[attr-defined]
    return (storage.data_ptr(), storage.nbytes())


def _covers_storage(tensor: object) -> bool:
    if tensor.numel() == 0:  # type: ignore[attr-defined]
        return False
    storage = tensor.untyped_storage()  # type: ignore[attr-defined]
    return (
        tensor.data_ptr() == storage.data_ptr()  # type: ignore[attr-defined]
        and tensor.numel() * tensor.element_size() == storage.nbytes()  # type: ignore[attr-defined]
    )


def storage_ids(tensors: dict) -> tuple[dict[str, str], list[list[str]]]:
    """Name the physical alias groups. Partial views are refused."""
    import torch

    groups: dict[tuple[int, int], list[str]] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise ConversionDisabled(f"state dict entry is not a tensor: {name}")
        if tensor.layout != torch.strided or tensor.numel() == 0:
            raise ConversionDisabled(f"tensor {name} is not a dense strided tensor")
        groups.setdefault(_storage_key(tensor), []).append(name)
    shared: list[list[str]] = []
    ids: dict[str, str] = {}
    for names in groups.values():
        ordered = sorted(names)
        if len(ordered) == 1:
            ids[ordered[0]] = "unique:" + ordered[0]
            continue
        members = [tensors[name] for name in ordered]
        if not all(_covers_storage(item) for item in members):
            raise ConversionDisabled("shared storage cannot be preserved: " + ", ".join(ordered))
        reference = members[0]
        for item in members[1:]:
            if item.dtype != reference.dtype or item.shape != reference.shape or not torch.equal(item, reference):
                raise ConversionDisabled("shared storage cannot be preserved: " + ", ".join(ordered))
        shared.append(ordered)
        group_id = "shared:" + "|".join(ordered)
        for name in ordered:
            ids[name] = group_id
    return ids, shared


def materialize_independent_tensors(tensors: dict) -> tuple[dict, list[list[str]], dict[str, str]]:
    """Clone complete shared tensors. Values stay equal; the alias is not kept."""
    ids, shared = storage_ids(tensors)
    independent = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in tensors.items()
    }
    for name, original in tensors.items():
        if tensor_values_sha256(original) != tensor_values_sha256(independent[name]):
            raise ConversionDisabled(f"clone changed tensor values: {name}")
    return independent, shared, ids


def _tensor_record(name: str, tensor: object, storage_id: str) -> dict[str, object]:
    return {
        "dtype": str(tensor.dtype),  # type: ignore[attr-defined]
        "name": name,
        "shape": list(tensor.shape),  # type: ignore[attr-defined]
        "storage_id": storage_id,
        "values_sha256": tensor_values_sha256(tensor),
    }


def _alias_contract(origin_groups: Sequence[Sequence[str]], derived_groups: Sequence[Sequence[str]]) -> dict[str, object]:
    origin = [list(item) for item in origin_groups]
    derived = [list(item) for item in derived_groups]
    preserved = {frozenset(item) for item in origin} == {frozenset(item) for item in derived}
    return {
        "derived_physical_groups": derived,
        "origin_shared_groups": origin,
        "physical_alias_preserved": preserved,
        "values_decoupled_without_change": True,
    }


def write_verified_safetensors(tensors: dict, destination_file: Path) -> dict[str, object]:
    """Save clones and prove the reloaded file matches. This does not read a pickle."""
    import torch
    from safetensors.torch import load_file, save_file

    if destination_file.exists() or destination_file.is_symlink():
        raise ConversionDisabled("refusing to overwrite a safetensors file")
    if not tensors:
        raise ConversionDisabled("state dict is empty")
    origin_ids, origin_groups = storage_ids(tensors)
    independent, shared, _ids = materialize_independent_tensors(tensors)
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(independent, str(destination_file), metadata={"format": "pt"})
    reloaded = load_file(str(destination_file))
    if set(reloaded) != set(tensors):
        raise ConversionDisabled("reloaded safetensors keys differ from the origin state dict")
    derived_ids, derived_groups = storage_ids(reloaded)
    origin_records = []
    derived_records = []
    for name in sorted(tensors):
        original = tensors[name]
        current = reloaded[name]
        if original.dtype != current.dtype or tuple(original.shape) != tuple(current.shape):
            raise ConversionDisabled(f"reloaded tensor metadata differs: {name}")
        if not torch.equal(current, independent[name]):
            raise ConversionDisabled(f"reloaded tensor values differ: {name}")
        origin_digest = tensor_values_sha256(original)
        if origin_digest != tensor_values_sha256(current):
            raise ConversionDisabled(f"reloaded tensor digest differs: {name}")
        origin_records.append(_tensor_record(name, original, origin_ids[name]))
        derived_records.append(_tensor_record(name, current, derived_ids[name]))
    if any(destination_file.parent.glob("*.bin")):
        raise ConversionDisabled("derived directory contains a pickle weight")
    return {
        "alias_contract": _alias_contract(origin_groups, derived_groups),
        "derived_tensors": derived_records,
        "origin_shared_storage_groups": shared,
        "origin_tensors": origin_records,
        "physical_alias_preserved": _alias_contract(origin_groups, derived_groups)["physical_alias_preserved"],
        "safetensors_sha256": sha256_file(destination_file),
        "safetensors_size": destination_file.stat().st_size,
    }


def _require_cpu_tensor(name: str, value: object) -> None:
    device = getattr(value, "device", None)
    if getattr(device, "type", None) != "cpu":
        raise ConversionDisabled(f"tensor {name} is not on CPU")


def _load_origin_bin(path: Path) -> dict:
    import torch

    # Authorization, the PyTorch 2.6 floor, and the uie-train refusal are
    # repeated here so a caller cannot reach torch.load through this function
    # by skipping main().
    assert_conversion_runtime(_running_torch_version())
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except ConversionDisabled:
        raise
    except Exception as exc:
        raise ConversionDisabled(f"origin bin is not a weights_only state dict: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ConversionDisabled("origin bin is not a tensor state dict")
    for name, value in payload.items():
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise ConversionDisabled(f"state dict entry is not a tensor: {name}")
        _require_cpu_tensor(name, value)
    return payload


def _is_weight_name(name: str) -> bool:
    return name.endswith((".bin", ".safetensors", ".pt", ".pth")) or name in {
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }


def _file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _repo_cache(origin: Path) -> Path:
    return origin.parent.parent


def _read_companion(origin: Path, scratch: Path, entry: Path) -> bytes:
    if entry.is_symlink():
        target = entry.resolve()
        cache = _repo_cache(origin)
        if not target.is_file() or not _inside(scratch, target):
            raise ConversionDisabled(f"companion escapes SCRATCH: {entry.name}")
        if not _inside(cache, target):
            raise ConversionDisabled(f"companion escapes the authorized cache: {entry.name}")
        return target.read_bytes()
    if entry.is_file() and _inside(origin, entry.resolve()):
        return entry.read_bytes()
    raise ConversionDisabled(f"companion escapes the origin snapshot: {entry.name}")


def partial_snapshot_dir(destination: Path, job_id: str) -> Path:
    """Job-private partial. Two conversions do not share this directory."""
    _require_job_id(job_id)
    return destination.with_name(f"{destination.name}.{job_id}.partial")


def _assert_destination(origin: Path, destination: Path, scratch: Path, job_id: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise ConversionDisabled("refusing to overwrite a derived snapshot")
    if not _inside(scratch, destination.parent):
        raise ConversionDisabled("derived snapshot is outside SCRATCH")
    cache = os.environ.get("HF_HUB_CACHE", "")
    if cache and _inside(Path(cache), destination):
        raise ConversionDisabled("refusing to write inside the Hugging Face cache")
    repo = _repo_cache(origin)
    if _inside(repo, destination) or _inside(origin, destination) or destination.resolve() == origin.resolve():
        raise ConversionDisabled("refusing to write inside the origin snapshot")
    partial = partial_snapshot_dir(destination, job_id)
    if partial.exists() or partial.is_symlink():
        raise ConversionDisabled("refusing to resume a partial derived snapshot")


def _existing_anchor(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            break
        current = current.parent
    return current


def _parse_quota_remaining(text: str) -> int | None:
    """Return one remaining-byte figure, or None when the report is ambiguous."""
    remaining: list[int] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].lower() == "filesystem":
            continue
        try:
            used = int(parts[1])
            limit = int(parts[3])
        except ValueError:
            continue
        if limit <= 0:
            continue
        remaining.append((limit - used) * QUOTA_BLOCK_BYTES)
    if len(remaining) != 1:
        return None
    return remaining[0]


def _user_quota_remaining(path: Path) -> int | None:
    """User quota remaining in bytes, or None when this host does not report one.

    Filesystem free space is a different number and is never substituted here.
    """
    if not path:
        return None
    try:
        completed = subprocess.run(
            ["quota", "-u", "-w"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return _parse_quota_remaining(completed.stdout)


def scratch_capacity(anchor: Path, required: int) -> dict[str, object]:
    """Refuse when filesystem free space, or an observed user quota, is short."""
    if required < 0:
        raise ConversionDisabled("required space is negative")
    try:
        free = int(shutil.disk_usage(anchor).free)
    except OSError as exc:
        raise ConversionDisabled(f"filesystem free space cannot be read: {exc}") from exc
    quota = _user_quota_remaining(anchor)
    if free < required:
        raise ConversionDisabled(f"filesystem free space {free} bytes is below {required}")
    if quota is not None and int(quota) < required:
        raise ConversionDisabled(f"user quota remaining {quota} bytes is below {required}")
    return {
        "filesystem_free_bytes": free,
        "quota_observed": quota is not None,
        "quota_remaining_bytes": quota,
        "required_bytes": required,
    }


def _publish_directory(partial: Path, destination: Path) -> None:
    """Move the verified partial into place. An existing destination is left untouched."""
    if destination.exists() or destination.is_symlink():
        raise ConversionDisabled("refusing to overwrite a derived snapshot")
    try:
        os.rename(partial, destination)
    except OSError as exc:
        if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ConversionDisabled("refusing to overwrite a derived snapshot") from exc
        raise


def convert_pinned_origin(
    origin: Path,
    destination: Path,
    scratch: Path,
    job_id: str,
) -> dict[str, object]:
    """Publish one Safetensors snapshot of the pinned directory. Origin is read-only."""
    import safetensors

    if not conversion_job_authorized():
        raise ConversionDisabled("conversion is disabled")
    if origin.is_symlink() or not origin.is_dir():
        raise ConversionDisabled("origin snapshot is not a real directory")
    if not _inside(scratch, origin):
        raise ConversionDisabled("origin snapshot is outside SCRATCH")
    bin_path = origin / ORIGIN_BIN_FILENAME
    if bin_path.is_symlink():
        target = bin_path.resolve()
        cache = _repo_cache(origin)
        if not target.is_file() or not _inside(scratch, target):
            raise ConversionDisabled("origin pytorch_model.bin escapes SCRATCH")
        if not _inside(cache, target):
            raise ConversionDisabled("origin pytorch_model.bin escapes the authorized cache")
    elif not bin_path.is_file() or not _inside(origin, bin_path.resolve()):
        raise ConversionDisabled("origin pytorch_model.bin is absent")
    before = sha256_file(bin_path)
    size = bin_path.stat().st_size
    if not origin_identity_matches(size, before):
        raise ConversionDisabled("origin pytorch_model.bin is not the pinned checkpoint")
    _require_job_id(job_id)
    _assert_destination(origin, destination, scratch, job_id)
    space = scratch_capacity(_existing_anchor(destination.parent), size * 2)

    partial = partial_snapshot_dir(destination, job_id)
    created = False
    try:
        partial.parent.mkdir(parents=True, exist_ok=True)
        try:
            partial.mkdir()
        except FileExistsError:
            raise ConversionDisabled("refusing to resume a partial derived snapshot") from None
        created = True
        copied: dict[str, dict[str, object]] = {}
        ignored: list[str] = []
        for entry in sorted(origin.iterdir(), key=lambda item: item.name):
            if entry.name == "__pycache__":
                if entry.is_symlink() or not entry.is_dir() or not _inside(origin, entry.resolve()):
                    raise ConversionDisabled("__pycache__ has an unexpected structure")
                continue
            if entry.name.endswith(".incomplete"):
                raise ConversionDisabled("origin snapshot has an incomplete download")
            if entry.is_dir():
                raise ConversionDisabled(f"unexpected directory in the origin snapshot: {entry.name}")
            if entry.name == ORIGIN_BIN_FILENAME or _is_weight_name(entry.name):
                ignored.append(entry.name)
                continue
            payload_bytes = _read_companion(origin, scratch, entry)
            target_path = partial / entry.name
            if target_path.exists() or target_path.is_symlink():
                raise ConversionDisabled(f"refusing to overwrite a companion: {entry.name}")
            target_path.write_bytes(payload_bytes)
            if target_path.is_symlink():
                raise ConversionDisabled(f"companion was not copied as a regular file: {entry.name}")
            copied[entry.name] = _file_record(target_path)
        for name in REQUIRED_COMPANIONS:
            if name not in copied:
                raise ConversionDisabled(f"required companion is absent: {name}")
        loaded = _load_origin_bin(bin_path)
        verified = write_verified_safetensors(loaded, partial / "model.safetensors")
        if any(partial.glob("*.bin")):
            raise ConversionDisabled("derived directory contains a pickle weight")
        after_size = bin_path.stat().st_size
        after = sha256_file(bin_path)
        if after != before or after_size != size:
            raise ConversionDisabled("origin pytorch_model.bin changed during conversion")
        output = _file_record(partial / "model.safetensors")
        material_errors = verify_derived_material(
            partial,
            {
                "alias_contract": verified["alias_contract"],
                "copied_files": copied,
                "derived_tensors": verified["derived_tensors"],
                "reported_loaded_sha256": output["sha256"],
            },
        )
        if material_errors:
            raise ConversionDisabled("; ".join(material_errors))
        _publish_directory(partial, destination)
        created = False
    except BaseException:
        if created and partial.exists() and not partial.is_symlink():
            shutil.rmtree(partial)
        raise
    published = _file_record(destination / "model.safetensors")
    if published != output:
        raise ConversionDisabled("published safetensors bytes differ from the verified partial")
    return {
        "alias_contract": verified["alias_contract"],
        "copied_files": copied,
        "derived_path": str(destination.resolve()),
        "derived_tensors": verified["derived_tensors"],
        "ignored_weight_files": ignored,
        "map_location": "cpu",
        "origin_path": str(origin.resolve()),
        "origin_sha256": before,
        "origin_shared_storage_groups": verified["origin_shared_storage_groups"],
        "origin_size": size,
        "origin_tensors": verified["origin_tensors"],
        "origin_unchanged": True,
        "output_files": {"model.safetensors": published},
        "physical_alias_preserved": verified["physical_alias_preserved"],
        "reported_loaded_sha256": published["sha256"],
        "repository": UIE_REPOSITORY,
        "revision": ORIGIN_REVISION,
        "safetensors_version": safetensors.__version__,
        "space": space,
        "tensor_content_hash": dict(TENSOR_CONTENT_HASH),
        "use_safetensors": True,
        "weights_only": True,
    }


def _base_snapshot(runs: Path) -> Path:
    pointer = runs / "uie" / "base_snapshot.json"
    if not pointer.is_file():
        raise ConversionDisabled("base snapshot pointer is absent")
    record = read_json(pointer)
    if record.get("repository") != UIE_REPOSITORY:
        raise ConversionDisabled("base snapshot repository is not the pinned origin")
    if record.get("revision") != ORIGIN_REVISION:
        raise ConversionDisabled("base snapshot revision is not the pinned origin")
    local = record.get("local_path")
    if not isinstance(local, str) or not local.strip():
        raise ConversionDisabled("base snapshot path is absent")
    return Path(local)


def _exclusive_write(path: Path, content: str) -> None:
    """Create a new file. An existing path, including a symlink, is left unchanged."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o644)
    try:
        view = memoryview(content.encode("utf-8"))
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"incomplete exclusive write: {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        fail(f"refusing to follow an evidence symlink: {path}")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        _exclusive_write(path, text)
    except FileExistsError:
        fail(f"refusing to overwrite existing evidence: {path}")
    print(f"WROTE {path}", flush=True)


def publication_lock_path(runs: Path) -> Path:
    return runs / "uie" / "conversion" / "publish.lock"


def _acquire_publish_lock(runs: Path, job_id: str) -> Path:
    lock = require_under_scratch(publication_lock_path(runs))
    if lock.is_symlink():
        raise ConversionDisabled("UIE conversion publication lock is a symlink")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        _exclusive_write(lock, job_id + "\n")
    except FileExistsError:
        raise ConversionDisabled("UIE conversion publication lock is already held") from None
    return lock


def _release_publish_lock(lock: Path, job_id: str) -> None:
    if lock.is_symlink() or not lock.is_file():
        return
    try:
        current = lock.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if current != job_id:
        return
    try:
        lock.unlink()
    except OSError:
        return


def _write_failure_report(runs: Path, job_id: str, message: str) -> None:
    report = require_under_scratch(conversion_report_path(runs, job_id))
    if report.exists() or report.is_symlink():
        fail(f"refusing to overwrite existing evidence: {report}")
    _exclusive_json(
        report,
        {
            "error": message,
            "slurm_job_id": job_id,
            "status": "CONVERSION_FAIL",
            "verified": False,
            "weights_only": True,
        },
    )


def _write_success(runs: Path, job_id: str, payload: dict[str, object]) -> None:
    report = require_under_scratch(conversion_report_path(runs, job_id))
    pointer = require_under_scratch(conversion_pointer(runs))
    if report.is_symlink() or pointer.is_symlink():
        fail("refusing to follow an evidence symlink")
    if report.exists() or pointer.exists():
        fail(f"refusing to overwrite existing evidence: {report if report.exists() else pointer}")
    _exclusive_json(report, payload)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    try:
        _exclusive_write(pointer, job_id + "\n")
    except FileExistsError:
        fail(f"refusing to overwrite existing evidence: {pointer}")
    print(f"WROTE {pointer}", flush=True)


def _conversion_report_errors(report: Mapping[str, object], job_id: str) -> list[str]:
    errors: list[str] = []
    if report.get("slurm_job_id") != job_id or report.get("conversion_job_id") != job_id:
        errors.append("conversion report belongs to another JOBID")
    if report.get("status") != "CONVERSION_PASS" or report.get("verified") is not True:
        errors.append("conversion is not a verified safetensors snapshot")
    if report.get("weights_only") is not True or report.get("use_safetensors") is not True:
        errors.append("conversion did not keep the safetensors load")
    if report.get("revision") != ORIGIN_REVISION:
        errors.append("conversion revision is not the pinned origin")
    if report.get("origin_unchanged") is not True:
        errors.append("conversion does not prove the origin was unchanged")
    derived = report.get("derived_path")
    if not isinstance(derived, str) or not derived.strip():
        errors.append("conversion path is absent")
        return errors
    path = Path(derived)
    try:
        scratch = scratch_root()
    except SystemExit:
        errors.append("derived snapshot is outside SCRATCH")
        return errors
    if path.is_symlink() or not path.is_dir() or not _inside(scratch, path):
        errors.append("derived snapshot is outside SCRATCH")
    return errors


def inspect_conversion_evidence(runs_root: Path) -> tuple[dict | None, list[str]]:
    """Return the accepted report, or None and the reasons it cannot be used.

    A missing pointer is not an error by itself. J12 reports that as a
    disabled conversion. A present but inconsistent pointer is an error and
    still does not authorize the load.
    """
    pointer = conversion_pointer(runs_root)
    if not pointer.is_file():
        return None, []
    job_id = pointer.read_text(encoding="utf-8").strip()
    if not _job_id_ok(job_id):
        return None, ["conversion job id is not a single path segment"]
    report_path = runs_root / "uie" / "conversion" / "jobs" / job_id / "conversion_report.json"
    if not report_path.is_file():
        return None, ["conversion report is missing"]
    try:
        report = read_json(report_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, ["conversion report is malformed"]
    if not isinstance(report, dict):
        return None, ["conversion report is malformed"]
    errors = _conversion_report_errors(report, job_id)
    if errors:
        return None, errors
    return report, []


def conversion_report_is_bound(provenance: Mapping[str, object] | None) -> bool:
    if not isinstance(provenance, Mapping):
        return False
    job = provenance.get("conversion_job_id")
    if not isinstance(job, str) or not _job_id_ok(job):
        return False
    return (
        provenance.get("status") == "CONVERSION_PASS"
        and provenance.get("verified") is True
        and provenance.get("slurm_job_id") == job
        and provenance.get("weights_only") is True
        and provenance.get("use_safetensors") is True
        and provenance.get("revision") == ORIGIN_REVISION
        and provenance.get("origin_unchanged") is True
    )


def provenance_digest(report: Mapping[str, object]) -> str:
    payload = {
        "alias_contract": report.get("alias_contract"),
        "derived_tensors": report.get("derived_tensors"),
        "origin_sha256": report.get("origin_sha256"),
        "origin_size": report.get("origin_size"),
        "origin_tensors": report.get("origin_tensors"),
        "reported_loaded_sha256": report.get("reported_loaded_sha256"),
        "revision": report.get("revision"),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def compare_tensor_records(
    origin: Sequence[Mapping[str, object]],
    derived: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare names, shapes, dtypes, values, and shared-storage groups.

    Records are plain mappings. This function does not read a checkpoint.
    Equal storage groups mean the records name the same aliases. They do not
    by themselves prove that a Safetensors file still shares physical storage.
    """

    def index(records: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
        return {str(item["name"]): item for item in records}

    left = index(origin)
    right = index(derived)
    names_equal = set(left) == set(right)
    shapes_equal = names_equal and all(left[name]["shape"] == right[name]["shape"] for name in left)
    dtypes_equal = names_equal and all(left[name]["dtype"] == right[name]["dtype"] for name in left)
    values_equal = names_equal and all(
        left[name]["values_sha256"] == right[name]["values_sha256"] for name in left
    )

    def groups(records: dict[str, Mapping[str, object]]) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = {}
        for name, item in records.items():
            grouped.setdefault(str(item["storage_id"]), set()).add(name)
        return {key: names for key, names in grouped.items() if len(names) > 1}

    shared_equal = names_equal and groups(left) == groups(right)
    return {
        "dtypes_equal": dtypes_equal,
        "equivalent": names_equal and shapes_equal and dtypes_equal and values_equal and shared_equal,
        "names_equal": names_equal,
        "shapes_equal": shapes_equal,
        "shared_storage_groups_equal": shared_equal,
        "values_equal": values_equal,
    }


def load_equivalence(
    *,
    origin_bin_sha256: str,
    derived_file_sha256: str,
    reported_loaded_sha256: str,
) -> dict[str, bool]:
    """A reported load matches the derived file and is not the origin pickle."""
    derived_match = reported_loaded_sha256 == derived_file_sha256
    origin_not_loaded = reported_loaded_sha256 != origin_bin_sha256
    distinct = origin_bin_sha256 != derived_file_sha256
    return {
        "derived_file_matches_reported_load": derived_match,
        "equivalent": derived_match and origin_not_loaded and distinct,
        "origin_bin_was_not_the_loaded_file": origin_not_loaded,
    }


def _normalize_groups(value: object) -> set[frozenset[str]]:
    if not isinstance(value, list):
        raise TypeError
    groups: set[frozenset[str]] = set()
    for item in value:
        if not isinstance(item, list) or len(item) < 2:
            raise TypeError
        names: list[str] = []
        for name in item:
            if not isinstance(name, str) or not name:
                raise TypeError
            names.append(name)
        groups.add(frozenset(names))
    return groups


def _record_groups(records: Sequence[Mapping[str, object]]) -> set[frozenset[str]]:
    grouped: dict[str, set[str]] = {}
    for item in records:
        grouped.setdefault(str(item["storage_id"]), set()).add(str(item["name"]))
    return {frozenset(names) for names in grouped.values() if len(names) > 1}


def _alias_claim_accepted(provenance: Mapping[str, object], compared: Mapping[str, object]) -> bool:
    """Accept equal values. A physical-alias claim must match the storage ids."""
    contract = provenance.get("alias_contract")
    values_ok = bool(
        compared["names_equal"]
        and compared["shapes_equal"]
        and compared["dtypes_equal"]
        and compared["values_equal"]
    )
    if not isinstance(contract, Mapping):
        return bool(compared["equivalent"])
    if not values_ok:
        return False
    origin_tensors = provenance.get("origin_tensors")
    derived_tensors = provenance.get("derived_tensors")
    if not isinstance(origin_tensors, list) or not isinstance(derived_tensors, list):
        return False
    origin_groups = _record_groups(origin_tensors)
    derived_groups = _record_groups(derived_tensors)
    declared_origin = _normalize_groups(contract.get("origin_shared_groups"))
    declared_derived = _normalize_groups(contract.get("derived_physical_groups"))
    if declared_origin != origin_groups or declared_derived != derived_groups:
        return False
    preserved = origin_groups == derived_groups
    if contract.get("physical_alias_preserved") is not preserved:
        return False
    if not preserved and contract.get("values_decoupled_without_change") is not True:
        return False
    return True


def _companion_name_ok(name: object) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and name not in {".", ".."}
        and "/" not in name
        and "\\" not in name
    )


def _records_from_safetensors(path: Path) -> tuple[list[dict[str, object]], list[list[str]]]:
    import torch
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    if not isinstance(tensors, dict) or not tensors:
        raise ConversionDisabled("safetensors snapshot is empty")
    for name, value in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise ConversionDisabled(f"reloaded entry is not a tensor: {name}")
        _require_cpu_tensor(name, value)
    ids, groups = storage_ids(tensors)
    records = [_tensor_record(name, tensors[name], ids[name]) for name in sorted(tensors)]
    return records, groups


def verify_derived_material(model_dir: Path, provenance: Mapping[str, object]) -> list[str]:
    """Recompute derived tensors and companion hashes from the written files.

    The origin pickle is not opened. An internally consistent JSON is not
    enough when it disagrees with these bytes. Value equality is checked
    separately from whether Safetensors kept the original shared storage.
    """
    errors: list[str] = []
    if model_dir.is_symlink() or not model_dir.is_dir():
        return ["derived snapshot is not a real directory"]
    try:
        scratch = scratch_root()
    except Exception:
        scratch = None
    if scratch is not None and not _inside(scratch, model_dir.resolve()):
        errors.append("derived snapshot is outside SCRATCH")
    copied = provenance.get("copied_files")
    if not isinstance(copied, dict) or not copied:
        errors.append("copied companion hashes are absent")
    else:
        for name in REQUIRED_COMPANIONS:
            if name not in copied:
                errors.append(f"required companion is not recorded: {name}")
        for name, record in copied.items():
            if not _companion_name_ok(name):
                errors.append(f"companion name is not a single path segment: {name!r}")
                continue
            path = model_dir / name
            if path.is_symlink() or not path.is_file() or not _inside(model_dir, path.resolve()):
                errors.append(f"companion is not a regular file: {name}")
                continue
            if not isinstance(record, Mapping):
                errors.append(f"companion record is malformed: {name}")
                continue
            measured = _file_record(path)
            if measured.get("sha256") != record.get("sha256") or measured.get("bytes") != record.get("bytes"):
                errors.append(f"companion hash does not match: {name}")
    weight = model_dir / "model.safetensors"
    if weight.is_symlink() or not weight.is_file() or not _inside(model_dir, weight.resolve()):
        errors.append("model.safetensors is not a regular file")
        return errors
    if any(path.is_file() or path.is_symlink() for path in model_dir.glob("*.bin")):
        errors.append("derived directory contains a pickle weight")
    if _file_record(weight).get("sha256") != provenance.get("reported_loaded_sha256"):
        errors.append("model.safetensors hash does not match the conversion record")
    declared = provenance.get("derived_tensors")
    if not isinstance(declared, list) or not declared:
        errors.append("derived tensor records are absent")
        return errors
    names = [item.get("name") for item in declared if isinstance(item, Mapping)]
    if len(names) != len(declared) or len(set(names)) != len(names):
        errors.append("conversion tensor records repeat a name")
        return errors
    try:
        records, groups = _records_from_safetensors(weight)
        compared = compare_tensor_records(declared, records)
    except (ConversionDisabled, OSError, TypeError, ValueError, KeyError) as exc:
        errors.append(f"reloaded tensors cannot be verified: {exc}")
        return errors
    except Exception as exc:
        errors.append(f"reloaded tensors cannot be verified: {exc}")
        return errors
    if not (
        compared["names_equal"]
        and compared["shapes_equal"]
        and compared["dtypes_equal"]
        and compared["values_equal"]
    ):
        errors.append("reloaded tensor values do not match the conversion record")
    contract = provenance.get("alias_contract")
    if not isinstance(contract, Mapping):
        errors.append("alias contract is absent")
        return errors
    try:
        declared_groups = _normalize_groups(contract.get("derived_physical_groups"))
        observed_groups = _normalize_groups(groups)
        origin_groups = _normalize_groups(contract.get("origin_shared_groups"))
    except TypeError:
        errors.append("alias contract groups are malformed")
        return errors
    if declared_groups != observed_groups:
        errors.append("reloaded alias groups do not match the conversion record")
    if contract.get("physical_alias_preserved") is not (observed_groups == origin_groups):
        errors.append("alias preservation claim does not match the derived file")
    if (
        contract.get("physical_alias_preserved") is False
        and contract.get("values_decoupled_without_change") is not True
    ):
        errors.append("alias loss is not recorded as an unchanged value clone")
    return errors


def derived_provenance_accepted(
    identity: Mapping[str, object] | None,
    provenance: Mapping[str, object] | None,
    origin_observation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Recompute pin identity, tensor equivalence, and load identity.

    A boolean claim is not evidence. Without an alias contract, acceptance
    still requires the storage ids to match. With a contract, equal values
    can be accepted when the contract honestly says the physical alias was
    not preserved. A contract that disagrees with the storage ids is refused.
    """
    measured = _measured_derived_sha(identity)
    origin_size = provenance.get("origin_size") if isinstance(provenance, Mapping) else None
    origin_sha = provenance.get("origin_sha256") if isinstance(provenance, Mapping) else None
    origin_ok = (
        isinstance(origin_size, int)
        and isinstance(origin_sha, str)
        and origin_identity_matches(origin_size, origin_sha)
    )
    if isinstance(origin_observation, Mapping) and origin_observation.get("observed") is True:
        origin_ok = origin_ok and origin_observation.get("matches_pin") is True
    compared = _compared_records(provenance)
    alias_ok = False
    if isinstance(provenance, Mapping):
        try:
            alias_ok = _alias_claim_accepted(provenance, compared)
        except (KeyError, TypeError):
            alias_ok = False
    reported = provenance.get("reported_loaded_sha256") if isinstance(provenance, Mapping) else None
    if isinstance(origin_sha, str) and isinstance(measured, str) and isinstance(reported, str):
        loaded = load_equivalence(
            origin_bin_sha256=origin_sha,
            derived_file_sha256=measured,
            reported_loaded_sha256=reported,
        )
    else:
        loaded = {
            "derived_file_matches_reported_load": False,
            "equivalent": False,
            "origin_bin_was_not_the_loaded_file": False,
        }
    identity_ok = isinstance(identity, Mapping) and identity.get("ok") is True and isinstance(measured, str)
    return {
        "accepted": bool(identity_ok and origin_ok and alias_ok and loaded["equivalent"]),
        "equivalence": compared,
        "load": loaded,
        "origin_matches_pin": origin_ok,
    }


def _measured_derived_sha(identity: Mapping[str, object] | None) -> str | None:
    if not isinstance(identity, Mapping):
        return None
    files = identity.get("files")
    if not isinstance(files, Mapping):
        return None
    item = files.get("model.safetensors")
    if not isinstance(item, Mapping):
        return None
    digest = item.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    return digest


def _compared_records(provenance: Mapping[str, object] | None) -> dict[str, object]:
    empty = {
        "dtypes_equal": False,
        "equivalent": False,
        "names_equal": False,
        "shapes_equal": False,
        "shared_storage_groups_equal": False,
        "values_equal": False,
    }
    if not isinstance(provenance, Mapping):
        return empty
    origin_tensors = provenance.get("origin_tensors")
    derived_tensors = provenance.get("derived_tensors")
    if not isinstance(origin_tensors, list) or not isinstance(derived_tensors, list):
        return empty
    if not origin_tensors or not derived_tensors:
        return empty
    try:
        return compare_tensor_records(origin_tensors, derived_tensors)
    except (KeyError, TypeError):
        return empty


def origin_pin_summary() -> dict[str, object]:
    return {
        "deserialization_authorized": deserialization_authorized(),
        "matches_known_pin": origin_identity_matches(ORIGIN_BIN_SIZE, ORIGIN_BIN_SHA256),
        "sha256": ORIGIN_BIN_SHA256,
        "size": ORIGIN_BIN_SIZE,
    }


def main() -> int:
    if not conversion_job_authorized():
        fail("UIE safetensors conversion is not authorized")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    _require_job_id(job_id)
    runs = Path(os.environ["P3_RUNS_ROOT"])
    report = conversion_report_path(runs, job_id)
    pointer = conversion_pointer(runs)
    if report.is_symlink() or pointer.is_symlink() or report.exists() or pointer.exists():
        fail("refusing to overwrite existing evidence")
    lock: Path | None = None
    try:
        lock = _acquire_publish_lock(runs, job_id)
        assert_conversion_runtime(_running_torch_version())
        origin = _base_snapshot(runs)
        destination = derived_snapshot_dir(Path(os.environ["P3_MODELS_ROOT"]))
        payload = convert_pinned_origin(origin, destination, scratch_root(), job_id)
        payload["conversion_job_id"] = job_id
        payload["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        payload["slurm_job_id"] = job_id
        payload["status"] = "CONVERSION_PASS"
        payload["torch_version"] = _running_torch_version()
        payload["venv"] = sys.prefix
        payload["verified"] = True
        _write_success(runs, job_id, payload)
    except ConversionDisabled as exc:
        if not report.exists() and not report.is_symlink():
            _write_failure_report(runs, job_id, str(exc))
        print(f"FAIL: {exc}", flush=True)
        return 1
    finally:
        if lock is not None:
            _release_publish_lock(lock, job_id)
    print(f"PASS UIE safetensors conversion job {job_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
