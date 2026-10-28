"""CPU conversion of the indexed GoLLIE shards. Synthetic tensors only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
sys.path.insert(0, str(SCRIPTS))

import gollie_safetensors_conversion as conversion  # noqa: E402
from cesvima_common import sha256_file  # noqa: E402
from j16_gollie_download import GOLLIE_REVISION, MODELING_FILE, MODELING_SIZE  # noqa: E402
from re_te_system.extractors.gollie import (  # noqa: E402
    CANONICAL_GOLLIE_MODEL_ID,
    CANONICAL_GOLLIE_REVISION,
    GoLLIEConfig,
    resolve_gollie_weights,
)


def _origin(scratch: Path) -> Path:
    origin = (
        scratch
        / "hf-cache"
        / "hub"
        / "models--HiTZ--GoLLIE-7B"
        / "snapshots"
        / GOLLIE_REVISION
    )
    origin.mkdir(parents=True)
    (origin / "config.json").write_text('{"model_type":"llama"}\n', encoding="utf-8")
    (origin / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (origin / MODELING_FILE).write_bytes(b"m" * MODELING_SIZE)
    return origin


def _save_index(origin: Path, weight_map: dict[str, str]) -> None:
    (origin / conversion.PYTORCH_INDEX).write_text(
        json.dumps({"weight_map": weight_map}, sort_keys=True),
        encoding="utf-8",
    )


def _fingerprints(origin: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in origin.iterdir() if path.is_file()}


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    scratch = tmp_path / "scratch"
    runs = scratch / "runs"
    models = scratch / "models"
    cache = scratch / "hf-cache" / "hub"
    for path in (runs, models, cache):
        path.mkdir(parents=True)
    monkeypatch.setenv("SCRATCH_ROOT", str(scratch))
    monkeypatch.setenv("P3_RUNS_ROOT", str(runs))
    monkeypatch.setenv("P3_MODELS_ROOT", str(models))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("SLURM_JOB_ID", "1352001")
    monkeypatch.delenv("GOLLIE_SAFETENSORS_CONVERT_AUTHORIZED", raising=False)
    return scratch, runs, models


def _j16(runs: Path, origin: Path) -> None:
    job = "1351274"
    report = runs / "gollie" / "download" / "jobs" / job / "download_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "checkpoint_complete": True,
                "local_path": str(origin.resolve()),
                "repository": "HiTZ/GoLLIE-7B",
                "revision": GOLLIE_REVISION,
                "slurm_job_id": job,
                "status": "DOWNLOAD_PASS",
            }
        ),
        encoding="utf-8",
    )
    (runs / "gollie" / "download" / "accepted_job_id.txt").write_text(job + "\n", encoding="utf-8")


def _two_shards(origin: Path) -> dict[str, str]:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bias = torch.tensor([0.5, -1.0])
    torch.save(
        {"model.embed_tokens.weight": weight, "lm_head.weight": weight},
        origin / "pytorch_model-00001-of-00002.bin",
    )
    torch.save({"model.layers.0.bias": bias}, origin / "pytorch_model-00002-of-00002.bin")
    (origin / "pytorch_model.bin").write_bytes(b"not-indexed")
    (origin / "model.safetensors").write_bytes(b"other-weights")
    mapping = {
        "lm_head.weight": "pytorch_model-00001-of-00002.bin",
        "model.embed_tokens.weight": "pytorch_model-00001-of-00002.bin",
        "model.layers.0.bias": "pytorch_model-00002-of-00002.bin",
    }
    _save_index(origin, mapping)
    return mapping


def test_indexed_shards_round_trip_without_touching_the_origin(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    real_load = torch.load

    def spy(path: object, *args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return real_load(path, *args, **kwargs)

    torch.load = spy  # type: ignore[assignment]
    try:
        scratch = tmp_path / "scratch"
        origin = _origin(scratch)
        _two_shards(origin)
        before = _fingerprints(origin)
        destination = scratch / "models" / "gollie" / "derived"
        indexed = sum(
            (origin / name).stat().st_size
            for name in (
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
            )
        )
        report = conversion.convert_indexed_checkpoint(
            origin,
            destination,
            expected_bytes=indexed,
            scratch=scratch,
        )
    finally:
        torch.load = real_load  # type: ignore[assignment]

    assert calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu", "weights_only": True},
    ]
    assert report["verified"] is True
    assert report["weights_only"] is True
    assert report["use_safetensors"] is True
    assert report["revision"] == CANONICAL_GOLLIE_REVISION
    assert report["repository"] == CANONICAL_GOLLIE_MODEL_ID
    assert report["origin_unchanged"] is True
    assert report["shared_storage_groups"] == [["lm_head.weight", "model.embed_tokens.weight"]]
    assert report["ignored_weight_files"] == ["model.safetensors", "pytorch_model.bin"]
    assert _fingerprints(origin) == before
    assert not (destination / "pytorch_model.bin").exists()
    assert not (destination / "model.safetensors").exists()
    assert not (destination / conversion.PYTORCH_INDEX).exists()
    from safetensors.torch import load_file

    first = load_file(str(destination / "pytorch_model-00001-of-00002.safetensors"))
    second = load_file(str(destination / "pytorch_model-00002-of-00002.safetensors"))
    assert torch.equal(first["lm_head.weight"], first["model.embed_tokens.weight"])
    assert torch.equal(first["model.embed_tokens.weight"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(second["model.layers.0.bias"], torch.tensor([0.5, -1.0]))
    assert first["lm_head.weight"].untyped_storage().data_ptr() != first["model.embed_tokens.weight"].untyped_storage().data_ptr()
    index = json.loads((destination / conversion.SAFETENSORS_INDEX).read_text(encoding="utf-8"))
    assert index["weight_map"]["model.layers.0.bias"] == "pytorch_model-00002-of-00002.safetensors"
    for name, record in report["input_files"].items():
        assert record["sha256"] == sha256_file(origin / name)
    for name, record in report["output_files"].items():
        assert record["sha256"] == sha256_file(destination / name)
    names = {item["name"]: item for item in report["tensors"]}
    assert names["model.embed_tokens.weight"]["dtype"] == "torch.float32"
    assert names["model.embed_tokens.weight"]["shape"] == [2, 2]
    assert (destination / "config.json").read_text(encoding="utf-8") == '{"model_type":"llama"}\n'
    assert (destination / MODELING_FILE).stat().st_size == MODELING_SIZE


def test_conversion_rejects_corrupt_partial_and_unsafe_shards(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scratch = tmp_path / "scratch"
    origin = _origin(scratch)
    base = torch.arange(4, dtype=torch.float32)
    torch.save({"full": base, "part": base[:2]}, origin / "pytorch_model-00001-of-00002.bin")
    _save_index(
        origin,
        {
            "full": "pytorch_model-00001-of-00002.bin",
            "part": "pytorch_model-00001-of-00002.bin",
        },
    )
    before = _fingerprints(origin)
    destination = scratch / "models" / "derived"
    indexed = (origin / "pytorch_model-00001-of-00002.bin").stat().st_size
    with pytest.raises(SystemExit):
        conversion.convert_indexed_checkpoint(
            origin,
            destination,
            expected_bytes=indexed,
            scratch=scratch,
        )
    assert "shared storage cannot be preserved" in capsys.readouterr().out
    assert not destination.exists()
    assert _fingerprints(origin) == before

    corrupt_root = tmp_path / "corrupt"
    corrupt = _origin(corrupt_root)
    (corrupt / "pytorch_model-00001-of-00002.bin").write_bytes(b"not-a-checkpoint")
    _save_index(corrupt, {"layer.weight": "pytorch_model-00001-of-00002.bin"})
    corrupt_before = sha256_file(corrupt / "pytorch_model-00001-of-00002.bin")
    with pytest.raises(SystemExit):
        conversion.convert_indexed_checkpoint(
            corrupt,
            corrupt_root / "derived",
            expected_bytes=(corrupt / "pytorch_model-00001-of-00002.bin").stat().st_size,
            scratch=corrupt_root,
        )
    assert "not a weights_only state dict" in capsys.readouterr().out
    assert corrupt_before == sha256_file(corrupt / "pytorch_model-00001-of-00002.bin")
    assert not (corrupt_root / "derived").exists()

    extra_root = tmp_path / "extra"
    extra = _origin(extra_root)
    torch.save({"layer.weight": torch.zeros(2)}, extra / "pytorch_model-00001-of-00002.bin")
    _save_index(extra, {"layer.weight": "pytorch_model-00001-of-00002.bin", "missing": "pytorch_model-00001-of-00002.bin"})
    with pytest.raises(SystemExit):
        conversion.convert_indexed_checkpoint(
            extra,
            extra_root / "derived",
            expected_bytes=(extra / "pytorch_model-00001-of-00002.bin").stat().st_size,
            scratch=extra_root,
        )
    assert "does not match the index" in capsys.readouterr().out

    escaped_root = tmp_path / "escaped"
    escaped = _origin(escaped_root)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    (escaped / "pytorch_model-00001-of-00002.bin").symlink_to(outside)
    _save_index(escaped, {"layer.weight": "pytorch_model-00001-of-00002.bin"})
    with pytest.raises(SystemExit):
        conversion.convert_indexed_checkpoint(
            escaped,
            escaped_root / "derived",
            expected_bytes=outside.stat().st_size,
            scratch=escaped_root,
        )
    assert "escapes" in capsys.readouterr().out
    assert outside.read_bytes() == b"secret"

    size_root = tmp_path / "size"
    wrong_size = _origin(size_root)
    torch.save({"layer.weight": torch.zeros(2)}, wrong_size / "pytorch_model-00001-of-00002.bin")
    _save_index(wrong_size, {"layer.weight": "pytorch_model-00001-of-00002.bin"})
    with pytest.raises(SystemExit):
        conversion.convert_indexed_checkpoint(
            wrong_size,
            size_root / "derived",
            expected_bytes=1,
            scratch=size_root,
        )
    assert "indexed shard bytes" in capsys.readouterr().out


def test_j17_accepts_only_a_verified_conversion_outside_the_gpu_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert conversion.torch_version_at_least("2.5.1+cu121") is False
    assert conversion.torch_version_at_least("2.6.0") is True
    assert conversion.INDEXED_SHARD_BYTES == 13_477_726_551
    monkeypatch.setattr(conversion, "_runtime_prefix_name", lambda: "gollie-isolated")
    with pytest.raises(SystemExit):
        conversion.assert_cpu_runtime()
    assert "GPU runtime" in capsys.readouterr().out
    monkeypatch.setattr(conversion, "_runtime_prefix_name", lambda: "re_te_system_312")
    monkeypatch.setattr(conversion, "_running_torch_version", lambda: "2.5.1+cu121")
    with pytest.raises(SystemExit):
        conversion.assert_cpu_runtime()
    assert "below 2.6" in capsys.readouterr().out
    monkeypatch.undo()

    scratch, runs, _models = _runtime(tmp_path, monkeypatch)
    origin = _origin(scratch)
    _two_shards(origin)
    _j16(runs, origin)
    download_report = (runs / "gollie" / "download" / "jobs" / "1351274" / "download_report.json").read_bytes()
    indexed = sum(
        (origin / name).stat().st_size
        for name in ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin")
    )
    monkeypatch.setattr(conversion, "INDEXED_SHARD_BYTES", indexed)
    monkeypatch.setattr(conversion, "_runtime_prefix_name", lambda: "cpu-torch26")
    with pytest.raises(SystemExit):
        conversion.main()
    assert "not authorized" in capsys.readouterr().out
    monkeypatch.setenv("GOLLIE_SAFETENSORS_CONVERT_AUTHORIZED", "1")
    assert conversion.main() == 0
    pointer = runs / "gollie" / "conversion" / "accepted_job_id.txt"
    assert pointer.read_text(encoding="utf-8") == "1352001\n"
    report = json.loads(
        (runs / "gollie" / "conversion" / "jobs" / "1352001" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "CONVERSION_PASS"
    assert report["verified"] is True
    assert report["revision"] == CANONICAL_GOLLIE_REVISION
    assert report["indexed_bytes"] == indexed
    assert report["origin_unchanged"] is True
    assert (runs / "gollie" / "download" / "jobs" / "1351274" / "download_report.json").read_bytes() == download_report
    checked = conversion.assert_j17_conversion(runs)
    assert checked["derived_path"] == report["derived_path"]
    source, options = resolve_gollie_weights(
        GoLLIEConfig(weight_snapshot=report["derived_path"])
    )
    assert source == report["derived_path"]
    assert options["use_safetensors"] is True
    assert "revision" not in options
    with pytest.raises(ValueError, match="refusing pytorch bins"):
        resolve_gollie_weights(GoLLIEConfig())
    (Path(report["derived_path"]) / "pytorch_model.bin").write_bytes(b"fallback")
    with pytest.raises(SystemExit):
        conversion.assert_j17_conversion(runs)
    assert "refusing pytorch weights" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        conversion.main()
    assert "overwrite" in capsys.readouterr().out

    source_text = (SCRIPTS / "gollie_safetensors_conversion.py").read_text(encoding="utf-8")
    loader = (ROOT / "src" / "re_te_system" / "extractors" / "gollie.py").read_text(encoding="utf-8")
    smoke = (SCRIPTS / "j17_gollie_native_smoke.sh").read_text(encoding="utf-8")
    sbatch = (ROOT / "jobs" / "cesvima" / "gollie_safetensors_convert.sbatch").read_text(encoding="utf-8")
    submitter = (SCRIPTS / "submit_train_first.sh").read_text(encoding="utf-8")
    assert 'torch.load(path, map_location="cpu", weights_only=True)' in source_text
    assert "weights_only=False" not in source_text
    assert "add_safe_globals" not in source_text
    assert '"use_safetensors": True' in loader
    assert "use_safetensors=False" not in loader
    assert '"use_safetensors": False' not in loader
    assert "official_re_documents.jsonl" in smoke
    assert "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9" in smoke
    assert "--gollie-weight-snapshot" in smoke
    assert "assert_j17_conversion" in smoke
    assert "GOLLIE_SAFETENSORS_CONVERT_AUTHORIZED=1" not in sbatch
    assert "gollie-isolated" in sbatch
    assert "gollie_safetensors_convert.sbatch" not in submitter
    assert "--all" not in sbatch


_SHARDS = (
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
)


def _shard_hashes(origin: Path) -> dict[str, str]:
    return {name: sha256_file(origin / name) for name in _SHARDS}


def test_ordinary_pycache_is_omitted_and_unsafe_entries_leave_no_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scratch = tmp_path / "allowed"
    origin = _origin(scratch)
    _two_shards(origin)
    cached = origin / "__pycache__"
    cached.mkdir()
    (cached / "modeling_flash_llama.cpython-312.pyc").write_bytes(b"cached")
    before = _shard_hashes(origin)
    destination = scratch / "models" / "derived"
    indexed = sum((origin / name).stat().st_size for name in _SHARDS)
    report = conversion.convert_indexed_checkpoint(
        origin,
        destination,
        expected_bytes=indexed,
        scratch=scratch,
    )
    assert not (destination / "__pycache__").exists()
    assert "__pycache__" not in report["copied_files"]
    assert "__pycache__" not in report["ignored_weight_files"]
    assert (cached / "modeling_flash_llama.cpython-312.pyc").read_bytes() == b"cached"
    assert _shard_hashes(origin) == before

    def plant_symlink(snapshot: Path) -> None:
        target = snapshot.parent / "elsewhere"
        target.mkdir()
        (snapshot / "__pycache__").symlink_to(target, target_is_directory=True)

    def plant_file(snapshot: Path) -> None:
        (snapshot / "__pycache__").write_text("not-a-directory\n", encoding="utf-8")

    def plant_directory(snapshot: Path) -> None:
        (snapshot / "blobs").mkdir()

    def plant_incomplete(snapshot: Path) -> None:
        (snapshot / "model.incomplete").write_bytes(b"partial")

    def refuse(name: str, message: str, plant) -> None:
        root = tmp_path / name
        runtime_scratch, runs, models = _runtime(root, monkeypatch)
        snapshot = _origin(runtime_scratch)
        _two_shards(snapshot)
        _j16(runs, snapshot)
        original = _shard_hashes(snapshot)
        plant(snapshot)
        monkeypatch.setattr(
            conversion,
            "INDEXED_SHARD_BYTES",
            sum((snapshot / shard).stat().st_size for shard in _SHARDS),
        )
        monkeypatch.setattr(conversion, "_runtime_prefix_name", lambda: "cpu-torch26")
        monkeypatch.setenv("GOLLIE_SAFETENSORS_CONVERT_AUTHORIZED", "1")
        with pytest.raises(SystemExit):
            conversion.main()
        assert message in capsys.readouterr().out
        derived = conversion.derived_snapshot_dir(models)
        assert not conversion.conversion_pointer(runs).exists()
        assert not derived.exists()
        assert not derived.with_name(derived.name + ".partial").exists()
        assert _shard_hashes(snapshot) == original

    refuse("symlink", "__pycache__ is a symlink", plant_symlink)
    refuse("structure", "__pycache__ has an unexpected structure", plant_file)
    refuse("unknown", "unexpected directory in the origin snapshot: blobs", plant_directory)
    refuse("incomplete", "origin snapshot has an incomplete download", plant_incomplete)
