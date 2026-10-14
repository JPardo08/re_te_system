"""J09: rebuild CoNLL04 on SCRATCH from the pinned clones and validate hashes."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from cesvima_common import fail, read_json, sha256_file


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env, text=True)
    if completed.returncode != 0:
        fail(f"command failed with status {completed.returncode}: {command[0]}")


def _load_processing(path: Path):
    spec = importlib.util.spec_from_file_location("sincere_processing", path)
    if spec is None or spec.loader is None:
        fail(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.processing_sentence


def _write_intermediate(processing_sentence, source: Path, destination: Path) -> None:
    dataset = json.loads(source.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    for split_name, rows in dataset.items():
        output = destination / f"{split_name}.jsonlines"
        lines = [json.dumps(processing_sentence(row)) for row in rows]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _expect(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        fail(f"{label} sha256 {observed} != frozen {expected} ({path})")
    print(f"PASS hash {label}", flush=True)


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        fail("J09 must run with PYTHONHASHSEED=0")
    manifest = read_json(Path(os.environ["UIE_MANIFEST"]))
    uie_root = Path(os.environ["P3_UPSTREAM_ROOT"]) / "UIE"
    processing_root = uie_root / "dataset_processing"
    source = Path(os.environ["P3_UPSTREAM_ROOT"]) / "sincere" / "data" / "conll04.json"
    if not source.is_file():
        fail(f"pinned sincere CoNLL04 file is missing: {source}")
    _expect(source, manifest["dataset"]["source"]["file_sha256"], "sincere conll04.json")

    staged = processing_root / "data" / "sincere"
    staged.mkdir(parents=True, exist_ok=True)
    staged_file = staged / "conll04.json"
    staged_file.write_bytes(source.read_bytes())
    _expect(staged_file, manifest["dataset"]["source"]["file_sha256"], "staged conll04.json")

    processing_sentence = _load_processing(
        processing_root / "scripts" / "sincere_processing.py"
    )
    intermediate = processing_root / "data" / "relation" / "conll04"
    _write_intermediate(processing_sentence, staged_file, intermediate)
    intermediate_hashes = manifest["dataset"]["intermediate"]
    for name, key in (
        ("train.jsonlines", "train_sha256"),
        ("dev.jsonlines", "dev_sha256"),
        ("test.jsonlines", "test_sha256"),
    ):
        _expect(intermediate / name, intermediate_hashes[key], name)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = str(processing_root)
    python = sys.executable
    _run(
        [
            python,
            "uie_convert.py",
            "-format",
            "spotasoc",
            "-config",
            "data_config/relation/conll04.yaml",
            "-output",
            "relation",
        ],
        processing_root,
        env,
    )
    full_dir = processing_root / "converted_data" / "text2spotasoc" / "relation" / "conll04"
    full_hashes = manifest["dataset"]["converted_full"]
    for name, key in (
        ("train.json", "train_sha256"),
        ("val.json", "validation_sha256"),
        ("test.json", "test_sha256"),
    ):
        _expect(full_dir / name, full_hashes[key], f"full {name}")

    shot_parent = (
        processing_root
        / "converted_data"
        / "text2spotasoc"
        / "relation"
        / "conll04_shot"
        / "seed1"
    )
    _run(
        [
            python,
            "scripts/sample_data_shot.py",
            "-src",
            "converted_data/text2spotasoc/relation/conll04",
            "-tgt",
            "converted_data/text2spotasoc/relation/conll04_shot/seed1",
            "-task",
            "relation",
            "-seed",
            "1",
        ],
        processing_root,
        env,
    )
    shot_dir = shot_parent / "10shot"
    split = manifest["split"]
    _expect(shot_dir / "train.json", split["split_hash"], "10shot train")
    _expect(shot_dir / "val.json", split["validation_sha256"], "10shot val")
    _expect(shot_dir / "test.json", split["test_sha256"], "10shot test")
    for name, key in (
        ("record.schema", "record_schema_sha256"),
        ("entity.schema", "entity_schema_sha256"),
        ("relation.schema", "relation_schema_sha256"),
        ("event.schema", "event_schema_sha256"),
    ):
        _expect(shot_dir / name, split[key], name)

    report_path = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "rebuild_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    validator = Path(os.environ["SYSTEM_ROOT"]) / "scripts" / "reproductions" / "uie_conll04" / "validate_reproduction.py"
    _run(
        [
            python,
            str(validator),
            "--uie-repo",
            str(uie_root),
            "--full-dir",
            str(full_dir),
            "--shot-dir",
            str(shot_dir),
            "--output",
            str(report_path),
        ],
        uie_root,
        env,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_combined = manifest["dataset"]["validation_report"]["report_combined_hash"]
    if report.get("combined_hash") != expected_combined:
        fail(
            "validation combined hash "
            f"{report.get('combined_hash')} != frozen {expected_combined}"
        )
    counts = {
        "train": split["train_examples"],
        "validation": split["validation_examples"],
        "test": split["test_examples"],
    }
    for key, expected in counts.items():
        observed = report["shot"][key]["rows"]
        if observed != expected:
            fail(f"10shot {key} rows {observed} != {expected}")
    pointer = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "shot_dir.txt"
    pointer.write_text(str(shot_dir) + "\n", encoding="utf-8")
    print("PASS J09 CoNLL04 10-shot rebuild", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
