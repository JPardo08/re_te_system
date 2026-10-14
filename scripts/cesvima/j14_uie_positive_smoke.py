"""J14: clean-room positive smoke of the J13 checkpoint on the frozen test example."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cesvima_common import fail, read_json, write_json


def _checkpoint() -> Path:
    manifest = read_json(Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "training_manifest.json")
    output_dir = Path(manifest["output_dir"])
    best = manifest.get("best_model_checkpoint")
    candidate = Path(best) if best else output_dir
    if not (candidate / "config.json").is_file():
        candidate = output_dir
    if not (candidate / "config.json").is_file():
        fail(f"no Transformers config.json under {output_dir}")
    return candidate


def _schema(example_dir: Path) -> Path:
    manifest = read_json(Path(os.environ["UIE_MANIFEST"]))
    payload = {
        "association_labels": manifest["split"]["relation_labels"],
        "schema_id": "conll04-relation-10shot-seed1",
        "schema_version": "uie-structural-schema-v1",
        "spot_labels": manifest["split"]["entity_labels"],
        "spot_to_association": manifest["split"]["spot_to_association"],
    }
    path = example_dir / "schema.json"
    write_json(path, payload)
    return path


def main() -> int:
    example = read_json(Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "smoke_example.json")
    work = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "positive_smoke"
    work.mkdir(parents=True, exist_ok=True)
    documents = work / "documents.jsonl"
    documents.write_text(
        json.dumps(
            {
                "document_id": example["document_id"],
                "source_text": example["source_text"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    schema = _schema(work)
    checkpoint = _checkpoint()
    output_root = work / "run"
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "re_te_system.cli",
        "run",
        "--extractor",
        "uie",
        "--documents",
        str(documents),
        "--uie-schema-file",
        str(schema),
        "--output-root",
        str(output_root),
        "--model-name",
        str(checkpoint),
        "--uie-training-manifest",
        str(Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "training_manifest.json"),
        "--local-files-only",
        "--device",
        "cuda",
        "--dtype",
        "float32",
        "--input-language",
        "en",
        "--uie-max-source-tokens",
        "256",
        "--uie-max-target-tokens",
        "192",
        "--uie-num-beams",
        "1",
        "--seed",
        "42",
        "--window-strategy",
        "NONE",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout, flush=True)
        print(completed.stderr, flush=True)
        fail(f"clean-room UIE smoke exited {completed.returncode}")
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    run_dir = Path(summary["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    model = manifest["model"]
    provenance = model.get("checkpoint_provenance") or {}
    base_revision = "966f8b1fc4c74e94ab552081605913ad5133cc41"
    resolved = model.get("resolved_revision")
    if resolved == base_revision or model.get("requested_revision") == base_revision:
        fail("fine-tuned UIE weights are recorded with the uie-base-en revision")
    if not str(resolved).startswith("local-weights:"):
        fail(f"fine-tuned revision is not a local weight identity: {resolved}")
    if provenance.get("base_revision") != base_revision:
        fail("base revision is missing from the fine-tuned provenance record")
    if provenance.get("base_model_id") != "luyaojie/uie-base-en":
        fail("base model id is missing from the fine-tuned provenance record")
    if provenance.get("weights_identity") != resolved:
        fail("weight identity does not match resolved_revision")
    if not provenance.get("weight_files"):
        fail("fine-tuned provenance has no weight hashes")
    training_run = provenance.get("training_run") or {}
    if not training_run.get("manifest_sha256") or not training_run.get("output_dir"):
        fail("fine-tuned provenance has no training-run identity")
    if Path(provenance.get("checkpoint_path", "")).resolve() != checkpoint.resolve():
        fail("provenance checkpoint path is not the checkpoint that was loaded")
    prediction = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    structures = prediction.get("parsed_structures") or []
    parsed = prediction.get("parsed") or []
    predicted_relations = sorted(
        {triple.get("relation") for triple in parsed if isinstance(triple, dict) and triple.get("relation")}
    )
    gold = example.get("gold_relation_types") or []
    recovered = sorted(set(gold) & set(predicted_relations))
    report = {
        "document_id": example["document_id"],
        "gold_relation_types": gold,
        "predicted_relations": predicted_relations,
        "recovered_relations": recovered,
        "relation_recovery": "observed" if recovered else "not_observed",
        "run_dir": str(run_dir),
        "structure_count": len(structures),
        "technical_status": "PASS" if structures else "FAIL",
    }
    write_json(work / "relation_recovery.json", report)
    if not structures:
        fail("positive smoke produced an empty Spot-Association structure")
    print(
        f"PASS J14 structures={len(structures)} relation_recovery={report['relation_recovery']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
