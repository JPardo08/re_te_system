"""J12: one-batch UIE preflight. API breakage exits non-zero and does not patch upstream."""

from __future__ import annotations

import copy
import json
import os
import sys
import traceback
from pathlib import Path

from cesvima_common import fail, read_json, require_cuda_family, write_json
from uie_fingerprint import assess_preflight_inputs, build_fingerprint, fingerprint_digest
from uie_origin_pin import ORIGIN_BIN_SHA256, ORIGIN_REVISION


def _first_train_row(shot_dir: Path) -> dict:
    for line in (shot_dir / "train.json").read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    fail("10shot train.json is empty")


def _refuse_pickle_model(model_path: Path) -> None:
    if (model_path / "pytorch_model.bin").is_file() or any(model_path.glob("*.bin")):
        fail("refusing to load pickle weights; origin pytorch_model.bin stays closed")
    if not (model_path / "model.safetensors").is_file():
        fail("derived model.safetensors is absent")


def _run_batch(uie_root: Path, shot_dir: Path, model_path: Path) -> None:
    _refuse_pickle_model(model_path)
    sys.path.insert(0, str(uie_root))
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from uie.extraction.constants import null_span
    from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser
    from uie.extraction.record_schema import RecordSchema
    from uie.seq2seq.data_collator import DataCollatorForMetaSeq2Seq, DynamicSSIGenerator

    if null_span != "<extra_id_6>":
        fail(f"upstream null span is {null_span}")
    device_name = require_cuda_family("V100")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    model.to("cuda")
    schema = RecordSchema.read_from_file(str(shot_dir / "record.schema"))
    sampler = DynamicSSIGenerator(
        tokenizer=tokenizer,
        schema=schema,
        positive_rate=1,
        negative=-1,
        ordered_prompt=False,
    )
    model.eval()
    if not sampler.full_spot(shuffle=False) or not sampler.full_asoc(shuffle=False):
        fail("full ordered SSI prefixes are empty")
    row = _first_train_row(shot_dir)
    noiser = SpotAsocNoiser(spot_noise_ratio=0.1, asoc_noise_ratio=0.1, null_span=null_span)
    noisy = noiser.add_noise(
        copy.deepcopy(row["spot_asoc"]),
        spot_label_list=list(sampler.spot_list),
        asoc_label_list=list(sampler.asoc_list),
    )
    if not isinstance(noisy, list):
        fail("spot/asoc noise path did not return a list")
    collator = DataCollatorForMetaSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        max_length=256,
        max_prefix_length=-1,
        max_target_length=192,
        negative_sampler=sampler,
        spot_asoc_nosier=noiser,
        decoding_format="spotasoc",
    )
    encoded = tokenizer(row["text"], truncation=True, max_length=256, add_special_tokens=True)
    feature = {
        "asocs": list(row["asoc"]),
        "attention_mask": list(encoded["attention_mask"]),
        "input_ids": list(encoded["input_ids"]),
        "labels": tokenizer.encode(row["record"]),
        "sample_prompt": True,
        "spot_asoc": copy.deepcopy(row["spot_asoc"]),
        "spots": list(row["spot"]),
    }
    model.train()
    batch = collator([feature])
    batch = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    if loss is None or not torch.isfinite(loss).item():
        fail(f"forward loss is not finite: {loss}")
    loss.backward()
    model.eval()
    generated = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        do_sample=False,
        max_length=32,
        num_beams=1,
    )
    if generated.shape[0] != 1:
        fail("generation did not return one sequence")
    print(f"PASS J12 batch on {device_name} loss={float(loss.detach().cpu()):.6f}", flush=True)


def _job_id() -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id or job_id in {".", ".."} or "/" in job_id or "\\" in job_id:
        fail("J12 requires a single-path-segment SLURM_JOB_ID")
    return job_id


def _optional_json_path(pointer: Path, key: str) -> Path | None:
    if not pointer.is_file():
        return None
    record = read_json(pointer)
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value)


def _write_once(path: Path, payload: dict) -> None:
    if path.name == "preflight_report.json" and path.parent.name == "preflight":
        fail("J12 refuses to write the legacy preflight path")
    if path.exists():
        fail(f"J12 refuses to overwrite existing evidence: {path}")
    write_json(path, payload)


def main() -> int:
    job_id = _job_id()
    runs = Path(os.environ["P3_RUNS_ROOT"])
    report_dir = runs / "uie" / "preflight" / "jobs" / job_id
    report_path = report_dir / "preflight_report.json"
    fingerprint_path = report_dir / "environment_fingerprint.json"
    if report_path.exists() or fingerprint_path.exists():
        fail(f"J12 refuses to overwrite existing evidence: {report_path}")
    shot_pointer = runs / "uie" / "shot_dir.txt"
    shot_dir = None
    if shot_pointer.is_file():
        raw = shot_pointer.read_text(encoding="utf-8").strip()
        if raw:
            shot_dir = Path(raw)
    origin_pointer = runs / "uie" / "base_snapshot.json"
    origin_dir = None
    origin_bin = None
    origin_revision_ok = False
    if origin_pointer.is_file():
        record = read_json(origin_pointer)
        origin_revision_ok = record.get("revision") == ORIGIN_REVISION
        origin_dir = _optional_json_path(origin_pointer, "local_path")
        if origin_dir is not None:
            origin_bin = origin_dir / "pytorch_model.bin"
    derived_pointer = runs / "uie" / "derived_snapshot.json"
    derived_record: dict = {}
    if derived_pointer.is_file():
        loaded_record = read_json(derived_pointer)
        if isinstance(loaded_record, dict):
            derived_record = loaded_record
    model_value = derived_record.get("local_path")
    model_dir = Path(model_value) if isinstance(model_value, str) and model_value.strip() else None
    assessment = assess_preflight_inputs(
        shot_dir,
        model_dir,
        origin_bin,
        origin_dir,
        provenance=derived_record or None,
    )
    if origin_pointer.is_file() and not origin_revision_ok:
        assessment["errors"].append("base snapshot revision is not the pinned origin")
        assessment["ready"] = False
    if assessment["ready"] is not True:
        _write_once(
            report_path,
            {
                "deserialization_authorized": False,
                "derived_model_dir": None if model_dir is None else str(model_dir),
                "derived_weight_hashes": assessment["derived"]["files"],
                "error": "; ".join(assessment["errors"]),
                "fingerprint_sha256": None,
                "frozen_10shot_hashes": {
                    name: item["sha256"] for name, item in assessment["shot"]["files"].items()
                },
                "origin_bin_sha256_pin": ORIGIN_BIN_SHA256,
                "origin_verification": assessment["origin"],
                "slurm_job_id": job_id,
                "status": "FAIL",
            },
        )
        _write_once(
            fingerprint_path,
            {
                "derived_weight_hashes": assessment["derived"]["files"],
                "frozen_10shot_hashes": {
                    name: item["sha256"] for name, item in assessment["shot"]["files"].items()
                },
                "status": "NOT_BUILT",
            },
        )
        print("FAIL: J12 blocked before weight load", flush=True)
        return 1

    uie_root = Path(os.environ["P3_UPSTREAM_ROOT"]) / "UIE"
    assert model_dir is not None
    assert shot_dir is not None
    fingerprint = None
    error = None
    try:
        fingerprint = build_fingerprint(uie_root, shot_dir, model_dir)
        _run_batch(uie_root, shot_dir, model_dir)
        status = "PREFLIGHT_PASS"
    except SystemExit as exc:
        status = "FAIL"
        error = str(exc)
    except Exception:
        status = "FAIL"
        error = traceback.format_exc()
    if fingerprint is None:
        fingerprint = {"fingerprint_error": error or "fingerprint was not built"}
    digest = fingerprint_digest(fingerprint) if "script_hashes" in fingerprint else None
    if status == "PREFLIGHT_PASS":
        pointer = runs / "uie" / "preflight" / "accepted_job_id.txt"
        if pointer.exists():
            fail(f"J12 refuses to overwrite existing evidence: {pointer}")
    _write_once(fingerprint_path, fingerprint)
    _write_once(
        report_path,
        {
            "deserialization_authorized": False,
            "derived_model_dir": str(model_dir),
            "derived_weight_hashes": fingerprint.get("derived_weight_hashes", {}),
            "error": error,
            "fingerprint_sha256": digest,
            "frozen_10shot_hashes": fingerprint.get("frozen_10shot_hashes", {}),
            "origin_bin_sha256_pin": ORIGIN_BIN_SHA256,
            "origin_verification": assessment["origin"],
            "slurm_job_id": job_id,
            "status": status,
        },
    )
    if status == "PREFLIGHT_PASS":
        pointer.write_text(job_id + "\n", encoding="utf-8")
        return 0
    print(error or "FAIL: UIE preflight failed", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
