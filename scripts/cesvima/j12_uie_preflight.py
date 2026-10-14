"""J12: one-batch UIE preflight. API breakage exits non-zero and does not patch upstream."""

from __future__ import annotations

import copy
import json
import os
import sys
import traceback
from pathlib import Path

from cesvima_common import fail, read_json, require_cuda_family, write_json
from uie_fingerprint import build_fingerprint, fingerprint_digest


def _shot_dir() -> Path:
    pointer = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "shot_dir.txt"
    if not pointer.is_file():
        fail(f"J09 shot pointer is missing: {pointer}")
    return Path(pointer.read_text(encoding="utf-8").strip())


def _base_snapshot() -> Path:
    pointer = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "base_snapshot.json"
    if not pointer.is_file():
        fail(f"J11 base snapshot pointer is missing: {pointer}")
    record = read_json(pointer)
    if record.get("revision") != "966f8b1fc4c74e94ab552081605913ad5133cc41":
        fail(f"base snapshot revision is {record.get('revision')}")
    return Path(record["local_path"])


def _first_train_row(shot_dir: Path) -> dict:
    for line in (shot_dir / "train.json").read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    fail("10shot train.json is empty")


def _run_batch(uie_root: Path, shot_dir: Path, model_path: Path) -> None:
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


def main() -> int:
    uie_root = Path(os.environ["P3_UPSTREAM_ROOT"]) / "UIE"
    shot_dir = _shot_dir()
    model_path = _base_snapshot()
    report_dir = Path(os.environ["P3_RUNS_ROOT"]) / "uie" / "preflight"
    fingerprint = None
    error = None
    try:
        fingerprint = build_fingerprint(uie_root, shot_dir)
        _run_batch(uie_root, shot_dir, model_path)
        status = "PASS"
    except SystemExit as exc:
        status = "FAIL"
        error = str(exc)
    except Exception:
        status = "FAIL"
        error = traceback.format_exc()
    if fingerprint is None:
        try:
            fingerprint = build_fingerprint(uie_root, shot_dir)
        except Exception as exc:
            fingerprint = {"fingerprint_error": str(exc)}
    digest = fingerprint_digest(fingerprint) if "script_hashes" in fingerprint else None
    write_json(report_dir / "environment_fingerprint.json", fingerprint)
    write_json(
        report_dir / "preflight_report.json",
        {
            "error": error,
            "fingerprint_sha256": digest,
            "status": status,
        },
    )
    if status != "PASS":
        print(error or "FAIL: UIE preflight failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
