"""Environment fingerprint shared by UIE J12 and J13."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from cesvima_common import canonical_json, fail, sha256_file
import uie_origin_pin as origin_pin
from uie_safetensors_conversion import derived_provenance_accepted


HASHED_UPSTREAM = (
    "run_uie_finetune.py",
    "uie/seq2seq/data_collator/meta_data_collator.py",
    "uie/extraction/noiser/spot_asoc_noiser.py",
    "config/exp_conf/base_model_conf_sa_shot.ini",
    "dataset_processing/data_config/relation/conll04.yaml",
    "dataset_processing/scripts/sample_data_shot.py",
    "dataset_processing/scripts/sincere_processing.py",
    "dataset_processing/uie_convert.py",
    "scripts/function_code.bash",
    "run_uie_finetune_shot.bash",
    "scripts_exp/run_exp_shot.bash",
)
HASHED_SHOT = ("train.json", "val.json", "test.json", "record.schema")
EXTRA_PACKAGES = ("datasets", "numpy", "PyYAML", "tqdm")


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def frozen_shot_identity(shot_dir: Path) -> dict:
    files: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    for name, expected in origin_pin.FROZEN_10SHOT_SHA256.items():
        path = shot_dir / name
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        digest = sha256_file(path)
        files[name] = {"matches_pin": digest == expected, "sha256": digest}
        if digest != expected:
            errors.append(f"hash mismatch {name}")
    return {"errors": errors, "files": files, "ok": not errors}


def derived_weight_identity(model_dir: Path | None, origin_dir: Path | None = None) -> dict:
    """Identity of a safetensors sibling. A pickle in the directory is a refusal."""
    errors: list[str] = []
    files: dict[str, dict[str, object]] = {}
    if model_dir is None:
        return {"errors": ["derived model directory is absent"], "files": files, "ok": False}
    if origin_dir is not None and model_dir.resolve() == origin_dir.resolve():
        errors.append("derived directory is the origin snapshot")
    if (model_dir / "pytorch_model.bin").is_file() or any(model_dir.glob("*.bin")):
        errors.append("pickle weight is present")
    safetensors = model_dir / "model.safetensors"
    if not safetensors.is_file():
        errors.append("model.safetensors is absent")
    else:
        files["model.safetensors"] = {
            "sha256": sha256_file(safetensors),
            "size": safetensors.stat().st_size,
        }
    return {"errors": errors, "files": files, "ok": not errors}


def assess_preflight_inputs(
    shot_dir: Path | None,
    model_dir: Path | None,
    origin_bin: Path | None,
    origin_dir: Path | None = None,
    provenance: dict | None = None,
) -> dict:
    shot = (
        frozen_shot_identity(shot_dir)
        if shot_dir is not None
        else {"errors": ["shot directory is absent"], "files": {}, "ok": False}
    )
    derived = derived_weight_identity(model_dir, origin_dir)
    origin = (
        origin_pin.verify_origin_bin(origin_bin)
        if origin_bin is not None
        else {
            "deserialization_authorized": False,
            "matches_pin": False,
            "observed": False,
            "sha256": None,
            "size": None,
        }
    )
    errors = list(shot["errors"]) + list(derived["errors"])
    provenance_verdict = derived_provenance_accepted(derived, provenance, origin)
    if not provenance_verdict["accepted"]:
        errors.append("derived provenance is not verified")
    if not origin_pin.CONVERSION_ENABLED:
        errors.append("safetensors conversion is disabled")
    if not origin_pin.UIE_J12_DERIVED_LOAD_AUTHORIZED:
        errors.append("J12 derived load is not authorized")
    identity_ok = bool(derived["ok"])
    return {
        "derived": derived,
        "derived_identity_ok": identity_ok,
        "deserialization_authorized": origin_pin.deserialization_authorized(),
        "errors": errors,
        "origin": origin,
        "provenance": provenance_verdict,
        "ready": bool(
            shot["ok"]
            and identity_ok
            and provenance_verdict["accepted"]
            and origin_pin.CONVERSION_ENABLED
            and origin_pin.UIE_J12_DERIVED_LOAD_AUTHORIZED
        ),
        "shot": shot,
    }


def build_fingerprint(uie_root: Path, shot_dir: Path, model_dir: Path) -> dict:
    content = assess_preflight_inputs(shot_dir, model_dir, None)
    problems = list(content["shot"]["errors"]) + list(content["derived"]["errors"])
    if problems:
        fail("fingerprint content is incomplete: " + "; ".join(problems))
    import torch
    import transformers

    if not torch.cuda.is_available():
        fail("fingerprint requires a visible CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if "A100" in device_name or "a100" in device_name.lower():
        fail(f"A100 is forbidden; visible device is {device_name}")
    commit = subprocess.check_output(
        ["git", "-C", str(uie_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    hashes: dict[str, str] = {}
    for relative in HASHED_UPSTREAM:
        path = uie_root / relative
        if not path.is_file():
            fail(f"upstream file missing for fingerprint: {path}")
        hashes[relative] = sha256_file(path)
    for name in HASHED_SHOT:
        path = shot_dir / name
        if not path.is_file():
            fail(f"shot file missing for fingerprint: {path}")
        hashes[f"seed1/10shot/{name}"] = sha256_file(path)
    config_path = Path(__file__).with_name("uie_train_config.py")
    hashes["re_te_system/scripts/cesvima/uie_train_config.py"] = sha256_file(config_path)
    extra = {name: _version(name) for name in EXTRA_PACKAGES}
    if _version("sentencepiece") == "MISSING" or any(value == "MISSING" for value in extra.values()):
        fail(f"UIE training environment is missing a required package: {extra}")
    return {
        "base_checkpoint_revision": "966f8b1fc4c74e94ab552081605913ad5133cc41",
        "cuda_device_name": device_name,
        "derived_weight_hashes": content["derived"]["files"],
        "extra_packages": extra,
        "frozen_10shot_hashes": {
            name: item["sha256"] for name, item in content["shot"]["files"].items()
        },
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "script_hashes": hashes,
        "sentencepiece": _version("sentencepiece"),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "upstream_uie_commit": commit,
    }


def fingerprint_digest(payload: dict) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
