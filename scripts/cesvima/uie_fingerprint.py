"""Environment fingerprint shared by UIE J12 and J13."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from cesvima_common import canonical_json, fail, sha256_file


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


def build_fingerprint(uie_root: Path, shot_dir: Path) -> dict:
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
        "extra_packages": extra,
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
