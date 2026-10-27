"""Offline checks for the isolated GoLLIE prep. No install, GPU, or download."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
BOOTSTRAP = SCRIPTS / "bootstrap_gollie_isolated_venv.sh"
SBATCH = ROOT / "jobs" / "cesvima" / "j15_gollie_isolated_preflight.sbatch"
PYTHON = Path("/opt/anaconda3/envs/re_te_system_312/bin/python")
WHEEL_NAME = "flash_attn-2.8.3.post2-cp312-cp312-linux_x86_64.whl"
EXTENSION = "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
OLD_EXTENSION_SHA = "1fed48745de90c41071d324ff0bbdf17c3c054c679279e5bc30c640b74007a03"
WHEEL_SHA = "8238e887fec100cbead2fb662a041bf8486499df00d8d026f234be3a7af2be82"

sys.path.insert(0, str(SCRIPTS))
from flash_attn_pin import (  # noqa: E402
    EXPECTED_BLACK,
    EXPECTED_EINOPS,
    EXPECTED_HUGGINGFACE_HUB,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    EXPECTED_TRITON,
    FLASH_ATTN_WHEEL_SHA256,
)


def _wheel(directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"not-a-real-wheel")


def _stack(wheelhouse: Path) -> None:
    _wheel(wheelhouse, f"torch-{EXPECTED_TORCH}-cp312-cp312-manylinux1_x86_64.whl")
    _wheel(wheelhouse, f"transformers-{EXPECTED_TRANSFORMERS}-py3-none-any.whl")
    _wheel(
        wheelhouse,
        f"black-{EXPECTED_BLACK}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    )
    _wheel(
        wheelhouse,
        f"triton-{EXPECTED_TRITON}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    )
    _wheel(wheelhouse, f"einops-{EXPECTED_EINOPS}-py3-none-any.whl")
    _wheel(wheelhouse, f"huggingface_hub-{EXPECTED_HUGGINGFACE_HUB}-py3-none-any.whl")


def _layout(tmp_path: Path) -> dict[str, Path]:
    scratch = tmp_path / "scratch"
    original = scratch / "venvs" / "gollie-l40s"
    isolated = scratch / "venvs" / "gollie-isolated"
    original.mkdir(parents=True)
    marker = original / "MARKER"
    marker.write_text("untouched", encoding="utf-8")
    wheel = scratch / "wheels" / WHEEL_NAME
    _wheel(wheel.parent, WHEEL_NAME)
    wheelhouse = scratch / "wheelhouse"
    _stack(wheelhouse)
    return {
        "scratch": scratch,
        "original": original,
        "isolated": isolated,
        "marker": marker,
        "wheel": wheel,
        "wheelhouse": wheelhouse,
    }


def _env(paths: dict[str, Path], **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HF_HOME", "P3_MODELS_ROOT", "P3_UPSTREAM_ROOT", "P3_RUNS_ROOT", "PROJECT_ROOT"):
        env.pop(key, None)
    env.update(
        {
            "SYSTEM_ROOT": str(ROOT),
            "SCRATCH_ROOT": str(paths["scratch"]),
            "GOLLIE_PYTHON312": str(PYTHON),
            "GOLLIE_VENV": str(paths["original"]),
            "GOLLIE_ISOLATED_VENV": str(paths["isolated"]),
            "GOLLIE_FLASH_ATTN_WHEEL": str(paths["wheel"]),
            "GOLLIE_WHEELHOUSE": str(paths["wheelhouse"]),
        }
    )
    env.pop("GOLLIE_ISOLATED_INSTALL", None)
    env.pop("GOLLIE_BOOTSTRAP_ALLOW_COMPILE", None)
    env.update(extra)
    return env


def _run(args: list[str], env: dict[str, str], trace: bool = False) -> subprocess.CompletedProcess[str]:
    command = ["bash"]
    if trace:
        command.append("-x")
    command.append(str(BOOTSTRAP))
    command.extend(args)
    return subprocess.run(command, text=True, capture_output=True, check=False, env=env)


@pytest.mark.skipif(not PYTHON.is_file(), reason="canonical Python 3.12 is absent")
def test_inventory_reports_the_wheel_and_does_not_install(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    completed = _run(["inventory"], _env(paths), trace=True)
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 1
    assert f"!= {WHEEL_SHA}" in combined
    assert "PRESENT torch" in completed.stdout
    assert "PRESENT transformers" in completed.stdout
    assert "PRESENT black" in completed.stdout
    assert "PRESENT triton" in completed.stdout
    assert "PRESENT einops" in completed.stdout
    assert "PRESENT huggingface_hub" in completed.stdout
    assert "INSTALL not performed" in completed.stdout
    assert "--dry-run" not in completed.stderr
    assert " -m venv " not in completed.stderr
    assert " -m pip " not in completed.stderr
    assert not paths["isolated"].exists()
    assert paths["marker"].read_text(encoding="utf-8") == "untouched"


@pytest.mark.skipif(not PYTHON.is_file(), reason="canonical Python 3.12 is absent")
def test_install_stays_closed_and_a_bad_hash_does_not_create_the_venv(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    refused = _run(["install"], _env(paths))
    assert refused.returncode == 1
    assert "R1 install is not authorized" in refused.stderr
    assert not paths["isolated"].exists()
    attempted = _run(["install"], _env(paths, GOLLIE_ISOLATED_INSTALL="1"), trace=True)
    assert attempted.returncode == 1
    assert "--dry-run" not in attempted.stderr
    assert " -m venv " not in attempted.stderr
    assert not paths["isolated"].exists()
    assert paths["marker"].read_text(encoding="utf-8") == "untouched"


@pytest.mark.skipif(not PYTHON.is_file(), reason="canonical Python 3.12 is absent")
def test_inventory_rejects_the_original_venv_compile_and_foreign_wheel(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    original_target = _run(["inventory"], _env(paths, GOLLIE_ISOLATED_VENV=str(paths["original"])))
    assert original_target.returncode == 1
    assert "gollie-l40s" in original_target.stderr
    compile_flag = _run(["inventory"], _env(paths, GOLLIE_BOOTSTRAP_ALLOW_COMPILE="1"))
    assert compile_flag.returncode == 1
    foreign = paths["scratch"].parent / WHEEL_NAME
    foreign.write_bytes(b"outside")
    outside = _run(["inventory"], _env(paths, GOLLIE_FLASH_ATTN_WHEEL=str(foreign)))
    assert outside.returncode == 1
    assert "outside SCRATCH_ROOT" in outside.stderr
    assert paths["marker"].read_text(encoding="utf-8") == "untouched"


@pytest.mark.skipif(not PYTHON.is_file(), reason="canonical Python 3.12 is absent")
def test_wheelhouse_classifier_accepts_one_linux_pin_and_rejects_ambiguity(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    accepted = subprocess.run(
        ["bash", "-c", 'source "$1"; classify_wheelhouse', "bash", str(BOOTSTRAP)],
        text=True,
        capture_output=True,
        check=False,
        env=_env(paths),
    )
    assert accepted.returncode == 0
    assert "MISSING" not in accepted.stdout
    _wheel(paths["wheelhouse"], f"torch-{EXPECTED_TORCH}-cp312-cp312-linux_x86_64.whl")
    ambiguous = subprocess.run(
        ["bash", "-c", 'source "$1"; classify_wheelhouse', "bash", str(BOOTSTRAP)],
        text=True,
        capture_output=True,
        check=False,
        env=_env(paths),
    )
    assert ambiguous.returncode == 1
    assert "AMBIGUOUS torch 2" in ambiguous.stdout
    macos = tmp_path / "macos"
    _wheel(macos, f"torch-{EXPECTED_TORCH}-cp312-cp312-macosx_14_0_arm64.whl")
    _wheel(macos, f"torch-2.5.1+cu118-cp312-cp312-manylinux1_x86_64.whl")
    _stack_without_torch = macos
    for name in (
        f"transformers-{EXPECTED_TRANSFORMERS}-py3-none-any.whl",
        f"black-{EXPECTED_BLACK}-py3-none-any.whl",
        f"triton-{EXPECTED_TRITON}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        f"einops-{EXPECTED_EINOPS}-py3-none-any.whl",
        f"huggingface_hub-{EXPECTED_HUGGINGFACE_HUB}-py3-none-any.whl",
    ):
        _wheel(macos, name)
    missing_torch = subprocess.run(
        ["bash", "-c", 'source "$1"; classify_wheelhouse', "bash", str(BOOTSTRAP)],
        text=True,
        capture_output=True,
        check=False,
        env=_env(paths, GOLLIE_WHEELHOUSE=str(macos)),
    )
    assert missing_torch.returncode == 2
    assert f"MISSING torch {EXPECTED_TORCH}" in missing_torch.stdout
    assert FLASH_ATTN_WHEEL_SHA256 == WHEEL_SHA


@pytest.mark.skipif(not PYTHON.is_file(), reason="canonical Python 3.12 is absent")
def test_assert_prep_accepts_only_the_isolated_record(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    isolated = paths["isolated"]
    extension = isolated / "lib" / "python3.12" / "site-packages" / EXTENSION
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"isolated-extension")
    (isolated / "bin").mkdir()
    (isolated / "bin" / "python").write_text("", encoding="utf-8")
    record = {
        "abi_pass": False,
        "gpu_preflight": False,
        "installed_extension_path": str(extension.resolve()),
        "installed_extension_sha256": "ab" * 32,
        "j16_authorized": False,
        "j17_authorized": False,
        "preflight_pass": False,
        "status": "PREP_PASS",
        "venv": str(isolated.resolve()),
        "wheel_sha256": WHEEL_SHA,
    }
    (isolated / "isolated_prep_record.json").write_text(json.dumps(record), encoding="utf-8")
    env = _env(
        paths,
        GOLLIE_VENV=str(isolated),
        GOLLIE_ORIGINAL_VENV=str(paths["original"]),
        P3_RUNS_ROOT=str(paths["scratch"] / "runs"),
    )
    env.pop("GOLLIE_WHEELHOUSE", None)
    env.pop("GOLLIE_FLASH_ATTN_WHEEL", None)
    accepted = _run(["assert-prep"], env)
    assert accepted.returncode == 0
    assert "PREP_ASSERT PASS" in accepted.stdout
    record["installed_extension_sha256"] = OLD_EXTENSION_SHA
    (isolated / "isolated_prep_record.json").write_text(json.dumps(record), encoding="utf-8")
    rejected = _run(["assert-prep"], env)
    assert rejected.returncode == 1
    assert paths["marker"].read_text(encoding="utf-8") == "untouched"


@pytest.mark.skipif(not PYTHON.is_file(), reason="canonical Python 3.12 is absent")
def test_modeling_module_must_already_be_local(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    cache = paths["scratch"] / "hf" / "hub"
    modeling = (
        cache
        / "models--HiTZ--GoLLIE-7B"
        / "snapshots"
        / "d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9"
        / "modeling_flash_llama.py"
    )
    absent = _run(["assert-modeling"], _env(paths, HF_HUB_CACHE=str(cache)))
    assert absent.returncode == 1
    assert "REFUSING_DOWNLOAD" in absent.stderr
    modeling.parent.mkdir(parents=True)
    modeling.write_text("class Llama: pass\n", encoding="utf-8")
    present = _run(["assert-modeling"], _env(paths, HF_HUB_CACHE=str(cache)))
    assert present.returncode == 0
    assert "LOCAL_MODELING" in present.stdout
    modeling.write_bytes(b"x" * 5_000_001)
    huge = _run(["assert-modeling"], _env(paths, HF_HUB_CACHE=str(cache)))
    assert huge.returncode == 1


def test_r2_selects_the_isolated_venv_after_env_and_stays_out_of_the_submitter() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    executable = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    source_at = text.index("source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh")
    isolated_at = text.index('export GOLLIE_ISOLATED_VENV="${SCRATCH_ROOT}/venvs/gollie-isolated"')
    selected_at = text.index('export GOLLIE_VENV="${GOLLIE_ISOLATED_VENV}"')
    assert source_at < isolated_at < selected_at
    assert text.index("assert-prep") < text.index("assert-modeling") < text.index("j15_gollie_preflight.py")
    assert "HF_HUB_OFFLINE=1" in text
    assert "GOLLIE_R2_PREFLIGHT_AUTHORIZED=1" not in text
    assert text.index("isolated J15 is not authorized") < source_at
    assert "unset GOLLIE_J17_SMOKE_AUTHORIZED" in text
    assert "GOLLIE_J17_SMOKE_AUTHORIZED=1" not in text
    assert all("pip install" not in line and not line.startswith("pip ") for line in executable)
    assert all("GOLLIE_ISOLATED_INSTALL" not in line for line in executable)
    submitter = (SCRIPTS / "submit_train_first.sh").read_text(encoding="utf-8")
    assert "j15_gollie_isolated_preflight.sbatch" not in submitter
    env = (SCRIPTS / "env.sh").read_text(encoding="utf-8")
    assert 'GOLLIE_VENV="${SCRATCH_ROOT}/venvs/gollie-l40s"' in env
    assert "gollie-isolated" not in env
    script = BOOTSTRAP.read_text(encoding="utf-8")
    script_executable = [
        line.strip()
        for line in script.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert all("--index-url" not in line and "pypi" not in line.lower() for line in script_executable)
    assert "download.pytorch.org" not in script
    assert "pip install flash-attn" not in script
    assert "--system-site-packages" not in script
    assert "cp -a" not in script
    assert script.index("--dry-run") < script.index("--ignore-installed") < script.index("--only-binary=:all:")
    assert "--no-index" in script
    assert script.index("--ignore-installed") < script.index("-m venv")
    assert script.index("-m venv") < script.index("--no-deps")
    bootstrap_syntax = subprocess.run(["bash", "-n", str(BOOTSTRAP)], check=False)
    sbatch_syntax = subprocess.run(["bash", "-n", str(SBATCH)], check=False)
    assert bootstrap_syntax.returncode == 0
    assert sbatch_syntax.returncode == 0
    assert stat.S_IXUSR & BOOTSTRAP.stat().st_mode
