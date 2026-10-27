"""EXP.4.9 recovery plan. Copies nothing from the network and does not install."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
sys.path.insert(0, str(SCRIPTS))

import gollie_wheelhouse as wheelhouse  # noqa: E402


BOOTSTRAP = SCRIPTS / "bootstrap_gollie_isolated_venv.sh"
UNRESOLVED = {
    "click",
    "filelock",
    "fsspec",
    "hf-xet",
    "jinja2",
    "mpmath",
    "mypy-extensions",
    "networkx",
    "numpy",
    "packaging",
    "pathspec",
    "platformdirs",
    "pyyaml",
    "regex",
    "requests",
    "safetensors",
    "tokenizers",
    "tqdm",
    "typing-extensions",
}


def _wheel(path: Path, name: str, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{info}/METADATA", f"Name: {name}\nVersion: {version}\n")
        archive.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )


def test_recovery_plan_keeps_the_cuda_pins_and_records_the_two_reproduced_wheels() -> None:
    official = {item["name"]: item for item in wheelhouse.OFFICIAL_LINUX_WHEELS}
    assert "nvidia-nvjitlink-cu12" not in official
    assert "setuptools" not in official
    plan = wheelhouse.recovery_plan()
    nvidia = [item for item in plan["selective_download"] if item["pinned_by_torch"] is True]
    assert len(nvidia) == 10
    for item in nvidia:
        source = official[item["name"]]
        assert item["version"] == source["version"]
        assert item["sha256"] == source["sha256"]
        assert item["url"] == source["url"]
        assert item["filename"] == source["filename"]
        assert item["download_authorized"] is False
    reproduced = {
        item["name"]: item
        for item in plan["selective_download"]
        if item["pinned_by_torch"] is False
    }
    assert set(reproduced) == {"setuptools", "nvidia-nvjitlink-cu12"}
    setuptools = reproduced["setuptools"]
    assert setuptools["version"] == "78.1.0"
    assert setuptools["filename"] == "setuptools-78.1.0-py3-none-any.whl"
    assert setuptools["sha256"] == "3e386e96793c8702ae83d17b853fb93d3e09ef82ec62722e61da5cd22376dcd8"
    assert setuptools["requires_dist"] == []
    assert setuptools["source"] == "gollie-l40s"
    nvjitlink = reproduced["nvidia-nvjitlink-cu12"]
    assert nvjitlink["version"] == "12.9.86"
    assert nvjitlink["filename"] == (
        "nvidia_nvjitlink_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl"
    )
    assert nvjitlink["sha256"] == "e3f1171dbdc83c5932a45f0f4c99180a70de9bd2718c1ab77d14104f6d7147f9"
    assert nvjitlink["requires_dist"] == []
    assert wheelhouse._tags_are_appropriate(
        ["py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64"]
    ) is True
    assert plan["nvjitlink_runtime_compatibility"] == "pending_r1_r2"
    assert plan["closure_complete"] is False
    assert plan["installable_wheelhouse"] is False
    assert plan["offline_resolve_pass"] is False
    assert plan["download_authorized"] is False
    assert plan["offline_resolve"] == list(wheelhouse.OFFLINE_RESOLVE_ARGUMENTS)
    unresolved = {item["name"]: item["status"] for item in plan["unresolved_transitive"]}
    assert set(unresolved) == UNRESOLVED
    assert set(unresolved.values()) == {"VERSION_NOT_OBSERVED"}
    unpinned = {item["name"]: item["specifier"] for item in wheelhouse.pending_dependencies()["unpinned"]}
    assert unpinned["nvidia-nvjitlink-cu12"] == ""
    assert unpinned["setuptools"] == ""
    assert wheelhouse.DOWNLOAD_AUTHORIZED is False
    with pytest.raises(RuntimeError, match="not authorized"):
        wheelhouse.fetch_planned_wheels()


def test_offline_resolve_contract_is_not_a_pass() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")
    dry_run = script.split("dry_run_stack()", 1)[1].split("load_pin_shell_vars()", 1)[0]
    previous = 0
    for flag in wheelhouse.OFFLINE_RESOLVE_ARGUMENTS:
        found = dry_run.index(flag)
        assert found >= previous
        previous = found
    assert "offline_resolve_pass" not in script
    recover = script.split("    recover)", 1)[1].split("    *)", 1)[0]
    assert "pip" not in recover
    assert "GOLLIE_RECOVERY_AUTHORIZED" in recover
    assert 'die "R1 install is not authorized"' in script


def test_reuse_copies_an_accredited_exp48_wheel_without_writing_the_source(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    source = scratch / "recovery" / "gollie-wheelhouse-exp48"
    wheel = source / "sympy-1.13.1-py3-none-any.whl"
    _wheel(wheel, "sympy", "1.13.1")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    before = wheel.stat().st_mtime_ns
    published = (
        {
            "filename": wheel.name,
            "name": "sympy",
            "sha256": digest,
            "version": "1.13.1",
        },
    )
    destination = scratch / "recovery" / "gollie-wheelhouse-exp49"
    report = wheelhouse.reuse_exp48_wheels(
        source,
        destination,
        authorized=True,
        scratch_root=scratch,
        expected_count=1,
        published=published,
    )
    copied = destination / wheel.name
    assert copied.is_file()
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest
    assert wheel.stat().st_mtime_ns == before
    assert report["closure_complete"] is False
    assert report["installable_wheelhouse"] is False
    assert report["offline_resolve_pass"] is False
    assert report["download_authorized"] is False
    assert (destination / "SHA256SUMS").read_text(encoding="utf-8") == f"{digest}  {wheel.name}\n"
    assert not (source / "recovery-plan.json").exists()
    with pytest.raises(ValueError, match="EXP.4.8"):
        wheelhouse.reuse_exp48_wheels(
            source,
            source,
            authorized=True,
            scratch_root=scratch,
            expected_count=1,
            published=published,
        )
    with pytest.raises(ValueError, match="EXP.4.8"):
        wheelhouse.reuse_exp48_wheels(
            source,
            source / "nested",
            authorized=True,
            scratch_root=scratch,
            expected_count=1,
            published=published,
        )
    assert wheel.stat().st_mtime_ns == before
    assert not (source / "nested").exists()


def test_reuse_rejects_a_version_match_that_is_not_the_published_wheel(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    source = scratch / "recovery" / "gollie-wheelhouse-exp48"
    wheel = source / "sympy-1.13.1-py3-none-any.whl"
    _wheel(wheel, "sympy", "1.13.1")
    published = (
        {
            "filename": wheel.name,
            "name": "sympy",
            "sha256": "0" * 64,
            "version": "1.13.1",
        },
    )
    destination = scratch / "recovery" / "gollie-wheelhouse-exp49"
    with pytest.raises(ValueError, match="published digest"):
        wheelhouse.reuse_exp48_wheels(
            source,
            destination,
            authorized=True,
            scratch_root=scratch,
            expected_count=1,
            published=published,
        )
    assert not destination.exists()
    assert list(source.iterdir()) == [wheel]


def test_recovery_refuses_without_authorization_and_does_not_open_install() -> None:
    with pytest.raises(RuntimeError, match="not authorized"):
        wheelhouse.reuse_exp48_wheels(
            Path("gollie-wheelhouse-exp48"),
            Path("gollie-wheelhouse-exp49"),
            authorized=False,
            scratch_root=Path("/tmp"),
        )
    with pytest.raises(RuntimeError, match="not authorized"):
        wheelhouse.main(["recover"])
    completed = subprocess.run(
        ["bash", str(BOOTSTRAP), "recover"],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "GOLLIE_RECOVERY_AUTHORIZED": ""},
    )
    assert completed.returncode == 1
    assert "not authorized" in completed.stderr
