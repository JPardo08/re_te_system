"""Offline materialization of an accredited cache candidate. No downloads or installs."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cesvima"
sys.path.insert(0, str(SCRIPTS))

import gollie_wheelhouse as wheelhouse  # noqa: E402


PYTHON = Path("/opt/anaconda3/envs/re_te_system_312/bin/python")
BOOTSTRAP = SCRIPTS / "bootstrap_gollie_isolated_venv.sh"


def _wheel(path: Path, name: str, version: str, tag: str = "py3-none-any") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    metadata = f"Name: {name}\nVersion: {version}\n"
    wheel = f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{info}/METADATA", metadata)
        archive.writestr(f"{info}/WHEEL", wheel)


def _signature(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_mtime_ns, stat.S_IMODE(info.st_mode)


def _write_manifest(path: Path, selected: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ambiguous": [],
        "installable_wheelhouse": False,
        "materialized": False,
        "rejected": [],
        "selected": selected,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_digests_are_the_observed_index_records() -> None:
    by_hash = {item["sha256"]: item for item in wheelhouse.published_wheels()}
    cudnn = by_hash["165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f"]
    assert cudnn["filename"] == "nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl"
    torch = by_hash["222be02548c2e74a21a8fbc8e5b8d2eef9f9faee865d70385d2eb1b9aabcbc76"]
    assert torch["filename"] == "torch-2.5.1+cu121-cp312-cp312-linux_x86_64.whl"
    assert torch["url"].startswith("https://download-r2.pytorch.org/whl/cu121/")
    sympy = by_hash["db36cdc64bf61b9b24578b6f7bab1ecdd2452cf008f34faa33776680c26d66f8"]
    assert sympy["filename"] == "sympy-1.13.1-py3-none-any.whl"
    assert wheelhouse.EXP472_MANIFEST_SHA256 == (
        "7308b6ae7c4aa7df5a324b5e057c9c8f655ee958568cc8716e208ca3ebab8357"
    )
    assert wheelhouse.EXP472_SELECTED == 8
    assert wheelhouse.EXP472_REJECTED == 20
    assert wheelhouse.EXP472_AMBIGUOUS == 0
    assert wheelhouse._tags_are_appropriate(
        ["cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64"]
    ) is True
    assert wheelhouse._tags_are_appropriate(
        ["cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64"]
    ) is True
    assert all("cu13" not in item["filename"] for item in by_hash.values())
    assert all(not item["version"].startswith("2.14") for item in by_hash.values())
    assert all(not item["version"].startswith("3.8") for item in by_hash.values())


def test_pending_closure_leaves_setuptools_and_nvjitlink_unversioned() -> None:
    pending = wheelhouse.pending_dependencies()
    assert pending["closure_complete"] is False
    missing = {item["name"] for item in pending["exact_missing"]}
    assert "nvidia-cuda-nvrtc-cu12" in missing
    assert "nvidia-cudnn-cu12" not in missing
    assert len(missing) == 10
    unpinned = {item["name"]: item["specifier"] for item in pending["unpinned"]}
    assert unpinned["setuptools"] == ""
    assert unpinned["nvidia-nvjitlink-cu12"] == ""
    open_names = {item["name"] for item in pending["version_not_accredited"]}
    assert open_names == {"setuptools", "nvidia-nvjitlink-cu12"}
    plan = wheelhouse.download_plan()
    assert plan["download_authorized"] is False
    assert {item["name"] for item in plan["exact_missing"]} == missing
    assert wheelhouse.DOWNLOAD_AUTHORIZED is False
    assert wheelhouse.MATERIALIZATION_AUTHORIZED is False


def test_materialize_copies_one_accredited_body_and_leaves_the_cache(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    cache = scratch / "pip-cache" / "http-v2"
    body = cache / "hashed.body"
    _wheel(body, "sympy", "1.13.1")
    other = cache / "other.body"
    _wheel(other, "filelock", "3.16.1")
    before = {path: _signature(path) for path in cache.iterdir()}
    selection = wheelhouse.select_records(wheelhouse.scan_cache(cache))
    selected = selection["selected"]
    assert [item["declared_name"] for item in selected] == ["sympy"]
    digest = hashlib.sha256(body.read_bytes()).hexdigest()
    manifest = scratch / "recovery" / "manifest.json"
    expected = _write_manifest(manifest, selected)
    published = (
        {
            "filename": "sympy-1.13.1-py3-none-any.whl",
            "name": "sympy",
            "provenance": "test-published-digest",
            "sha256": digest,
            "version": "1.13.1",
        },
    )
    destination = scratch / "recovery" / "gollie-wheelhouse"
    report = wheelhouse.materialize_accredited(
        manifest,
        cache,
        destination,
        authorized=True,
        expected_sha256=expected,
        scratch_root=scratch,
        published=published,
    )
    copied = destination / "sympy-1.13.1-py3-none-any.whl"
    assert report["materialized"] is True
    assert report["installable_wheelhouse"] is False
    assert report["closure_complete"] is False
    assert report["download_authorized"] is False
    assert copied.is_file()
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest
    sums = (destination / "SHA256SUMS").read_text(encoding="utf-8")
    assert sums == f"{digest}  sympy-1.13.1-py3-none-any.whl\n"
    stored = json.loads((destination / "materialization.json").read_text(encoding="utf-8"))
    assert stored["copied"][0]["sha256"] == digest
    assert stored["copied"][0]["provenance"] == "test-published-digest"
    assert {path: _signature(path) for path in cache.iterdir()} == before
    with pytest.raises(ValueError, match="existing directory"):
        wheelhouse.materialize_accredited(
            manifest,
            cache,
            destination,
            authorized=True,
            expected_sha256=expected,
            scratch_root=scratch,
            published=published,
        )
    assert copied.is_file()
    assert {path: _signature(path) for path in cache.iterdir()} == before


def test_unaccredited_and_mismatched_records_are_not_copied(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    cache = scratch / "pip-cache" / "http-v2"
    _wheel(cache / "sympy.body", "sympy", "1.13.1")
    selected = wheelhouse.select_records(wheelhouse.scan_cache(cache))["selected"]
    manifest = scratch / "recovery" / "manifest.json"
    expected = _write_manifest(manifest, selected)
    destination = scratch / "recovery" / "wheelhouse"
    blocked = wheelhouse.materialize_accredited(
        manifest,
        cache,
        destination,
        authorized=True,
        expected_sha256=expected,
        scratch_root=scratch,
        published=(),
    )
    assert blocked["materialized"] is False
    assert blocked["blocked"][0]["reason"] == "no published sha256 matches this body"
    assert not destination.exists()
    tampered = json.loads(json.dumps(selected))
    tampered[0]["declared_name"] = "torch"
    tampered_manifest = scratch / "recovery" / "tampered.json"
    tampered_sha = _write_manifest(tampered_manifest, tampered)
    digest = selected[0]["sha256"]
    published = (
        {
            "filename": "sympy-1.13.1-py3-none-any.whl",
            "name": "sympy",
            "provenance": "test-published-digest",
            "sha256": digest,
            "version": "1.13.1",
        },
    )
    mismatch = wheelhouse.materialize_accredited(
        tampered_manifest,
        cache,
        destination,
        authorized=True,
        expected_sha256=tampered_sha,
        scratch_root=scratch,
        published=published,
    )
    assert mismatch["materialized"] is False
    assert "METADATA name" in mismatch["blocked"][0]["reason"]
    assert not destination.exists()
    with pytest.raises(ValueError, match="manifest sha256"):
        wheelhouse.materialize_accredited(
            manifest,
            cache,
            destination,
            authorized=True,
            expected_sha256="0" * 64,
            scratch_root=scratch,
            published=published,
        )
    with pytest.raises(RuntimeError, match="not authorized"):
        wheelhouse.materialize_accredited(
            manifest,
            cache,
            destination,
            authorized=False,
            expected_sha256=expected,
            scratch_root=scratch,
            published=published,
        )
    with pytest.raises(ValueError, match="inside the pip cache"):
        wheelhouse.materialize_accredited(
            manifest,
            cache,
            cache / "wheelhouse",
            authorized=True,
            expected_sha256=expected,
            scratch_root=scratch,
            published=published,
        )
    with pytest.raises(ValueError, match="SCRATCH_ROOT"):
        wheelhouse.materialize_accredited(
            manifest,
            cache,
            tmp_path / "outside",
            authorized=True,
            expected_sha256=expected,
            scratch_root=scratch,
            published=published,
        )
    with pytest.raises(ValueError, match="venv"):
        wheelhouse.materialize_accredited(
            manifest,
            cache,
            scratch / "venvs" / "gollie-l40s" / "wheels",
            authorized=True,
            expected_sha256=expected,
            scratch_root=scratch,
            published=published,
        )
    assert not destination.exists()


def test_materialize_command_stays_closed_without_the_authorization_flag() -> None:
    with pytest.raises(RuntimeError, match="not authorized"):
        wheelhouse.main(["materialize"])
    completed = subprocess.run(
        ["bash", str(BOOTSTRAP), "materialize"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "not authorized" in completed.stderr
    scratch = Path("/tmp")
    manifest = scratch / "recovery" / "gollie-wheel-manifest-exp472-20261027.json"
    with pytest.raises(ValueError, match="refusing manifest"):
        wheelhouse.assert_exp472_manifest_path(scratch, scratch / "other.json")
    assert wheelhouse.exp472_manifest_path(scratch) == manifest


@pytest.mark.skipif(not PYTHON.is_file(), reason="canonical Python 3.12 is absent")
def test_flash_attention_pin_remains_and_install_stays_closed() -> None:
    pin = (SCRIPTS / "flash_attn_pin.py").read_text(encoding="utf-8")
    assert "GOLLIE_J16_DOWNLOAD_AUTHORIZED = False" in pin
    gold = (SCRIPTS / "j21_qwen3_invented_smoke.py").read_text(encoding="utf-8")
    assert "C0/C1 Gold inference is not authorized" in gold
    install = (SCRIPTS / "bootstrap_gollie_isolated_venv.sh").read_text(encoding="utf-8")
    assert 'die "R1 install is not authorized"' in install
    assert "--ignore-installed" in install
    assert "--no-index" in install
    assert "--only-binary=:all:" in install
