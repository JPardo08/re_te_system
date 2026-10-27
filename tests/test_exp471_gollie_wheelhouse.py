"""Offline wheelhouse selection. No cache writes, downloads, or installs."""

from __future__ import annotations

import hashlib
import json
import os
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
from flash_attn_pin import GOLLIE_J16_DOWNLOAD_AUTHORIZED, GOLLIE_J17_SMOKE_AUTHORIZED  # noqa: E402


PYTHON = Path("/opt/anaconda3/envs/re_te_system_312/bin/python")
BOOTSTRAP = SCRIPTS / "bootstrap_gollie_isolated_venv.sh"
NVIDIA_MISSING = {
    "nvidia-cuda-nvrtc-cu12": "12.1.105",
    "nvidia-cuda-runtime-cu12": "12.1.105",
    "nvidia-cuda-cupti-cu12": "12.1.105",
    "nvidia-cublas-cu12": "12.1.3.1",
    "nvidia-cufft-cu12": "11.0.2.54",
    "nvidia-curand-cu12": "10.3.2.106",
    "nvidia-cusolver-cu12": "11.4.5.107",
    "nvidia-cusparse-cu12": "12.1.0.106",
    "nvidia-nccl-cu12": "2.21.5",
    "nvidia-nvtx-cu12": "12.1.105",
}


def _wheel(path: Path, name: str, version: str, requires: tuple[str, ...] = (), tag: str = "py3-none-any") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    metadata = f"Name: {name}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {item}\n" for item in requires)
    wheel = f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{info}/METADATA", metadata)
        archive.writestr(f"{info}/WHEEL", wheel)


def _signature(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_mtime_ns, stat.S_IMODE(info.st_mode)


def test_closure_names_the_ten_cu121_nvidia_wheels_and_leaves_setuptools_unpinned() -> None:
    closure = wheelhouse.prepared_closure()
    missing = {item["name"]: item["version"] for item in closure["exact_missing"]}
    assert missing == NVIDIA_MISSING
    for item in closure["exact_missing"]:
        assert item["status"] == "MISSING"
    findings = {(item["name"], item["version"]): item for item in closure["cache_findings"]}
    present = findings[("sympy", "1.13.1")]
    assert present["status"] == "PRESENT_IN_CACHE"
    assert present["accreditation"] == "NOT_ACCREDITED"
    assert "sha256" not in present
    assert "origin_path" not in present
    assert findings[("sympy", "1.14.0")]["status"] == "INCOMPATIBLE"
    assert "nvidia-cudnn-cu12" not in missing
    assert all(not version.startswith("12.4") for version in missing.values())
    assert all("cu13" not in name for name in missing)
    unpinned = {item["name"]: item["specifier"] for item in closure["unpinned"]}
    assert "setuptools" in unpinned
    assert "nvidia-nvjitlink-cu12" in unpinned
    assert ">=4.8.0" in unpinned["typing-extensions"]
    assert ">=22.0" in unpinned["packaging"]
    assert "2.14" not in str(closure)
    assert "triton 3.8" not in str(closure["exact_missing"])
    assert closure["provenance_accredited"] is False


def test_download_plan_keeps_official_hashes_and_refuses_to_fetch() -> None:
    plan = wheelhouse.download_plan()
    assert plan["download_authorized"] is False
    names = {item["name"] for item in plan["exact_missing"]}
    assert names == set(NVIDIA_MISSING)
    present = plan["present_not_accredited"]
    assert [(item["name"], item["version"]) for item in present] == [("sympy", "1.13.1")]
    assert "sha256" not in present[0]
    nvrtc = next(item for item in plan["exact_missing"] if item["name"] == "nvidia-cuda-nvrtc-cu12")
    assert nvrtc["sha256"] == "339b385f50c309763ca65456ec75e17bbefcbbf2893f462cb8b90584cd27a1c2"
    assert nvrtc["url"].startswith("https://files.pythonhosted.org/")
    assert "setuptools" in plan["omitted_unpinned"]
    assert "nvidia-nvjitlink-cu12" in plan["omitted_unpinned"]
    assert wheelhouse.DOWNLOAD_AUTHORIZED is False
    with pytest.raises(RuntimeError, match="not authorized"):
        wheelhouse.fetch_planned_wheels()


def test_cache_selection_is_read_only_and_does_not_accredit_metadata(tmp_path: Path) -> None:
    cache = tmp_path / "http-v2" / "a" / "b"
    body = cache / "hashed.body"
    _wheel(body, "nvidia-cuda-nvrtc-cu12", "12.1.105")
    _wheel(cache / "torch-2.14.0.whl", "torch", "2.14.0")
    _wheel(cache / "triton-3.8.0.whl", "triton", "3.8.0")
    _wheel(cache / "nvidia-cublas-cu12-12.4.5.8.whl", "nvidia-cublas-cu12", "12.4.5.8")
    _wheel(cache / "nvidia_cuda_nvrtc_cu13.whl", "nvidia-cuda-nvrtc-cu13", "12.8.0")
    _wheel(cache / "filelock.whl", "filelock", "3.16.1")
    (cache / "not-a-wheel.body").write_bytes(b"http-cache-metadata")
    before = {path: _signature(path) for path in cache.iterdir()}
    records = wheelhouse.scan_cache(tmp_path / "http-v2")
    after = {path: _signature(path) for path in cache.iterdir()}
    assert before == after
    selection = wheelhouse.select_records(records)
    assert selection["provenance_accredited"] is False
    selected = selection["selected"]
    assert [item["declared_name"] for item in selected] == ["nvidia-cuda-nvrtc-cu12"]
    assert selected[0]["provenance_accredited"] is False
    assert selected[0]["sha256"] == hashlib.sha256(body.read_bytes()).hexdigest()
    assert selected[0]["origin_path"] == str(body.resolve())
    assert selected[0]["zip_ok"] is True
    rejected_names = {item.get("declared_name") for item in selection["rejected"]}
    assert {"torch", "triton", "nvidia-cublas-cu12", "nvidia-cuda-nvrtc-cu13"} <= rejected_names
    assert [item["declared_name"] for item in selection["unpinned_observed"]] == ["filelock"]
    destination = tmp_path / "wheelhouse"
    written = wheelhouse.stage_selected(tmp_path / "http-v2", selected, destination)
    assert written[0].is_file()
    assert _signature(body) == before[body]
    with pytest.raises(ValueError, match="inside the pip cache"):
        wheelhouse.stage_selected(tmp_path / "http-v2", selected, cache / "copy")


def test_ambiguous_and_corrupt_cache_entries_are_not_selected(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _wheel(cache / "one.body", "sympy", "1.13.1")
    _wheel(cache / "two.body", "sympy", "1.13.1")
    (cache / "broken.whl").write_bytes(b"PK\x03\x04not-a-zip")
    selection = wheelhouse.select_records(wheelhouse.scan_cache(cache))
    assert selection["selected"] == []
    assert len(selection["ambiguous"]) == 2
    assert {item["status"] for item in selection["ambiguous"]} == {"AMBIGUOUS"}
    assert any(item.get("rejection") == "not a zip wheel" for item in selection["rejected"])
    manifest = cache / "manifest.json"
    assert wheelhouse.manifest_path_is_outside_cache(cache, manifest) is False


def test_sympy_1131_py3_none_any_is_selected_without_materializing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "pip-cache" / "http-v2"
    good = cache / "sympy-1.13.1.body"
    _wheel(good, "sympy", "1.13.1", tag="py3-none-any")
    _wheel(cache / "sympy-1.14.0.body", "sympy", "1.14.0", tag="py3-none-any")
    _wheel(cache / "sympy-win.body", "sympy", "1.13.1", tag="py3-none-win_amd64")
    before = {path: _signature(path) for path in cache.iterdir()}
    selection = wheelhouse.select_records(wheelhouse.scan_cache(cache))
    assert len(selection["selected"]) == 1
    chosen = selection["selected"][0]
    assert chosen["declared_name"] == "sympy"
    assert chosen["declared_version"] == "1.13.1"
    assert chosen["tags"] == ["py3-none-any"]
    assert chosen["status"] == "PRESENT_IN_CACHE"
    assert chosen["accreditation"] == "NOT_ACCREDITED"
    assert chosen["provenance_accredited"] is False
    assert chosen["origin_path"] == str(good.resolve())
    assert selection["ambiguous"] == []
    incompatible = {
        (item.get("declared_version"), item.get("status"))
        for item in selection["rejected"]
    }
    assert ("1.14.0", "INCOMPATIBLE") in incompatible
    assert ("1.13.1", "INCOMPATIBLE") in incompatible
    assert wheelhouse._tags_are_appropriate(["cp312-cp312-manylinux_2_17_x86_64"]) is True
    manifest = tmp_path / "manifests" / "wheels.json"
    monkeypatch.setenv("GOLLIE_PIP_HTTP_CACHE", str(cache))
    monkeypatch.setenv("GOLLIE_WHEEL_MANIFEST", str(manifest))
    assert wheelhouse.main(["select-cache"]) == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["installable_wheelhouse"] is False
    assert payload["materialized"] is False
    assert not (tmp_path / "wheelhouse").exists()
    assert {path: _signature(path) for path in cache.iterdir()} == before


@pytest.mark.skipif(not PYTHON.is_file(), reason="canonical Python 3.12 is absent")
def test_resolve_uses_a_clean_index_free_dry_run_and_install_stays_closed(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    isolated = scratch / "venvs" / "gollie-isolated"
    wheelhouse_dir = scratch / "wheelhouse"
    wheelhouse_dir.mkdir(parents=True)
    (wheelhouse_dir / "torch-2.5.1+cu121-cp312-cp312-manylinux1_x86_64.whl").write_bytes(b"not-a-wheel")
    env = os.environ.copy()
    env.update(
        {
            "SYSTEM_ROOT": str(ROOT),
            "SCRATCH_ROOT": str(scratch),
            "GOLLIE_PYTHON312": str(PYTHON),
            "GOLLIE_ISOLATED_VENV": str(isolated),
            "GOLLIE_FLASH_ATTN_WHEEL": str(scratch / "flash.whl"),
            "GOLLIE_WHEELHOUSE": str(wheelhouse_dir),
            "PIP_NO_CACHE_DIR": "1",
        }
    )
    env.pop("GOLLIE_ISOLATED_INSTALL", None)
    env.pop("GOLLIE_R2_PREFLIGHT_AUTHORIZED", None)
    completed = subprocess.run(
        ["bash", "-x", str(BOOTSTRAP), "resolve"],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    trace = completed.stdout + completed.stderr
    assert "--ignore-installed" in trace
    assert "--no-index" in trace
    assert "--only-binary=:all:" in trace
    assert " -m venv " not in trace
    assert not isolated.exists()
    empty = scratch / "empty-wheelhouse"
    empty.mkdir()
    env["GOLLIE_WHEELHOUSE"] = str(empty)
    empty_run = subprocess.run(
        ["bash", "-x", str(BOOTSTRAP), "resolve"],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    empty_trace = empty_run.stdout + empty_run.stderr
    assert empty_run.returncode != 0
    assert "--ignore-installed" in empty_trace
    assert "--no-index" in empty_trace
    assert "--only-binary=:all:" in empty_trace
    assert not isolated.exists()
    blocked = subprocess.run(
        ["bash", str(BOOTSTRAP), "download"],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert blocked.returncode == 1
    assert "not authorized" in blocked.stderr
    gold = (SCRIPTS / "j21_qwen3_invented_smoke.py").read_text(encoding="utf-8")
    assert "C0/C1 Gold inference is not authorized" in gold
    assert GOLLIE_J16_DOWNLOAD_AUTHORIZED is False
    assert GOLLIE_J17_SMOKE_AUTHORIZED is False
