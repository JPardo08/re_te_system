"""Operational GoLLIE closure. No network, no install, and no sbatch."""

from __future__ import annotations

import hashlib
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
import j15_gollie_preflight as j15  # noqa: E402


PYTHON = Path("/opt/anaconda3/envs/re_te_system_312/bin/python")
GO = SCRIPTS / "gollie_go.sh"


def _wheel(path: Path, name: str, version: str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{info}/METADATA", f"Name: {name}\nVersion: {version}\n")
        archive.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    return path.read_bytes()


def _record(name: str, version: str, payload: bytes, *, download: bool) -> dict[str, object]:
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    return {
        "download": download,
        "filename": filename,
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": f"https://files.pythonhosted.org/packages/aa/{filename}",
        "version": version,
    }


def test_observed_closure_uses_published_linux_wheels() -> None:
    catalog = wheelhouse.closure_catalog()
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in catalog:
        grouped.setdefault(wheelhouse.canonicalize_name(str(item["name"])), []).append(item)
        filename = str(item["filename"])
        assert "cu13" not in filename
        assert "win" not in filename and "macosx" not in filename
        tags = [str(tag) for tag in wheelhouse.parse_wheel_filename(filename)[3]]
        assert wheelhouse._tags_are_appropriate(tags) is True
        if item.get("download") is True:
            assert str(item["url"]).split("/")[2] == "files.pythonhosted.org"
    assert set(wheelhouse.REUSED_DISTRIBUTIONS) <= set(grouped)
    downloaded = [item for item in catalog if item.get("download") is True]
    assert len(downloaded) == 36
    assert wheelhouse.DOWNLOAD_AUTHORIZED is False
    with pytest.raises(RuntimeError, match="not authorized"):
        wheelhouse.fetch_planned_wheels()


def test_close_recovers_a_hashed_cache_wheel_and_rejects_a_version_match(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    cache = scratch / "pip-cache" / "http-v2"
    destination = scratch / "recovery" / "gollie-wheelhouse-exp49"
    destination.mkdir(parents=True)
    good = cache / "good.body"
    payload = _wheel(good, "click", "8.5.0")
    wrong = cache / "wrong.body"
    _wheel(wrong, "click", "8.5.0")
    before = good.stat().st_mtime_ns
    record = _record("click", "8.5.0", payload, download=True)
    calls: list[str] = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return b"not-the-wheel"

    report = wheelhouse.close_wheelhouse(
        cache,
        destination,
        download_authorized=True,
        scratch_root=scratch,
        catalog=(record,),
        fetcher=fetcher,
    )
    copied = destination / str(record["filename"])
    assert report["closure_complete"] is True
    assert report["results"][0]["status"] == "recovered"
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == record["sha256"]
    assert good.stat().st_mtime_ns == before
    assert calls == []
    assert not (scratch / "recovery" / "gollie-wheelhouse-exp48").exists()
    with pytest.raises(ValueError, match="EXP.4.8"):
        wheelhouse.close_wheelhouse(
            cache,
            scratch / "recovery" / "gollie-wheelhouse-exp48",
            download_authorized=True,
            scratch_root=scratch,
            catalog=(record,),
            fetcher=fetcher,
        )


def test_close_downloads_only_a_missing_official_wheel(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    destination = scratch / "recovery" / "gollie-wheelhouse-exp49"
    destination.mkdir(parents=True)
    payload = _wheel(tmp_path / "source.whl", "requests", "2.34.2")
    record = _record("requests", "2.34.2", payload, download=True)
    report = wheelhouse.close_wheelhouse(
        scratch / "missing-cache",
        destination,
        download_authorized=True,
        scratch_root=scratch,
        catalog=(record,),
        fetcher=lambda _url: payload,
    )
    assert report["results"][0]["status"] == "downloaded"
    assert (destination / str(record["filename"])).is_file()
    with pytest.raises(ValueError, match="EXP.4.8"):
        wheelhouse.close_wheelhouse(
            scratch / "missing-cache",
            scratch / "recovery" / "gollie-wheelhouse-exp48",
            download_authorized=True,
            scratch_root=scratch,
            catalog=(record,),
            fetcher=lambda _url: payload,
        )


def test_close_blocks_a_bad_download_without_keeping_the_bytes(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    destination = scratch / "recovery" / "gollie-wheelhouse-exp49"
    destination.mkdir(parents=True)
    payload = _wheel(tmp_path / "source.whl", "idna", "3.20")
    record = _record("idna", "3.20", payload, download=True)
    report = wheelhouse.close_wheelhouse(
        scratch / "missing-cache",
        destination,
        download_authorized=True,
        scratch_root=scratch,
        catalog=(record,),
        fetcher=lambda _url: b"wrong-bytes",
    )
    assert report["closure_complete"] is False
    assert report["blocked"][0]["status"] == "blocked"
    assert list(destination.glob("*.whl")) == []
    with pytest.raises(RuntimeError, match="not authorized"):
        wheelhouse.close_wheelhouse(
            scratch / "missing-cache",
            destination,
            download_authorized=False,
            scratch_root=scratch,
            catalog=(record,),
        )


def test_modeling_fetch_rejects_the_wrong_blob_and_does_not_download_weights(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    with pytest.raises(j15.ModelingIdentityError, match="REFUSING_DOWNLOAD"):
        j15.ensure_modeling_file(cache, authorized=False, fetcher=lambda _url: b"weights")
    assert list(cache.rglob("*")) == []
    with pytest.raises(j15.ModelingIdentityError, match="pinned blob"):
        j15.ensure_modeling_file(cache, authorized=True, fetcher=lambda _url: b"class Llama: pass\n")
    assert list(cache.rglob("*.py")) == []
    assert j15.MODELING_BLOB_OID == "1330bc1ca5c6769f0880555350cfd33ea5ea4fbd"
    assert j15.MODELING_SIZE == 46523
    assert j15.MODELING_URL.endswith("/modeling_flash_llama.py")
    assert "resolve/main" not in j15.MODELING_URL


def test_go_phases_stay_closed_until_each_authorization(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    wheelhouse = scratch / "recovery" / "gollie-wheelhouse-exp49"
    wheelhouse.mkdir(parents=True)
    isolated = scratch / "venvs" / "gollie-isolated"
    env = os.environ.copy()
    env.update(
        {
            "SCRATCH_ROOT": str(scratch),
            "SYSTEM_ROOT": str(ROOT),
            "GOLLIE_PYTHON312": str(PYTHON),
            "GOLLIE_WHEELHOUSE": str(wheelhouse),
            "GOLLIE_ISOLATED_VENV": str(isolated),
            "GOLLIE_FLASH_ATTN_WHEEL": str(scratch / "flash.whl"),
            "HF_HUB_CACHE": str(scratch / "hf"),
            "GOLLIE_PIP_HTTP_CACHE": str(scratch / "pip-cache" / "http-v2"),
        }
    )
    env.pop("GOLLIE_DOWNLOAD_AUTHORIZED", None)
    env.pop("GOLLIE_ISOLATED_INSTALL", None)
    env.pop("GOLLIE_R2_PREFLIGHT_AUTHORIZED", None)
    marker = tmp_path / "sbatch-called"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "sbatch"
    fake.write_text("#!/bin/sh\ntouch \"$SBATCH_MARKER\"\necho Submitted batch job 999\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
    env["SBATCH_MARKER"] = str(marker)
    prepare = subprocess.run(["bash", str(GO), "prepare"], text=True, capture_output=True, check=False, env=env)
    assert prepare.returncode == 1
    assert "not authorized" in prepare.stderr
    assert not (wheelhouse / "wheelhouse-close.json").exists()
    submit = subprocess.run(["bash", str(GO), "submit"], text=True, capture_output=True, check=False, env=env)
    assert submit.returncode == 1
    assert "not authorized" in submit.stderr
    assert not marker.exists()
    text = GO.read_text(encoding="utf-8")
    assert "j15_gollie_isolated_preflight.sbatch" in text
    assert "j16_" not in text and "j17_" not in text and "j13_" not in text
    assert "GOLLIE_R2_PREFLIGHT_AUTHORIZED=1" in text
    assert "GOLLIE_FLASH_ATTN_WHEEL" in text


def test_creator_prefers_the_pyvenv_home_and_can_fall_back_to_gollie_l40s(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    original = scratch / "venvs" / "gollie-l40s"
    (original / "bin").mkdir(parents=True)
    marker = original / "MARKER"
    marker.write_text("untouched", encoding="utf-8")
    os.symlink(PYTHON, original / "bin" / "python")
    home = tmp_path / "base" / "bin"
    home.mkdir(parents=True)
    os.symlink(PYTHON, home / "python3.12")
    (original / "pyvenv.cfg").write_text(f"home = {home}\ninclude-system-site-packages = false\n", encoding="utf-8")
    preferred = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; unset GOLLIE_PYTHON312; resolve_gollie_creator; printf "%s\\n" "$GOLLIE_PYTHON312"',
            "bash",
            str(SCRIPTS / "bootstrap_gollie_isolated_venv.sh"),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "SCRATCH_ROOT": str(scratch),
            "GOLLIE_ORIGINAL_VENV": str(original),
        },
    )
    assert preferred.returncode == 0
    assert preferred.stdout.strip().splitlines()[-1] == str(home / "python3.12")
    (original / "pyvenv.cfg").unlink()
    fallback = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; unset GOLLIE_PYTHON312; resolve_gollie_creator; printf "%s\\n" "$GOLLIE_PYTHON312"',
            "bash",
            str(SCRIPTS / "bootstrap_gollie_isolated_venv.sh"),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "SCRATCH_ROOT": str(scratch),
            "GOLLIE_ORIGINAL_VENV": str(original),
        },
    )
    assert fallback.returncode == 0
    assert fallback.stdout.strip().splitlines()[-1] == str(original / "bin" / "python")
    assert marker.read_text(encoding="utf-8") == "untouched"


def test_go_resolves_an_absent_interpreter_before_requiring_it(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    original = scratch / "venvs" / "gollie-l40s"
    (original / "bin").mkdir(parents=True)
    marker = original / "MARKER"
    marker.write_text("untouched", encoding="utf-8")
    os.symlink(PYTHON, original / "bin" / "python")
    home = tmp_path / "base" / "bin"
    home.mkdir(parents=True)
    os.symlink(PYTHON, home / "python3.12")
    (original / "pyvenv.cfg").write_text(
        f"home = {home}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    wheelhouse = scratch / "recovery" / "gollie-wheelhouse-exp49"
    wheelhouse.mkdir(parents=True)
    env = os.environ.copy()
    env.pop("GOLLIE_PYTHON312", None)
    env.pop("GOLLIE_DOWNLOAD_AUTHORIZED", None)
    env.update(
        {
            "SCRATCH_ROOT": str(scratch),
            "SYSTEM_ROOT": str(ROOT),
            "GOLLIE_WHEELHOUSE": str(wheelhouse),
            "GOLLIE_ISOLATED_VENV": str(scratch / "venvs" / "gollie-isolated"),
            "GOLLIE_ORIGINAL_VENV": str(original),
            "GOLLIE_FLASH_ATTN_WHEEL": str(scratch / "flash.whl"),
        }
    )
    preferred = subprocess.run(
        ["bash", str(GO), "prepare"],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert preferred.returncode == 1
    assert "wheel download is not authorized" in preferred.stderr
    assert "GOLLIE_PYTHON312 are required" not in preferred.stderr
    assert f"CREATOR {home / 'python3.12'}" in preferred.stdout
    assert not (wheelhouse / "wheelhouse-close.json").exists()
    (original / "pyvenv.cfg").unlink()
    fallback = subprocess.run(
        ["bash", str(GO), "prepare"],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert fallback.returncode == 1
    assert "wheel download is not authorized" in fallback.stderr
    assert "GOLLIE_PYTHON312 are required" not in fallback.stderr
    assert f"CREATOR {original / 'bin' / 'python'}" in fallback.stdout
    assert marker.read_text(encoding="utf-8") == "untouched"
    assert not (scratch / "venvs" / "gollie-isolated").exists()
