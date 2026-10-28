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
    assert plan["unresolved_transitive"] == []
    observed = {item["name"]: item for item in wheelhouse.OBSERVED_TRANSITIVE_WHEELS}
    assert len(observed) == 24
    assert observed["requests"]["sha256"] == "2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0"
    assert observed["jinja2"]["sha256"] == "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67"
    assert observed["markupsafe"]["version"] == "3.0.3"
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
    assert '"offline_resolve_pass": True' not in script
    assert 'os.environ["DRY_STATUS"] == "0"' in script
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


def _tagged_wheel(
    path: Path,
    name: str,
    version: str,
    tags: list[str],
    *,
    vendor: bool = False,
    second_root: bool = False,
) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{info}/METADATA", f"Name: {name}\nVersion: {version}\n")
        archive.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\n"
            + "".join(f"Tag: {tag}\n" for tag in tags),
        )
        if vendor:
            archive.writestr(
                f"{name}/_vendor/packaging-24.2.dist-info/METADATA",
                "Name: packaging\nVersion: 24.2\n",
            )
            archive.writestr(
                f"{name}/_vendor/packaging-24.2.dist-info/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
        if second_root:
            archive.writestr("other-1.0.0.dist-info/METADATA", "Name: other\nVersion: 1.0.0\n")
            archive.writestr(
                "other-1.0.0.dist-info/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
    return path.read_bytes()


def _reproduction(name: str) -> dict[str, object]:
    return next(item for item in wheelhouse.REPRODUCTION_WHEELS if item["name"] == name)


def _with_payload(record: dict[str, object], payload: bytes) -> dict[str, object]:
    return {**record, "download": True, "sha256": hashlib.sha256(payload).hexdigest()}


NV_TAGS = ["py3-none-manylinux2010_x86_64", "py3-none-manylinux_2_12_x86_64"]


def test_recorded_reproduction_wheels_match_name_version_and_tags(tmp_path: Path) -> None:
    setuptools = _reproduction("setuptools")
    nvjitlink = _reproduction("nvidia-nvjitlink-cu12")
    setup_payload = _tagged_wheel(
        tmp_path / str(setuptools["filename"]),
        "setuptools",
        "78.1.0",
        ["py3-none-any"],
        vendor=True,
    )
    nv_payload = _tagged_wheel(
        tmp_path / str(nvjitlink["filename"]),
        "nvidia-nvjitlink-cu12",
        "12.9.86",
        NV_TAGS,
    )
    assert wheelhouse._rejection("setuptools", "78.1.0") == "distribution has no exact pin"
    assert wheelhouse._rejection("nvidia-nvjitlink-cu12", "12.9.86") is None
    for record, payload in ((setuptools, setup_payload), (nvjitlink, nv_payload)):
        accepted = wheelhouse._matching_record(payload, [_with_payload(record, payload)])
        assert accepted is not None
        assert accepted["filename"] == record["filename"]
        assert accepted["version"] == record["version"]
    inspected = wheelhouse.inspect_wheel_bytes(setup_payload, tmp_path / str(setuptools["filename"]))
    assert inspected["zip_ok"] is True
    assert inspected["declared_name"] == "setuptools"
    assert inspected["declared_version"] == "78.1.0"
    assert inspected["tags"] == ["py3-none-any"]


def test_reproduction_rules_reject_other_versions_hashes_and_tags(tmp_path: Path) -> None:
    setuptools = _reproduction("setuptools")
    nvjitlink = _reproduction("nvidia-nvjitlink-cu12")
    assert wheelhouse._rejection("nvidia-nvjitlink-cu12", "12.4.99") == (
        "nvidia version is outside the cu121 closure"
    )
    assert wheelhouse._rejection("nvidia-cublas-cu12", "12.9.86") == (
        "nvidia version is outside the cu121 closure"
    )
    assert wheelhouse._rejection("setuptools", "70.0.0") == "distribution has no exact pin"
    other = _tagged_wheel(
        tmp_path / "nvidia_nvjitlink_cu12-12.4.99-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        "nvidia-nvjitlink-cu12",
        "12.4.99",
        NV_TAGS,
    )
    other_record = {
        **nvjitlink,
        "filename": "nvidia_nvjitlink_cu12-12.4.99-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        "sha256": hashlib.sha256(other).hexdigest(),
        "version": "12.4.99",
    }
    assert wheelhouse._matching_record(other, [other_record]) is None
    setup_payload = _tagged_wheel(
        tmp_path / str(setuptools["filename"]),
        "setuptools",
        "78.1.0",
        ["py3-none-any"],
        vendor=True,
    )
    assert wheelhouse._matching_record(setup_payload, [setuptools]) is None
    with pytest.raises(ValueError, match="downloaded sha256 does not match"):
        wheelhouse._download_official(setuptools, lambda _url: setup_payload)
    macos = _tagged_wheel(
        tmp_path / "setuptools-78.1.0-py3-none-macosx_14_0_arm64.whl",
        "setuptools",
        "78.1.0",
        ["py3-none-macosx_14_0_arm64"],
    )
    macos_record = {
        **setuptools,
        "filename": "setuptools-78.1.0-py3-none-macosx_14_0_arm64.whl",
        "sha256": hashlib.sha256(macos).hexdigest(),
    }
    assert wheelhouse._matching_record(macos, [macos_record]) is None
    nv_macos = _tagged_wheel(
        tmp_path / "nvidia_nvjitlink_cu12-12.9.86-py3-none-macosx_14_0_arm64.whl",
        "nvidia-nvjitlink-cu12",
        "12.9.86",
        ["py3-none-macosx_14_0_arm64"],
    )
    nv_macos_record = {
        **nvjitlink,
        "filename": "nvidia_nvjitlink_cu12-12.9.86-py3-none-macosx_14_0_arm64.whl",
        "sha256": hashlib.sha256(nv_macos).hexdigest(),
    }
    assert wheelhouse._matching_record(nv_macos, [nv_macos_record]) is None
    ambiguous = _tagged_wheel(
        tmp_path / "ambiguous.whl",
        "setuptools",
        "78.1.0",
        ["py3-none-any"],
        second_root=True,
    )
    assert wheelhouse.inspect_wheel_bytes(ambiguous, tmp_path / "ambiguous.whl")["rejection"] == (
        "wheel metadata is ambiguous"
    )
    foreign = {**setuptools, "url": "https://example.invalid/setuptools-78.1.0-py3-none-any.whl"}
    with pytest.raises(ValueError, match="unofficial host"):
        wheelhouse._download_official(foreign, lambda _url: setup_payload)


def test_close_resumes_without_replacing_forty_two_present_wheels(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    destination = scratch / "recovery" / "gollie-wheelhouse-exp49"
    destination.mkdir(parents=True)
    catalog: list[dict[str, object]] = []
    present: list[tuple[Path, str, int]] = []
    for index in range(42):
        name = f"kept{index:02d}"
        filename = f"{name}-1.0.0-py3-none-any.whl"
        path = destination / filename
        payload = _tagged_wheel(path, name, "1.0.0", ["py3-none-any"])
        catalog.append(
            {
                "download": True,
                "filename": filename,
                "name": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "url": f"https://files.pythonhosted.org/packages/aa/{filename}",
                "version": "1.0.0",
            }
        )
        present.append((path, hashlib.sha256(payload).hexdigest(), path.stat().st_mtime_ns))
    setuptools = _reproduction("setuptools")
    nvjitlink = _reproduction("nvidia-nvjitlink-cu12")
    setup_payload = _tagged_wheel(
        tmp_path / str(setuptools["filename"]),
        "setuptools",
        "78.1.0",
        ["py3-none-any"],
        vendor=True,
    )
    nv_payload = _tagged_wheel(
        tmp_path / str(nvjitlink["filename"]),
        "nvidia-nvjitlink-cu12",
        "12.9.86",
        NV_TAGS,
    )
    catalog.extend((_with_payload(setuptools, setup_payload), _with_payload(nvjitlink, nv_payload)))
    calls: list[str] = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        if url == nvjitlink["url"]:
            return nv_payload
        if url == setuptools["url"]:
            return setup_payload
        raise AssertionError(url)

    report = wheelhouse.close_wheelhouse(
        scratch / "missing-cache",
        destination,
        download_authorized=True,
        scratch_root=scratch,
        catalog=tuple(catalog),
        fetcher=fetcher,
    )
    assert report["blocked"] == []
    assert report["closure_complete"] is True
    assert calls == [nvjitlink["url"], setuptools["url"]]
    for path, digest, modified in present:
        assert path.stat().st_mtime_ns == modified
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    calls.clear()
    again = wheelhouse.close_wheelhouse(
        scratch / "missing-cache",
        destination,
        download_authorized=True,
        scratch_root=scratch,
        catalog=tuple(catalog),
        fetcher=fetcher,
    )
    assert calls == []
    assert again["blocked"] == []
    assert again["closure_complete"] is True
    assert {item["status"] for item in again["results"]} == {"present"}
    for path, digest, modified in present:
        assert path.stat().st_mtime_ns == modified
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
