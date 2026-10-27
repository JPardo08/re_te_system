"""Read-only selection of GoLLIE wheels from a pip HTTP cache.

Internal wheel METADATA is a claim made by the archive. It is not an external
provenance accreditation. SHA-256 values in the download plan were read from
the PyPI JSON of those exact versions on 2026-10-27 and still have to be
recomputed on the downloaded bytes before a wheel may enter the wheelhouse.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import (
    InvalidTag,
    Tag,
    TooManyTagsError,
    UnsortedTagsError,
    compatible_tags,
    cpython_tags,
    parse_tag,
)
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from flash_attn_pin import (
    EXPECTED_BLACK,
    EXPECTED_EINOPS,
    EXPECTED_HUGGINGFACE_HUB,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    EXPECTED_TRITON,
)


ZIP_MAGIC = b"PK\x03\x04"
DOWNLOAD_AUTHORIZED = False
MATERIALIZATION_AUTHORIZED = False
# CPython 3.12 on Linux x86_64. manylinux tags stop at glibc 2.28.
TARGET_PYTHON = (3, 12)
TARGET_ARCH = "x86_64"
TARGET_GLIBC = (2, 28)

# Requires-Dist read from the official file
# torch-2.5.1+cu121-cp312-cp312-linux_x86_64.whl. The PyPI project metadata
# for torch 2.5.1 declares the CUDA 12.4 pins instead and is not this list.
TORCH_CU121_REQUIRES = (
    "filelock",
    "typing-extensions (>=4.8.0)",
    "networkx",
    "jinja2",
    "fsspec",
    "nvidia-cuda-nvrtc-cu12 (==12.1.105) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-cuda-runtime-cu12 (==12.1.105) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-cuda-cupti-cu12 (==12.1.105) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-cudnn-cu12 (==9.1.0.70) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-cublas-cu12 (==12.1.3.1) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-cufft-cu12 (==11.0.2.54) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-curand-cu12 (==10.3.2.106) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-cusolver-cu12 (==11.4.5.107) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-cusparse-cu12 (==12.1.0.106) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-nccl-cu12 (==2.21.5) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "nvidia-nvtx-cu12 (==12.1.105) ; platform_system == \"Linux\" and platform_machine == \"x86_64\"",
    "triton (==3.1.0) ; platform_system == \"Linux\" and platform_machine == \"x86_64\" and python_version < \"3.13\"",
    "sympy (==1.12.1) ; python_version == \"3.8\"",
    "setuptools ; python_version >= \"3.12\"",
    "sympy (==1.13.1) ; python_version >= \"3.9\"",
)
TRANSFORMERS_REQUIRES = (
    "filelock",
    "huggingface-hub (<1.0,>=0.34.0)",
    "numpy (>=1.17)",
    "packaging (>=20.0)",
    "pyyaml (>=5.1)",
    "regex (!=2019.12.17)",
    "requests",
    "tokenizers (<=0.23.0,>=0.22.0)",
    "safetensors (>=0.4.3)",
    "tqdm (>=4.27)",
)
BLACK_REQUIRES = (
    "click (>=8.0.0)",
    "mypy-extensions (>=0.4.3)",
    "packaging (>=22.0)",
    "pathspec (>=0.9.0)",
    "platformdirs (>=2)",
    "tomli (>=1.1.0) ; python_version < \"3.11\"",
    "typing-extensions (>=4.0.1) ; python_version < \"3.11\"",
)
TRITON_REQUIRES = ("filelock",)
EINOPS_REQUIRES = ()
HUGGINGFACE_HUB_REQUIRES = (
    "filelock",
    "fsspec (>=2023.5.0)",
    "hf-xet (<2.0.0,>=1.1.3) ; platform_machine == \"x86_64\" or platform_machine == \"amd64\" or platform_machine == \"arm64\" or platform_machine == \"aarch64\"",
    "packaging (>=20.9)",
    "pyyaml (>=5.1)",
    "requests",
    "tqdm (>=4.42.1)",
    "typing-extensions (>=3.7.4.3)",
)
CUDNN_REQUIRES = ("nvidia-cublas-cu12",)
CUSOLVER_REQUIRES = ("nvidia-cublas-cu12", "nvidia-nvjitlink-cu12", "nvidia-cusparse-cu12")
CUSPARSE_REQUIRES = ("nvidia-nvjitlink-cu12",)
SYMPY_REQUIRES = ("mpmath (<1.4,>=1.1.0)",)

STACK_EXACT = {
    "torch": EXPECTED_TORCH,
    "transformers": EXPECTED_TRANSFORMERS,
    "black": EXPECTED_BLACK,
    "triton": EXPECTED_TRITON,
    "einops": EXPECTED_EINOPS,
    "huggingface-hub": EXPECTED_HUGGINGFACE_HUB,
}
CACHE_EXACT_AVAILABLE = {
    **STACK_EXACT,
    "nvidia-cudnn-cu12": "9.1.0.70",
}
# Version strings named by the CESVIMA inventory of pip-cache/http-v2.
# The inventory did not provide a path, tag, file count, or sha256.
CESVIMA_NAMED_CACHE_VERSIONS = {
    "sympy": ("1.13.1", "1.14.0"),
}
# Counts and hashes reported by the CESVIMA select-cache run. This machine
# did not re-read that cache. The other six selected names were not reported.
CESVIMA_SELECT_CACHE = {
    "ambiguous": 0,
    "exit_code": 0,
    "installable_wheelhouse": False,
    "manifest_sha256": "e73bc2dc212fb31324eb0b07c4f1c9e0dfeb0f078d783dd4fafcb0ba23883fe0",
    "materialized": False,
    "provenance_accredited": False,
    "records": (
        {
            "measured_sha256": "db36cdc64bf61b9b24578b6f7bab1ecdd2452cf008f34faa33776680c26d66f8",
            "name": "sympy",
            "outcome": "SELECTED_CANDIDATE",
            "provenance_accredited": False,
            "version": "1.13.1",
        },
        {
            "measured_sha256": "165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f",
            "name": "nvidia-cudnn-cu12",
            "outcome": "REJECTED",
            "provenance_accredited": False,
            "tag": "py3-none-manylinux2014_x86_64",
            "version": "9.1.0.70",
        },
        {"name": "sympy", "outcome": "REJECTED", "version": "1.14.0"},
        {"name": "torch", "outcome": "REJECTED", "version": "2.14"},
        {"name": "triton", "outcome": "REJECTED", "version": "3.8"},
    ),
    "rejected": 21,
    "selected": 7,
    "unpinned_observed": 77,
}

# Official linux artifacts reported by PyPI JSON on 2026-10-27. The hash is
# the index digest, not a hash recomputed from a local file.
OFFICIAL_LINUX_WHEELS = (
    {
        "name": "nvidia-cuda-nvrtc-cu12",
        "version": "12.1.105",
        "filename": "nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
        "sha256": "339b385f50c309763ca65456ec75e17bbefcbbf2893f462cb8b90584cd27a1c2",
        "url": "https://files.pythonhosted.org/packages/b6/9f/c64c03f49d6fbc56196664d05dba14e3a561038a81a638eeb47f4d4cfd48/nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
    },
    {
        "name": "nvidia-cuda-runtime-cu12",
        "version": "12.1.105",
        "filename": "nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
        "sha256": "6e258468ddf5796e25f1dc591a31029fa317d97a0a94ed93468fc86301d61e40",
        "url": "https://files.pythonhosted.org/packages/eb/d5/c68b1d2cdfcc59e72e8a5949a37ddb22ae6cade80cd4a57a84d4c8b55472/nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
    },
    {
        "name": "nvidia-cuda-cupti-cu12",
        "version": "12.1.105",
        "filename": "nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
        "sha256": "e54fde3983165c624cb79254ae9818a456eb6e87a7fd4d56a2352c24ee542d7e",
        "url": "https://files.pythonhosted.org/packages/7e/00/6b218edd739ecfc60524e585ba8e6b00554dd908de2c9c66c1af3e44e18d/nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
    },
    {
        "name": "nvidia-cublas-cu12",
        "version": "12.1.3.1",
        "filename": "nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl",
        "sha256": "ee53ccca76a6fc08fb9701aa95b6ceb242cdaab118c3bb152af4e579af792728",
        "url": "https://files.pythonhosted.org/packages/37/6d/121efd7382d5b0284239f4ab1fc1590d86d34ed4a4a2fdb13b30ca8e5740/nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl",
    },
    {
        "name": "nvidia-cufft-cu12",
        "version": "11.0.2.54",
        "filename": "nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl",
        "sha256": "794e3948a1aa71fd817c3775866943936774d1c14e7628c74f6f7417224cdf56",
        "url": "https://files.pythonhosted.org/packages/86/94/eb540db023ce1d162e7bea9f8f5aa781d57c65aed513c33ee9a5123ead4d/nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl",
    },
    {
        "name": "nvidia-curand-cu12",
        "version": "10.3.2.106",
        "filename": "nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl",
        "sha256": "9d264c5036dde4e64f1de8c50ae753237c12e0b1348738169cd0f8a536c0e1e0",
        "url": "https://files.pythonhosted.org/packages/44/31/4890b1c9abc496303412947fc7dcea3d14861720642b49e8ceed89636705/nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl",
    },
    {
        "name": "nvidia-cusolver-cu12",
        "version": "11.4.5.107",
        "filename": "nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl",
        "sha256": "8a7ec542f0412294b15072fa7dab71d31334014a69f953004ea7a118206fe0dd",
        "url": "https://files.pythonhosted.org/packages/bc/1d/8de1e5c67099015c834315e333911273a8c6aaba78923dd1d1e25fc5f217/nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl",
    },
    {
        "name": "nvidia-cusparse-cu12",
        "version": "12.1.0.106",
        "filename": "nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl",
        "sha256": "f3b50f42cf363f86ab21f720998517a659a48131e8d538dc02f8768237bd884c",
        "url": "https://files.pythonhosted.org/packages/65/5b/cfaeebf25cd9fdec14338ccb16f6b2c4c7fa9163aefcf057d86b9cc248bb/nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl",
    },
    {
        "name": "nvidia-nccl-cu12",
        "version": "2.21.5",
        "filename": "nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl",
        "sha256": "8579076d30a8c24988834445f8d633c697d42397e92ffc3f63fa26766d25e0a0",
        "url": "https://files.pythonhosted.org/packages/df/99/12cd266d6233f47d00daf3a72739872bdc10267d0383508b0b9c84a18bb6/nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl",
    },
    {
        "name": "nvidia-nvtx-cu12",
        "version": "12.1.105",
        "filename": "nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
        "sha256": "dc21cf308ca5691e7c04d962e213f8a4aa9bbfa23d95412f452254c2caeb09e5",
        "url": "https://files.pythonhosted.org/packages/da/d3/8057f0587683ed2fcd4dbfbdfdfa807b9160b809976099d36b8f60d08f03/nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl",
    },
    {
        "name": "sympy",
        "version": "1.13.1",
        "filename": "sympy-1.13.1-py3-none-any.whl",
        "sha256": "db36cdc64bf61b9b24578b6f7bab1ecdd2452cf008f34faa33776680c26d66f8",
        "url": "https://files.pythonhosted.org/packages/b2/fe/81695a1aa331a842b582453b605175f419fe8540355886031328089d840a/sympy-1.13.1-py3-none-any.whl",
    },
)


def linux_py312_environment() -> dict[str, str]:
    environment = default_environment()
    environment.update(
        {
            "extra": "",
            "implementation_name": "cpython",
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_system": "Linux",
            "platform_python_implementation": "CPython",
            "python_full_version": "3.12.3",
            "python_version": "3.12",
            "sys_platform": "linux",
        }
    )
    return environment


def _metadata_text(name: str, version: str, requires: tuple[str, ...]) -> str:
    lines = [f"Name: {name}", f"Version: {version}"]
    lines.extend(f"Requires-Dist: {item}" for item in requires)
    return "\n".join(lines) + "\n"


def closure_metadata() -> dict[str, str]:
    """Metadata used to close the cu121 stack. Extras are absent."""
    pairs = {
        "torch": (EXPECTED_TORCH, TORCH_CU121_REQUIRES),
        "transformers": (EXPECTED_TRANSFORMERS, TRANSFORMERS_REQUIRES),
        "black": (EXPECTED_BLACK, BLACK_REQUIRES),
        "triton": (EXPECTED_TRITON, TRITON_REQUIRES),
        "einops": (EXPECTED_EINOPS, EINOPS_REQUIRES),
        "huggingface-hub": (EXPECTED_HUGGINGFACE_HUB, HUGGINGFACE_HUB_REQUIRES),
        "nvidia-cudnn-cu12": ("9.1.0.70", CUDNN_REQUIRES),
        "nvidia-cusolver-cu12": ("11.4.5.107", CUSOLVER_REQUIRES),
        "nvidia-cusparse-cu12": ("12.1.0.106", CUSPARSE_REQUIRES),
        "sympy": ("1.13.1", SYMPY_REQUIRES),
    }
    return {name: _metadata_text(name, version, requires) for name, (version, requires) in pairs.items()}


def active_requirements(metadata_text: str) -> list[Requirement]:
    environment = linux_py312_environment()
    found = []
    for line in metadata_text.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        requirement = Requirement(line.split(":", 1)[1].strip())
        if requirement.marker is None or requirement.marker.evaluate(environment):
            found.append(requirement)
    return found


def _exact_version(requirement: Requirement) -> str | None:
    specs = list(requirement.specifier)
    if len(specs) == 1 and specs[0].operator == "==":
        return specs[0].version
    return None


def build_closure(metadata_by_name: dict[str, str]) -> dict[str, object]:
    exact = {canonicalize_name(name): version for name, version in STACK_EXACT.items()}
    unpinned: dict[str, str] = {}
    for metadata_text in metadata_by_name.values():
        for requirement in active_requirements(metadata_text):
            name = canonicalize_name(requirement.name)
            version = _exact_version(requirement)
            if version is not None:
                previous = exact.get(name)
                if previous is not None and previous != version:
                    raise ValueError(f"exact version conflict for {name}: {previous} != {version}")
                exact[name] = version
                unpinned.pop(name, None)
            elif name not in exact:
                combined = requirement.specifier
                if name in unpinned:
                    combined = SpecifierSet(unpinned[name]) & combined
                unpinned[name] = str(combined)
    available = {canonicalize_name(name): version for name, version in CACHE_EXACT_AVAILABLE.items()}
    named = {
        canonicalize_name(name): versions
        for name, versions in CESVIMA_NAMED_CACHE_VERSIONS.items()
    }
    missing = []
    findings = []
    for name, version in sorted(exact.items()):
        reported = named.get(name, ())
        if available.get(name) == version:
            continue
        if version in reported:
            findings.append(
                {
                    "accreditation": "NOT_ACCREDITED",
                    "evidence": "static_version_list",
                    "name": name,
                    "reason": (
                        "Earlier CESVIMA inventory of pip-cache/http-v2 named this version. "
                        "This entry records that version. It is not a select-cache candidate "
                        "and it does not carry the hash measured later."
                    ),
                    "selected_candidate": False,
                    "status": "VERSION_OBSERVED",
                    "version": version,
                }
            )
            continue
        missing.append(
            {
                "evidence": "static_version_list",
                "name": name,
                "selected_candidate": False,
                "status": "MISSING",
                "version": version,
            }
        )
    for name, versions in sorted(named.items()):
        required = exact.get(name)
        for version in versions:
            if version == required:
                continue
            findings.append(
                {
                    "accreditation": "NOT_ACCREDITED",
                    "name": name,
                    "reason": (
                        "The same pip-cache/http-v2 inventory named this version. "
                        f"The cu121 closure requires {required}."
                    ),
                    "status": "INCOMPATIBLE",
                    "version": version,
                }
            )
    return {
        "available_exact": [
            {
                "evidence": "static_version_list",
                "name": name,
                "selected_candidate": False,
                "status": "VERSION_OBSERVED",
                "version": version,
            }
            for name, version in sorted(available.items())
        ],
        "cache_findings": findings,
        "cesvima_select_cache": CESVIMA_SELECT_CACHE,
        "evidence": "static_version_list",
        "exact_missing": missing,
        "exact_required": [{"name": name, "version": version} for name, version in sorted(exact.items())],
        "installable_wheelhouse": False,
        "materialized": False,
        "provenance_accredited": False,
        "selection_executed": False,
        "source": "torch 2.5.1+cu121 wheel METADATA plus PyPI requires_dist of the named stack; not the CUDA 12.4 pins of PyPI torch 2.5.1",
        "unpinned": [{"name": name, "specifier": spec} for name, spec in sorted(unpinned.items())],
    }


def prepared_closure() -> dict[str, object]:
    return build_closure(closure_metadata())


def _allowed_exact() -> dict[str, str]:
    closure = prepared_closure()
    return {str(item["name"]): str(item["version"]) for item in closure["exact_required"]}


def _rejection(name: str, version: str) -> str | None:
    canonical = canonicalize_name(name)
    lowered = f"{canonical} {version}".lower()
    if "cu13" in lowered or "cuda13" in lowered:
        return "cuda 13 is outside the cu121 closure"
    if canonical == "torch" and (version == "2.14" or version.startswith("2.14.")):
        return "torch 2.14 is outside the cu121 closure"
    if canonical == "triton" and (version == "3.8" or version.startswith("3.8.")):
        return "triton 3.8 is outside the cu121 closure"
    allowed = _allowed_exact()
    if canonical in {"torch", "triton"} and allowed.get(canonical) != version:
        return "version is outside the pinned stack"
    if canonical.startswith("nvidia-") and allowed.get(canonical) != version:
        return "nvidia version is outside the cu121 closure"
    if canonical in allowed and allowed[canonical] != version:
        return "version is outside the cu121 closure"
    if canonical not in allowed:
        return "distribution has no exact pin"
    return None


def inspect_wheel_bytes(payload: bytes, origin: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "declared_name": None,
        "declared_version": None,
        "origin_path": str(origin),
        "provenance_accredited": False,
        "requires_dist": [],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "tags": [],
        "zip_ok": False,
    }
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            broken = archive.testzip()
            if broken is not None:
                record["rejection"] = f"zip integrity failed: {broken}"
                return record
            metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            wheel_names = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                record["rejection"] = "wheel metadata is ambiguous"
                return record
            metadata = archive.read(metadata_names[0]).decode()
            wheel_text = archive.read(wheel_names[0]).decode()
    except zipfile.BadZipFile:
        record["rejection"] = "not a zip wheel"
        return record
    record["zip_ok"] = True
    fields: dict[str, str] = {}
    requires = []
    for line in metadata.splitlines():
        if line.startswith("Requires-Dist:"):
            requires.append(line.split(":", 1)[1].strip())
        elif ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    record["declared_name"] = fields.get("name")
    record["declared_version"] = fields.get("version")
    record["requires_dist"] = requires
    record["tags"] = [
        line.split(":", 1)[1].strip()
        for line in wheel_text.splitlines()
        if line.startswith("Tag:")
    ]
    if not record["declared_name"] or not record["declared_version"]:
        record["rejection"] = "METADATA has no name or version"
        return record
    reason = _rejection(str(record["declared_name"]), str(record["declared_version"]))
    if reason is not None:
        record["rejection"] = reason
        record["status"] = "INCOMPATIBLE"
        return record
    if not _tags_are_appropriate(list(record["tags"])):
        record["rejection"] = "wheel tags are not appropriate for the cu121 cp312 runtime"
        record["status"] = "INCOMPATIBLE"
    return record


def _glibc_platforms(glibc: tuple[int, int], arch: str) -> tuple[str, ...]:
    """manylinux tags for one glibc ceiling. Does not read the host platform.

    The descent matches packaging's manylinux generator: current glibc down to
    2.5 on x86_64, with the legacy manylinux2014, manylinux2010 and manylinux1
    aliases. A tag that requires a newer glibc is absent.
    """
    legacy = {(2, 17): "manylinux2014", (2, 12): "manylinux2010", (2, 5): "manylinux1"}
    oldest_minor = 4 if arch in {"x86_64", "i686"} else 16
    platforms = [f"linux_{arch}"]
    for minor in range(glibc[1], oldest_minor, -1):
        platforms.append(f"manylinux_{glibc[0]}_{minor}_{arch}")
        alias = legacy.get((glibc[0], minor))
        if alias is not None:
            platforms.append(f"{alias}_{arch}")
    return tuple(platforms)


@functools.lru_cache(maxsize=1)
def target_tag_set() -> frozenset[Tag]:
    """Tags installable by CPython 3.12 on Linux x86_64 with glibc 2.28."""
    platforms = _glibc_platforms(TARGET_GLIBC, TARGET_ARCH)
    found = set(cpython_tags(python_version=TARGET_PYTHON, abis=["cp312"], platforms=platforms))
    found.update(
        compatible_tags(python_version=TARGET_PYTHON, interpreter="cp312", platforms=platforms)
    )
    return frozenset(found)


def _tags_are_appropriate(tags: list[str]) -> bool:
    """True when one wheel tag is in the simulated cp312 linux x86_64 set."""
    if not tags:
        return False
    accepted = target_tag_set()
    matched = False
    for raw in tags:
        try:
            parsed = parse_tag(raw)
        except (InvalidTag, TooManyTagsError, UnsortedTagsError, ValueError):
            return False
        if parsed & accepted:
            matched = True
    return matched


def iter_cache_bodies(cache: Path) -> list[Path]:
    """Regular files only. Symlinks and the cache itself are not modified."""
    bodies = []
    for directory, dirnames, filenames in os.walk(cache, followlinks=False):
        dirnames[:] = [name for name in dirnames if not (Path(directory) / name).is_symlink()]
        for name in filenames:
            path = Path(directory) / name
            if not path.is_file() or path.is_symlink():
                continue
            if name.endswith(".body") or name.endswith(".whl"):
                bodies.append(path)
                continue
            with path.open("rb") as handle:
                if handle.read(4) == ZIP_MAGIC:
                    bodies.append(path)
    return bodies


def scan_cache(cache: Path) -> list[dict[str, object]]:
    records = []
    for path in iter_cache_bodies(cache):
        payload = path.read_bytes()
        if not payload.startswith(ZIP_MAGIC):
            continue
        records.append(inspect_wheel_bytes(payload, path.resolve()))
    return records


def select_records(records: list[dict[str, object]]) -> dict[str, object]:
    selected: dict[tuple[str, str], dict[str, object]] = {}
    rejected = []
    unpinned = []
    ambiguous = []
    for record in records:
        if record.get("rejection") and not record.get("zip_ok"):
            rejected.append(record)
            continue
        name = record.get("declared_name")
        version = record.get("declared_version")
        if not isinstance(name, str) or not isinstance(version, str):
            rejected.append(record)
            continue
        if record.get("rejection"):
            if record["rejection"] == "distribution has no exact pin":
                unpinned.append(record)
            else:
                record["status"] = "INCOMPATIBLE"
                rejected.append(record)
            continue
        key = (canonicalize_name(name), version)
        if key in selected:
            record["status"] = "AMBIGUOUS"
            selected[key]["status"] = "AMBIGUOUS"
            ambiguous.append(record)
            ambiguous.append(selected.pop(key))
            continue
        record["status"] = "SELECTED_CANDIDATE"
        record["accreditation"] = "NOT_ACCREDITED"
        record["filename_provenance_accredited"] = False
        record["hash_evidence"] = "measured"
        record["selected_candidate"] = True
        selected[key] = record
    return {
        "ambiguous": ambiguous,
        "provenance_accredited": False,
        "rejected": rejected,
        "selected": list(selected.values()),
        "unpinned_observed": unpinned,
    }


def manifest_path_is_outside_cache(cache: Path, manifest: Path) -> bool:
    cache_root = cache.resolve()
    target = manifest.resolve()
    return target != cache_root and cache_root not in target.parents


def _distribution_file_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def candidate_wheel_filename(name: str, version: str, tags: list[str]) -> str:
    """PEP 427 filename. A cache hash is not a wheel name."""
    if not tags:
        raise ValueError("wheel tags are required to name a wheel")
    parsed: set[Tag] = set()
    for tag in tags:
        parsed |= parse_tag(tag)
    if len(tags) == 1:
        tag_component = tags[0]
    else:
        interpreters = ".".join(sorted({item.interpreter for item in parsed}))
        abis = ".".join(sorted({item.abi for item in parsed}))
        platforms = ".".join(sorted({item.platform for item in parsed}))
        tag_component = f"{interpreters}-{abis}-{platforms}"
    filename = f"{_distribution_file_name(name)}-{version.replace('-', '_')}-{tag_component}.whl"
    assert_wheel_filename_matches(filename, name, version, tags)
    return filename


def assert_wheel_filename_matches(filename: str, name: str, version: str, tags: list[str]) -> str:
    """Reject a truncated name such as torch-2.5.1+cu121.whl."""
    parsed_name, parsed_version, _build, parsed_tags = parse_wheel_filename(filename)
    expected: set[Tag] = set()
    for tag in tags:
        expected |= parse_tag(tag)
    if parsed_name != canonicalize_name(name):
        raise ValueError("wheel filename does not match METADATA name")
    if parsed_version != Version(version):
        raise ValueError("wheel filename does not match METADATA version")
    if parsed_tags != expected:
        raise ValueError("wheel filename does not match WHEEL tags")
    return filename


def stage_selected(cache: Path, records: list[dict[str, object]], destination: Path) -> list[Path]:
    """Validate a future copy, then refuse to write it.

    select-cache does not call this. No wheel is copied in this revision.
    """
    if not manifest_path_is_outside_cache(cache, destination):
        raise ValueError("refusing to write the wheelhouse inside the pip cache")
    for record in records:
        if record.get("filename_provenance_accredited") is not True:
            raise ValueError("original wheel filename is not accredited")
        raw_tags = record["tags"]
        if not isinstance(raw_tags, list) or not raw_tags:
            raise ValueError("wheel tags are required to name a wheel")
        tags = [str(tag) for tag in raw_tags]
        filename = assert_wheel_filename_matches(
            str(record.get("original_filename", "")),
            str(record["declared_name"]),
            str(record["declared_version"]),
            tags,
        )
        if filename != candidate_wheel_filename(
            str(record["declared_name"]),
            str(record["declared_version"]),
            tags,
        ):
            raise ValueError("wheel filename does not match the metadata and tags")
        origin = Path(str(record["origin_path"]))
        before = origin.stat().st_mtime_ns
        payload = origin.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"sha256 does not match {origin}")
        if origin.stat().st_mtime_ns != before:
            raise ValueError(f"cache file was modified: {origin}")
    if MATERIALIZATION_AUTHORIZED is not True:
        raise RuntimeError("wheel materialization is not authorized")
    raise RuntimeError("wheel materialization is not authorized")


def download_plan() -> dict[str, object]:
    closure = prepared_closure()
    missing = {(item["name"], item["version"]) for item in closure["exact_missing"]}
    planned = []
    for item in OFFICIAL_LINUX_WHEELS:
        key = (canonicalize_name(item["name"]), item["version"])
        if key not in missing:
            continue
        if "cu13" in item["filename"] or "2.14" in item["version"] or item["version"].startswith("3.8"):
            raise ValueError(f"refusing incompatible official artifact {item['filename']}")
        host = item["url"].split("/")[2]
        if host != "files.pythonhosted.org":
            raise ValueError(f"refusing unofficial host {host}")
        planned.append({**item, "download_authorized": False, "provenance": "pypi-json-2026-10-27"})
    unpinned = [item["name"] for item in closure["unpinned"]]
    present = [item for item in closure["cache_findings"] if item["status"] == "VERSION_OBSERVED"]
    return {
        "download_authorized": False,
        "exact_missing": planned,
        "omitted_unpinned": unpinned,
        "present_not_accredited": present,
        "provenance_accredited": False,
    }


def fetch_planned_wheels() -> None:
    if DOWNLOAD_AUTHORIZED is not True:
        raise RuntimeError("selective wheel download is not authorized")
    raise RuntimeError("download execution is not implemented")


def _emit(payload: dict[str, object]) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["missing"]:
        _emit(prepared_closure())
        return 0
    if args == ["download-plan"]:
        _emit(download_plan())
        return 0
    if args == ["download"]:
        fetch_planned_wheels()
        return 1
    if args == ["select-cache"]:
        cache = Path(os.environ["GOLLIE_PIP_HTTP_CACHE"])
        manifest = Path(os.environ["GOLLIE_WHEEL_MANIFEST"])
        if not manifest_path_is_outside_cache(cache, manifest):
            print("FAIL: refusing to write the manifest inside the pip cache", file=sys.stderr)
            return 1
        selection = select_records(scan_cache(cache))
        selection["cache"] = str(cache.resolve())
        selection["installable_wheelhouse"] = False
        selection["materialized"] = False
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"WHEEL_MANIFEST {manifest}")
        return 0
    print(
        "usage: gollie_wheelhouse.py [missing|download-plan|download|select-cache]",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
