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
import shutil
import sys
import urllib.request
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
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
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
# Later CESVIMA select-cache, after the manylinux2014 tag correction.
# Per-file hashes of this manifest were not copied into the repository.
EXP472_MANIFEST_SHA256 = "7308b6ae7c4aa7df5a324b5e057c9c8f655ee958568cc8716e208ca3ebab8357"
EXP472_MANIFEST_RELATIVE = Path("recovery") / "gollie-wheel-manifest-exp472-20261027.json"
EXP472_SELECTED = 8
EXP472_REJECTED = 20
EXP472_AMBIGUOUS = 0

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

# Digests read from package indexes on 2026-10-27. No wheel body was downloaded.
# A cache file is accredited only when its recomputed SHA-256 equals one of these.
PUBLISHED_CACHE_WHEELS = (
    {
        "filename": "torch-2.5.1+cu121-cp312-cp312-linux_x86_64.whl",
        "name": "torch",
        "provenance": "pytorch-cu121-simple-index-2026-10-27",
        "sha256": "222be02548c2e74a21a8fbc8e5b8d2eef9f9faee865d70385d2eb1b9aabcbc76",
        "url": "https://download-r2.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp312-cp312-linux_x86_64.whl#sha256=222be02548c2e74a21a8fbc8e5b8d2eef9f9faee865d70385d2eb1b9aabcbc76",
        "version": "2.5.1+cu121",
    },
    {
        "filename": "nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl",
        "name": "nvidia-cudnn-cu12",
        "provenance": "pypi-json-2026-10-27",
        "sha256": "165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f",
        "version": "9.1.0.70",
    },
    {
        "filename": "transformers-4.57.6-py3-none-any.whl",
        "name": "transformers",
        "provenance": "pypi-json-2026-10-27",
        "sha256": "4c9e9de11333ddfe5114bc872c9f370509198acf0b87a832a0ab9458e2bd0550",
        "version": "4.57.6",
    },
    {
        "filename": "black-24.10.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl",
        "name": "black",
        "provenance": "pypi-json-2026-10-27",
        "sha256": "14b3502784f09ce2443830e3133dacf2c0110d45191ed470ecb04d0f5f6fcb0f",
        "version": "24.10.0",
    },
    {
        "filename": "black-24.10.0-py3-none-any.whl",
        "name": "black",
        "provenance": "pypi-json-2026-10-27",
        "sha256": "3bb2b7a1f7b685f85b11fed1ef10f8a9148bceb49853e47a294a3dd963c1dd7d",
        "version": "24.10.0",
    },
    {
        "filename": "triton-3.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "name": "triton",
        "provenance": "pypi-json-2026-10-27",
        "sha256": "c8182f42fd8080a7d39d666814fa36c5e30cc00ea7eeeb1a2983dbb4c99a0fdc",
        "version": "3.1.0",
    },
    {
        "filename": "einops-0.8.2-py3-none-any.whl",
        "name": "einops",
        "provenance": "pypi-json-2026-10-27",
        "sha256": "54058201ac7087911181bfec4af6091bb59380360f069276601256a76af08193",
        "version": "0.8.2",
    },
    {
        "filename": "huggingface_hub-0.36.2-py3-none-any.whl",
        "name": "huggingface-hub",
        "provenance": "pypi-json-2026-10-27",
        "sha256": "48f0c8eac16145dfce371e9d2d7772854a4f591bcb56c9cf548accf531d54270",
        "version": "0.36.2",
    },
)

# Versions read in gollie-l40s. Torch metadata does not require these versions.
# nvjitlink 12.9.86 has not been shown to load under the cu121 stack; R1/R2 remain closed.
EXP48_WHEELHOUSE_NAME = "gollie-wheelhouse-exp48"
EXP49_WHEELHOUSE_NAME = "gollie-wheelhouse-exp49"
EXP48_REUSED_COUNT = 8
OFFLINE_RESOLVE_ARGUMENTS = (
    "--dry-run",
    "--ignore-installed",
    "--no-index",
    "--find-links",
    "--only-binary=:all:",
)
REPRODUCTION_WHEELS = (
    {
        "filename": "setuptools-78.1.0-py3-none-any.whl",
        "name": "setuptools",
        "pinned_by_torch": False,
        "provenance": "pypi-json-2026-10-27",
        "requires_dist": (),
        "sha256": "3e386e96793c8702ae83d17b853fb93d3e09ef82ec62722e61da5cd22376dcd8",
        "source": "gollie-l40s",
        "url": "https://files.pythonhosted.org/packages/54/21/f43f0a1fa8b06b32812e0975981f4677d28e0f3271601dc88ac5a5b83220/setuptools-78.1.0-py3-none-any.whl",
        "version": "78.1.0",
    },
    {
        "filename": "nvidia_nvjitlink_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        "name": "nvidia-nvjitlink-cu12",
        "pinned_by_torch": False,
        "provenance": "pypi-json-2026-10-27",
        "requires_dist": (),
        "runtime_compatibility": "pending_r1_r2",
        "sha256": "e3f1171dbdc83c5932a45f0f4c99180a70de9bd2718c1ab77d14104f6d7147f9",
        "source": "gollie-l40s",
        "url": "https://files.pythonhosted.org/packages/46/0c/c75bbfb967457a0b7670b8ad267bfc4fffdf341c074e0a80db06c24ccfd4/nvidia_nvjitlink_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        "version": "12.9.86",
    },
)

# Versions read in gollie-l40s on 2026-10-27. Digests are PyPI JSON, not wheel bodies.
# A cache file is recovered only when its SHA-256 is one of these digests.
OBSERVED_TRANSITIVE_WHEELS = (
    {"name": "click", "version": "8.5.0", "filename": "click-8.5.0-py3-none-any.whl", "sha256": "255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360", "url": "https://files.pythonhosted.org/packages/58/50/6c0d534c5f134586a8e1ba4e330569e32f057e33372ae556463212fb4cd3/click-8.5.0-py3-none-any.whl"},
    {"name": "filelock", "version": "3.32.3", "filename": "filelock-3.32.3-py3-none-any.whl", "sha256": "7f0ca4bcc0e181c60dbbd8aa9ab5b120ebb99e4e064e83636340056f833a1f09", "url": "https://files.pythonhosted.org/packages/a7/8e/50f46a9c0ce8d2861a394c1347caae037ea0431d2f67d7feb151cbc4649a/filelock-3.32.3-py3-none-any.whl"},
    {"name": "fsspec", "version": "2026.7.0", "filename": "fsspec-2026.7.0-py3-none-any.whl", "sha256": "b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279", "url": "https://files.pythonhosted.org/packages/fd/3c/6a2bf344106328fd04963664a60b9bb6496fc25df8e962fcdc1367285fb9/fsspec-2026.7.0-py3-none-any.whl"},
    {"name": "hf-xet", "version": "1.7.0", "filename": "hf_xet-1.7.0-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", "sha256": "2814a6e999d13464c4d679b788cc5d784eb5a4edfc638a31f10e9a11ab531ef8", "url": "https://files.pythonhosted.org/packages/a3/4c/2bf3b66c215d409655f28de1622393dde04c9461280d48c7924bb3b2decd/hf_xet-1.7.0-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"},
    {"name": "jinja2", "version": "3.1.6", "filename": "jinja2-3.1.6-py3-none-any.whl", "sha256": "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67", "url": "https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl"},
    {"name": "mpmath", "version": "1.3.0", "filename": "mpmath-1.3.0-py3-none-any.whl", "sha256": "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c", "url": "https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl"},
    {"name": "mypy-extensions", "version": "1.1.0", "filename": "mypy_extensions-1.1.0-py3-none-any.whl", "sha256": "1be4cccdb0f2482337c4743e60421de3a356cd97508abadd57d47403e94f5505", "url": "https://files.pythonhosted.org/packages/79/7b/2c79738432f5c924bef5071f933bcc9efd0473bac3b4aa584a6f7c1c8df8/mypy_extensions-1.1.0-py3-none-any.whl"},
    {"name": "networkx", "version": "3.6.1", "filename": "networkx-3.6.1-py3-none-any.whl", "sha256": "d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762", "url": "https://files.pythonhosted.org/packages/9e/c9/b2622292ea83fbb4ec318f5b9ab867d0a28ab43c5717bb85b0a5f6b3b0a4/networkx-3.6.1-py3-none-any.whl"},
    {"name": "numpy", "version": "2.5.4", "filename": "numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", "sha256": "fbde6962867ee75b48b0ee29b2b9372ec5d617799dbaf38e82dc0596f2f7738a", "url": "https://files.pythonhosted.org/packages/43/a3/c799c62e19c337e6d3770b08e475887fb30ce8477d3c09efca6b2f0228a6/numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"},
    {"name": "packaging", "version": "26.3", "filename": "packaging-26.3-py3-none-any.whl", "sha256": "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c", "url": "https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl"},
    {"name": "pathspec", "version": "1.1.1", "filename": "pathspec-1.1.1-py3-none-any.whl", "sha256": "a00ce642f577bf7f473932318056212bc4f8bfdf53128c78bbd5af0b9b20b189", "url": "https://files.pythonhosted.org/packages/f1/d9/7fb5aa316bc299258e68c73ba3bddbc499654a07f151cba08f6153988714/pathspec-1.1.1-py3-none-any.whl"},
    {"name": "platformdirs", "version": "4.13.5", "filename": "platformdirs-4.13.5-py3-none-any.whl", "sha256": "53d3ec4280b815c1e0b02edad7f9ee8fda1ba9c0865dd9c2e6e63849ce82fc1e", "url": "https://files.pythonhosted.org/packages/6c/8a/1d8cb3842909dc193a73835adedf412e32e18f8ffd59f43983949622fe6f/platformdirs-4.13.5-py3-none-any.whl"},
    {"name": "pyyaml", "version": "6.0.3", "filename": "pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl", "sha256": "ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc", "url": "https://files.pythonhosted.org/packages/8b/9d/b3589d3877982d4f2329302ef98a8026e7f4443c765c46cfecc8858c6b4b/pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"},
    {"name": "regex", "version": "2026.9.29", "filename": "regex-2026.9.29-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl", "sha256": "39ab5894d971f9ac68baa6eca5c50387db579cfcacf36ae8df3feceb1815e6d0", "url": "https://files.pythonhosted.org/packages/26/83/d2fbd2e4e3afb1167daa825187d196f313cbaa1a4768f311fb041bb0e3d2/regex-2026.9.29-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"},
    {"name": "requests", "version": "2.34.2", "filename": "requests-2.34.2-py3-none-any.whl", "sha256": "2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0", "url": "https://files.pythonhosted.org/packages/a0/f4/c67b0b3f1b9245e8d266f0f112c500d50e5b4e83cb6f3b71b6528104182a/requests-2.34.2-py3-none-any.whl"},
    {"name": "safetensors", "version": "0.8.0", "filename": "safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "sha256": "fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774", "url": "https://files.pythonhosted.org/packages/28/50/f203ff3a3ddfe19308efc83c5a3a29ed02bf786732ec35e68bf9162f3365/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"},
    {"name": "tokenizers", "version": "0.22.2", "filename": "tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "sha256": "369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67", "url": "https://files.pythonhosted.org/packages/2e/76/932be4b50ef6ccedf9d3c6639b056a967a86258c6d9200643f01269211ca/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"},
    {"name": "tqdm", "version": "4.70.1", "filename": "tqdm-4.70.1-py3-none-any.whl", "sha256": "c293e525e6fef9c20e8728fd4612df02a0aa31bb5fe91ecd93e123b1b7bffa73", "url": "https://files.pythonhosted.org/packages/a7/03/921a3d3c75785aca9ebfbfcabfbc3a1be12e2ab5265deb026d55a5a3f83e/tqdm-4.70.1-py3-none-any.whl"},
    {"name": "typing-extensions", "version": "4.16.0", "filename": "typing_extensions-4.16.0-py3-none-any.whl", "sha256": "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8", "url": "https://files.pythonhosted.org/packages/49/d3/b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/typing_extensions-4.16.0-py3-none-any.whl"},
    {"name": "markupsafe", "version": "3.0.3", "filename": "markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl", "sha256": "d6dd0be5b5b189d31db7cda48b91d7e0a9795f31430b7f271219ab30f1d3ac9d", "url": "https://files.pythonhosted.org/packages/3c/2e/8d0c2ab90a8c1d9a24f0399058ab8519a3279d1bd4289511d74e909f060e/markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"},
    {"name": "certifi", "version": "2026.10.23", "filename": "certifi-2026.10.23-py3-none-any.whl", "sha256": "e9b54de76e530bb6cf0ccd036787103acb58fd42ba930ea8bf948ddbaa2f61d1", "url": "https://files.pythonhosted.org/packages/32/d1/a3652fe2c19134d0d15922177821d863cbd7e842d53a21bbe7e8e3a1b5ac/certifi-2026.10.23-py3-none-any.whl"},
    {"name": "charset-normalizer", "version": "3.5.2", "filename": "charset_normalizer-3.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl", "sha256": "3d31298449090ab8d47b7b1b2a555ff73cac7ed438a08b7ac160980c7ebed649", "url": "https://files.pythonhosted.org/packages/7f/c5/38806a25ab5e65fc178f39affeda20858efafede2fce1ffc2556cfc9fe73/charset_normalizer-3.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"},
    {"name": "idna", "version": "3.20", "filename": "idna-3.20-py3-none-any.whl", "sha256": "ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c", "url": "https://files.pythonhosted.org/packages/58/a2/bb081bab032533a855d44de1d56f8e8426114ff1ba5d1f07a438a0a654f8/idna-3.20-py3-none-any.whl"},
    {"name": "urllib3", "version": "2.8.0", "filename": "urllib3-2.8.0-py3-none-any.whl", "sha256": "0cf3cae568d36aa9576b28dfb35f11328f1cb974ca7647d9475ebb86c75ac6e3", "url": "https://files.pythonhosted.org/packages/92/9d/c4e665119135114480843e7ab388fa94d8480650450e6f8e26b70d323a4c/urllib3-2.8.0-py3-none-any.whl"},
)
REUSED_DISTRIBUTIONS = frozenset(
    {
        "black",
        "einops",
        "huggingface-hub",
        "nvidia-cudnn-cu12",
        "sympy",
        "torch",
        "transformers",
        "triton",
    }
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


def _is_inside(root: Path, path: Path) -> bool:
    resolved_root = root.resolve()
    resolved = path.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def published_wheels() -> tuple[dict[str, str], ...]:
    """Official digests. The ten missing CUDA wheels stay in the download plan."""
    combined = tuple(OFFICIAL_LINUX_WHEELS) + PUBLISHED_CACHE_WHEELS
    seen: dict[str, str] = {}
    for item in combined:
        previous = seen.get(item["sha256"])
        if previous is not None and previous != item["filename"]:
            raise ValueError(f"published sha256 {item['sha256']} names two files")
        seen[item["sha256"]] = item["filename"]
        if "cu13" in item["filename"] or item["version"].startswith(("2.14", "3.8")):
            raise ValueError(f"refusing incompatible published artifact {item['filename']}")
    return combined


def pending_dependencies() -> dict[str, object]:
    """Versions that are still open. No version is chosen for the unpinned names."""
    closure = prepared_closure()
    return {
        "closure_complete": False,
        "exact_missing": closure["exact_missing"],
        "reason": (
            "The ten CUDA 12 wheels are not in a wheelhouse and an offline dry-run "
            "has not succeeded."
        ),
        "unpinned": closure["unpinned"],
        "version_not_accredited": [
            {
                "name": "setuptools",
                "reason": (
                    "torch 2.5.1+cu121 requires setuptools for Python >= 3.12 with no version. "
                    "gollie-l40s has setuptools 78.1.0; that choice is not a torch pin. "
                    "The published wheel's base Requires-Dist is empty."
                ),
            },
            {
                "name": "nvidia-nvjitlink-cu12",
                "reason": (
                    "nvidia-cusolver-cu12 11.4.5.107 and nvidia-cusparse-cu12 12.1.0.106 "
                    "require nvidia-nvjitlink-cu12 with no version. gollie-l40s has 12.9.86; "
                    "that choice is not a torch pin. The published wheel has no Requires-Dist. "
                    "Runtime compatibility remains pending R1 and R2."
                ),
            },
        ],
    }


def unresolved_transitive() -> list[dict[str, str]]:
    """Requirements with no version observed from gollie-l40s.

    requests, jinja2 and the other names in this list are not given a wheel.
    Their own dependencies were not read, because that would require choosing
    a version.
    """
    reproduced = {canonicalize_name(item["name"]) for item in REPRODUCTION_WHEELS}
    reproduced.update(canonicalize_name(item["name"]) for item in OBSERVED_TRANSITIVE_WHEELS)
    found = []
    for item in prepared_closure()["unpinned"]:
        name = str(item["name"])
        if canonicalize_name(name) in reproduced:
            continue
        found.append(
            {
                "name": name,
                "reason": (
                    "No gollie-l40s version of this requirement was recorded. "
                    "A version match in the HTTP cache would not by itself identify a wheel."
                ),
                "specifier": str(item["specifier"]),
                "status": "VERSION_NOT_OBSERVED",
            }
        )
    return found


def recovery_plan() -> dict[str, object]:
    """Reusable EXP.4.8 wheels, the fixed CUDA 12 downloads, and the two reproduced pins."""
    downloads = []
    for item in OFFICIAL_LINUX_WHEELS:
        if canonicalize_name(item["name"]) not in {
            canonicalize_name(missing["name"]) for missing in prepared_closure()["exact_missing"]
        }:
            continue
        downloads.append(
            {
                **item,
                "action": "DOWNLOAD_REQUIRED",
                "download_authorized": False,
                "pinned_by_torch": True,
            }
        )
    for item in REPRODUCTION_WHEELS:
        downloads.append(
            {
                "action": "DOWNLOAD_REQUIRED",
                "download_authorized": False,
                "filename": item["filename"],
                "name": item["name"],
                "pinned_by_torch": False,
                "provenance": item["provenance"],
                "requires_dist": list(item["requires_dist"]),
                "sha256": item["sha256"],
                "source": item["source"],
                "url": item["url"],
                "version": item["version"],
            }
        )
    return {
        "closure_complete": False,
        "download_authorized": False,
        "installable_wheelhouse": False,
        "nvjitlink_runtime_compatibility": "pending_r1_r2",
        "offline_resolve": list(OFFLINE_RESOLVE_ARGUMENTS),
        "offline_resolve_pass": False,
        "reused_from_exp48": {
            "directory": f"recovery/{EXP48_WHEELHOUSE_NAME}",
            "wheel_count": EXP48_REUSED_COUNT,
            "write": False,
        },
        "selective_download": downloads,
        "unresolved_transitive": unresolved_transitive(),
    }


def reuse_exp48_wheels(
    source: Path,
    destination: Path,
    *,
    authorized: bool,
    scratch_root: Path,
    expected_count: int = EXP48_REUSED_COUNT,
    published: tuple[dict[str, str], ...] | None = None,
) -> dict[str, object]:
    """Copy the verified EXP.4.8 wheels into a new directory. The source is not written."""
    if authorized is not True:
        raise RuntimeError("wheelhouse recovery is not authorized")
    if source.is_symlink() or not source.is_dir():
        raise ValueError("EXP.4.8 wheelhouse must be a real directory")
    if source.name != EXP48_WHEELHOUSE_NAME:
        raise ValueError("source is not the EXP.4.8 wheelhouse")
    resolved_source = source.resolve()
    resolved_destination = destination.resolve()
    if (
        destination.name == EXP48_WHEELHOUSE_NAME
        or resolved_source == resolved_destination
        or resolved_source in resolved_destination.parents
    ):
        raise ValueError("refusing to modify the EXP.4.8 wheelhouse")
    if destination.exists() or destination.is_symlink():
        raise ValueError("refusing to write into an existing directory")
    if not _is_inside(scratch_root, destination) or resolved_destination == scratch_root.resolve():
        raise ValueError("wheelhouse must be a new directory inside SCRATCH_ROOT")
    if not _is_inside(scratch_root, source):
        raise ValueError("EXP.4.8 wheelhouse must stay inside SCRATCH_ROOT")
    forbidden = {"gollie-l40s", "gollie-isolated", "uie-train", "re_te_system_gpu_312", "re_te_system_312"}
    if forbidden & set(resolved_destination.parts):
        raise ValueError("refusing to write inside a venv path")
    catalog = {item["sha256"]: item for item in (published_wheels() if published is None else published)}
    wheels = [
        path
        for path in source.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".whl"
    ]
    if len(wheels) != expected_count:
        raise ValueError(f"EXP.4.8 wheelhouse contains {len(wheels)} wheels, expected {expected_count}")
    staged = []
    for path in sorted(wheels):
        before = path.stat().st_mtime_ns
        payload = path.read_bytes()
        if path.stat().st_mtime_ns != before:
            raise ValueError(f"EXP.4.8 wheel changed while reading {path.name}")
        digest = hashlib.sha256(payload).hexdigest()
        official = catalog.get(digest)
        if official is None or official["filename"] != path.name:
            raise ValueError(f"EXP.4.8 wheel {path.name} does not match a published digest")
        tags = parse_wheel_filename(path.name)[3]
        if not _tags_are_appropriate([str(tag) for tag in tags]):
            raise ValueError(f"EXP.4.8 wheel {path.name} has incompatible tags")
        staged.append((path, payload, digest, before))
    if len({item[0].name for item in staged}) != len(staged):
        raise ValueError("refusing a filename collision")
    if len({item[2] for item in staged}) != len(staged):
        raise ValueError("refusing a digest collision")
    plan = recovery_plan()
    plan["download_authorized"] = False
    created = False
    try:
        destination.mkdir(parents=False, exist_ok=False)
        created = True
        copied = []
        for path, payload, digest, before in staged:
            if path.stat().st_mtime_ns != before:
                raise ValueError(f"EXP.4.8 wheel changed before copy: {path.name}")
            target = destination / path.name
            if target.exists():
                raise ValueError(f"refusing to overwrite {target.name}")
            with target.open("xb") as handle:
                handle.write(payload)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"sha256 mismatch after copy: {target.name}")
            if path.stat().st_mtime_ns != before:
                raise ValueError(f"EXP.4.8 wheel was modified: {path.name}")
            copied.append({"filename": path.name, "sha256": digest})
        report = {
            "closure_complete": False,
            "copied": copied,
            "download_authorized": False,
            "installable_wheelhouse": False,
            "offline_resolve_pass": False,
            "plan": plan,
            "source": str(source.resolve()),
        }
        (destination / "SHA256SUMS").write_text(
            "".join(f"{item['sha256']}  {item['filename']}\n" for item in copied),
            encoding="utf-8",
        )
        (destination / "recovery-plan.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if created:
            shutil.rmtree(destination)
        raise
    return report


def exp472_manifest_path(scratch: Path) -> Path:
    return scratch / EXP472_MANIFEST_RELATIVE


def assert_exp472_manifest_path(scratch: Path, manifest: Path) -> None:
    expected = exp472_manifest_path(scratch)
    if manifest.resolve() != expected.resolve():
        raise ValueError(f"refusing manifest {manifest}; expected {expected}")


def _blocked(record: dict[str, object], reason: str) -> dict[str, object]:
    return {
        "accredited": False,
        "declared_name": record.get("declared_name"),
        "declared_version": record.get("declared_version"),
        "origin_path": record.get("origin_path"),
        "reason": reason,
        "sha256": record.get("sha256"),
    }


def _accredit_origin(
    record: dict[str, object],
    cache: Path,
    published: tuple[dict[str, str], ...],
) -> dict[str, object]:
    """Re-read the cache body. ZIP names alone do not accredit it."""
    origin_raw = record.get("origin_path")
    recorded = record.get("sha256")
    if not isinstance(origin_raw, str) or not isinstance(recorded, str):
        return _blocked(record, "manifest record has no origin path or sha256")
    origin = Path(origin_raw)
    if origin.is_symlink() or not origin.is_file():
        return _blocked(record, "origin is not a regular cache file")
    if not _is_inside(cache, origin) or origin.resolve() == cache.resolve():
        return _blocked(record, "origin is outside the pip cache")
    before = origin.stat().st_mtime_ns
    payload = origin.read_bytes()
    if origin.stat().st_mtime_ns != before:
        return _blocked(record, "cache file changed while reading")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != recorded:
        return _blocked(record, "recomputed sha256 does not match the manifest")
    inspected = inspect_wheel_bytes(payload, origin.resolve())
    if inspected.get("rejection"):
        return _blocked(record, str(inspected["rejection"]))
    if inspected.get("declared_name") != record.get("declared_name"):
        return _blocked(record, "reparsed METADATA name does not match the manifest")
    if inspected.get("declared_version") != record.get("declared_version"):
        return _blocked(record, "reparsed METADATA version does not match the manifest")
    if list(inspected.get("tags") or []) != list(record.get("tags") or []):
        return _blocked(record, "reparsed WHEEL tags do not match the manifest")
    tags = [str(tag) for tag in inspected["tags"]]
    if not _tags_are_appropriate(tags):
        return _blocked(record, "wheel tags are not appropriate for the cu121 cp312 runtime")
    matches = [item for item in published if item["sha256"] == digest]
    if not matches:
        return _blocked(record, "no published sha256 matches this body")
    if len(matches) != 1:
        return _blocked(record, "published sha256 matches more than one official filename")
    official = matches[0]
    if canonicalize_name(official["name"]) != canonicalize_name(str(inspected["declared_name"])):
        return _blocked(record, "published filename names a different distribution")
    if Version(official["version"]) != Version(str(inspected["declared_version"])):
        return _blocked(record, "published filename names a different version")
    filename = official["filename"]
    if Path(filename).name != filename:
        return _blocked(record, "published filename is not a single path component")
    try:
        assert_wheel_filename_matches(
            filename,
            str(inspected["declared_name"]),
            str(inspected["declared_version"]),
            tags,
        )
    except (InvalidWheelFilename, ValueError) as exc:
        return _blocked(record, f"published filename does not match metadata and tags: {exc}")
    return {
        "accredited": True,
        "declared_name": inspected["declared_name"],
        "declared_version": inspected["declared_version"],
        "filename": filename,
        "origin_mtime_ns": before,
        "origin_path": str(origin.resolve()),
        "provenance": official.get("provenance", "published-sha256"),
        "sha256": digest,
    }


def _load_selection_manifest(manifest: Path, expected_sha256: str) -> dict[str, object]:
    payload = manifest.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"manifest sha256 {digest} != {expected_sha256}")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("manifest is not a JSON object")
    if expected_sha256 == EXP472_MANIFEST_SHA256:
        selected = parsed.get("selected")
        rejected = parsed.get("rejected")
        ambiguous = parsed.get("ambiguous")
        if not isinstance(selected, list) or len(selected) != EXP472_SELECTED:
            raise ValueError("EXP.4.7.2 manifest does not contain 8 selected records")
        if not isinstance(rejected, list) or len(rejected) != EXP472_REJECTED:
            raise ValueError("EXP.4.7.2 manifest does not contain 20 rejected records")
        if not isinstance(ambiguous, list) or len(ambiguous) != EXP472_AMBIGUOUS:
            raise ValueError("EXP.4.7.2 manifest does not contain 0 ambiguous records")
        if parsed.get("materialized") is not False or parsed.get("installable_wheelhouse") is not False:
            raise ValueError("EXP.4.7.2 manifest is not an unmaterialized selection")
    return parsed


def materialize_accredited(
    manifest: Path,
    cache: Path,
    destination: Path,
    *,
    authorized: bool,
    expected_sha256: str,
    scratch_root: Path,
    published: tuple[dict[str, str], ...] | None = None,
) -> dict[str, object]:
    """Copy accredited cache bodies into a new SCRATCH directory.

    The cache is only read. An existing destination is refused. The resulting
    directory is not an installable closure: the ten CUDA 12 wheels are still
    absent and no offline dry-run has succeeded.
    """
    if authorized is not True:
        raise RuntimeError("wheel materialization is not authorized")
    if not manifest_path_is_outside_cache(cache, destination):
        raise ValueError("refusing to write the wheelhouse inside the pip cache")
    if destination.exists() or destination.is_symlink():
        raise ValueError("refusing to write into an existing directory")
    if not _is_inside(scratch_root, destination) or destination.resolve() == scratch_root.resolve():
        raise ValueError("wheelhouse must be a new directory inside SCRATCH_ROOT")
    forbidden = {"gollie-l40s", "gollie-isolated", "uie-train", "re_te_system_gpu_312", "re_te_system_312"}
    if forbidden & set(destination.resolve().parts):
        raise ValueError("refusing to write inside a venv path")
    catalog = published_wheels() if published is None else published
    selection = _load_selection_manifest(manifest, expected_sha256)
    selected = selection.get("selected")
    if not isinstance(selected, list):
        raise ValueError("manifest has no selected list")
    accredited = []
    blocked = []
    for record in selected:
        if not isinstance(record, dict):
            blocked.append(_blocked({}, "selected entry is not an object"))
            continue
        if record.get("status") != "SELECTED_CANDIDATE":
            blocked.append(_blocked(record, "record is not a selected candidate"))
            continue
        decision = _accredit_origin(record, cache, catalog)
        if decision["accredited"] is True:
            accredited.append(decision)
        else:
            blocked.append(decision)
    filenames = [str(item["filename"]) for item in accredited]
    if len(filenames) != len(set(filenames)):
        raise ValueError("accredited wheel filenames collide")
    pending = pending_dependencies()
    report: dict[str, object] = {
        "blocked": blocked,
        "closure_complete": False,
        "copied": [],
        "download_authorized": False,
        "installable_wheelhouse": False,
        "materialized": False,
        "pending": pending,
    }
    if not accredited:
        return report
    created = False
    try:
        destination.mkdir(parents=False, exist_ok=False)
        created = True
        copied = []
        for item in accredited:
            filename = str(item["filename"])
            target = destination / filename
            if target.exists():
                raise ValueError(f"refusing to overwrite {target}")
            origin = Path(str(item["origin_path"]))
            before = origin.stat().st_mtime_ns
            if before != item["origin_mtime_ns"]:
                raise ValueError(f"cache file changed before copy: {origin}")
            payload = origin.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"sha256 changed before copy: {origin}")
            with target.open("xb") as handle:
                handle.write(payload)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"sha256 mismatch after copy: {target}")
            if origin.stat().st_mtime_ns != before:
                raise ValueError(f"cache file was modified: {origin}")
            copied.append(
                {
                    "filename": filename,
                    "origin_path": str(origin),
                    "provenance": item["provenance"],
                    "sha256": digest,
                }
            )
        sums = "".join(f"{item['sha256']}  {item['filename']}\n" for item in copied)
        (destination / "SHA256SUMS").write_text(sums, encoding="utf-8")
        report["copied"] = copied
        report["materialized"] = True
        (destination / "materialization.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if created:
            shutil.rmtree(destination)
        raise
    return report


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


def closure_catalog() -> tuple[dict[str, object], ...]:
    """Reused EXP.4.9 wheels plus the official downloads for the observed closure."""
    catalog: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in published_wheels():
        if canonicalize_name(item["name"]) not in REUSED_DISTRIBUTIONS:
            continue
        catalog.append({**item, "download": False, "role": "reused"})
        seen.add(item["sha256"])
    for item in recovery_plan()["selective_download"]:
        if not isinstance(item, dict) or item["sha256"] in seen:
            continue
        catalog.append({**item, "download": True, "role": "official"})
        seen.add(str(item["sha256"]))
    for item in OBSERVED_TRANSITIVE_WHEELS:
        if item["sha256"] in seen:
            raise ValueError(f"duplicate published sha256 {item['sha256']}")
        filename = item["filename"]
        tags = [str(tag) for tag in parse_wheel_filename(filename)[3]]
        if not _tags_are_appropriate(tags):
            raise ValueError(f"refusing incompatible observed wheel {filename}")
        host = item["url"].split("/")[2]
        if host != "files.pythonhosted.org":
            raise ValueError(f"refusing unofficial host {host}")
        catalog.append(
            {
                **item,
                "download": True,
                "provenance": "pypi-json-2026-10-27",
                "role": "observed",
            }
        )
        seen.add(item["sha256"])
    return tuple(catalog)


def _grouped_catalog(
    catalog: tuple[dict[str, object], ...] | None = None,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in closure_catalog() if catalog is None else catalog:
        grouped.setdefault(canonicalize_name(str(item["name"])), []).append(item)
    return grouped


def _matching_record(payload: bytes, records: list[dict[str, object]]) -> dict[str, object] | None:
    digest = hashlib.sha256(payload).hexdigest()
    matches = [item for item in records if item["sha256"] == digest]
    if len(matches) != 1:
        return None
    official = matches[0]
    inspected = inspect_wheel_bytes(payload, Path(str(official["filename"])))
    if not inspected.get("zip_ok"):
        return None
    if inspected.get("rejection") not in {None, "distribution has no exact pin"}:
        return None
    tags = [str(tag) for tag in inspected["tags"]]
    if not _tags_are_appropriate(tags):
        return None
    try:
        assert_wheel_filename_matches(
            str(official["filename"]),
            str(inspected["declared_name"]),
            str(inspected["declared_version"]),
            tags,
        )
    except (InvalidWheelFilename, ValueError):
        return None
    if Version(str(official["version"])) != Version(str(inspected["declared_version"])):
        return None
    return official


def _place_wheel(payload: bytes, target: Path, digest: str) -> None:
    if target.exists() or target.is_symlink():
        raise ValueError(f"refusing to overwrite {target.name}")
    partial = target.with_name("." + target.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError(f"refusing to overwrite {partial.name}")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != digest:
            raise ValueError(f"sha256 mismatch after write: {target.name}")
        os.link(partial, target)
    finally:
        if partial.exists() and not partial.is_symlink():
            partial.unlink()


def _download_official(record: dict[str, object], fetcher) -> bytes:
    url = str(record["url"])
    host = url.split("/")[2]
    if host != "files.pythonhosted.org":
        raise ValueError(f"refusing unofficial host {host}")
    payload = fetcher(url)
    if not isinstance(payload, bytes):
        raise ValueError("download did not return bytes")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"downloaded sha256 does not match {record['filename']}")
    if _matching_record(payload, [record]) is None:
        raise ValueError(f"downloaded wheel metadata does not match {record['filename']}")
    return payload


def _default_fetcher(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "re_te_system-gollie-wheelhouse"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def assert_wheelhouse_closed(
    destination: Path,
    catalog: tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    """Recompute hashes. A version match without the published digest stays blocked."""
    grouped = _grouped_catalog(catalog)
    wheels = [
        path
        for path in destination.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".whl"
    ]
    by_hash: dict[str, list[Path]] = {}
    results = []
    blocked = []
    for path in sorted(wheels):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        by_hash.setdefault(digest, []).append(path)
        owner = [
            item
            for records in grouped.values()
            for item in records
            if item["sha256"] == digest and item["filename"] == path.name
        ]
        if len(owner) != 1 or _matching_record(payload, owner) is None:
            blocked.append(
                {
                    "filename": path.name,
                    "reason": "wheel is present but its bytes are not a published closure wheel",
                    "status": "blocked",
                }
            )
            continue
        results.append(
            {
                "filename": path.name,
                "name": owner[0]["name"],
                "sha256": digest,
                "status": "present",
            }
        )
    present_names = {canonicalize_name(str(item["name"])) for item in results}
    for name, records in sorted(grouped.items()):
        if name in present_names:
            continue
        blocked.append(
            {
                "filename": records[0]["filename"],
                "name": records[0]["name"],
                "reason": "published wheel is absent from the wheelhouse",
                "status": "blocked",
            }
        )
    return {
        "blocked": blocked,
        "closure_complete": not blocked and len(present_names) == len(grouped),
        "installable_wheelhouse": False,
        "offline_resolve_pass": False,
        "results": results,
    }


def close_wheelhouse(
    cache: Path,
    destination: Path,
    *,
    download_authorized: bool,
    scratch_root: Path,
    catalog: tuple[dict[str, object], ...] | None = None,
    fetcher=None,
) -> dict[str, object]:
    """Recover known wheels from the cache and download only the absent official files."""
    if download_authorized is not True:
        raise RuntimeError("wheel download is not authorized")
    if destination.name == EXP48_WHEELHOUSE_NAME or destination.name != EXP49_WHEELHOUSE_NAME:
        raise ValueError("refusing to modify the EXP.4.8 wheelhouse")
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("EXP.4.9 wheelhouse must already exist")
    if not _is_inside(scratch_root, destination):
        raise ValueError("wheelhouse must stay inside SCRATCH_ROOT")
    resolved = destination.resolve()
    if EXP48_WHEELHOUSE_NAME in resolved.parts:
        raise ValueError("refusing to modify the EXP.4.8 wheelhouse")
    forbidden = {"gollie-l40s", "gollie-isolated", "uie-train", "re_te_system_gpu_312", "re_te_system_312"}
    if forbidden & set(resolved.parts):
        raise ValueError("refusing to write inside a venv path")
    if cache.is_dir() and not manifest_path_is_outside_cache(cache, destination):
        raise ValueError("refusing to write the wheelhouse inside the pip cache")
    grouped = _grouped_catalog(catalog)
    records = [item for values in grouped.values() for item in values]
    by_hash = {str(item["sha256"]): item for item in records}
    results = []
    blocked = []
    existing: dict[str, Path] = {}
    for path in destination.iterdir():
        if path.name.startswith("."):
            blocked.append({"filename": path.name, "reason": "partial file left in the wheelhouse", "status": "blocked"})
            continue
        if not path.is_file() or path.is_symlink() or path.suffix != ".whl":
            continue
        payload = path.read_bytes()
        official = _matching_record(payload, [by_hash[hashlib.sha256(payload).hexdigest()]]) if hashlib.sha256(payload).hexdigest() in by_hash else None
        if official is None or official["filename"] != path.name:
            blocked.append(
                {
                    "filename": path.name,
                    "reason": "wheel is present but its bytes are not a published closure wheel",
                    "status": "blocked",
                }
            )
            continue
        name = canonicalize_name(str(official["name"]))
        if name in existing:
            blocked.append({"filename": path.name, "reason": f"filename collision for {name}", "status": "blocked"})
            continue
        existing[name] = path
        results.append({"filename": path.name, "name": official["name"], "sha256": official["sha256"], "status": "present"})
    cache_hits: dict[str, bytes] = {}
    if cache.is_dir():
        for origin in iter_cache_bodies(cache):
            before = origin.stat().st_mtime_ns
            payload = origin.read_bytes()
            if origin.stat().st_mtime_ns != before:
                continue
            digest = hashlib.sha256(payload).hexdigest()
            official = by_hash.get(digest)
            if official is None or _matching_record(payload, [official]) is None:
                continue
            cache_hits.setdefault(canonicalize_name(str(official["name"])), payload)
    fetch = _default_fetcher if fetcher is None else fetcher
    for name, options in sorted(grouped.items()):
        if name in existing:
            continue
        payload = cache_hits.get(name)
        chosen = options[0]
        if payload is not None:
            chosen = _matching_record(payload, options) or chosen
            target = destination / str(chosen["filename"])
            try:
                _place_wheel(payload, target, str(chosen["sha256"]))
            except ValueError as exc:
                blocked.append({"filename": chosen["filename"], "name": chosen["name"], "reason": str(exc), "status": "blocked"})
                continue
            results.append(
                {
                    "filename": chosen["filename"],
                    "name": chosen["name"],
                    "provenance": chosen.get("provenance", "published-sha256"),
                    "sha256": chosen["sha256"],
                    "status": "recovered",
                }
            )
            continue
        downloadable = [item for item in options if item.get("download") is True and item.get("url")]
        if not downloadable:
            blocked.append(
                {
                    "filename": chosen["filename"],
                    "name": chosen["name"],
                    "reason": "reused wheel is absent and will not be downloaded",
                    "status": "blocked",
                }
            )
            continue
        chosen = downloadable[0]
        try:
            payload = _download_official(chosen, fetch)
            _place_wheel(payload, destination / str(chosen["filename"]), str(chosen["sha256"]))
        except Exception as exc:
            blocked.append(
                {
                    "filename": chosen["filename"],
                    "name": chosen["name"],
                    "reason": str(exc),
                    "status": "blocked",
                }
            )
            continue
        results.append(
            {
                "filename": chosen["filename"],
                "name": chosen["name"],
                "provenance": chosen.get("provenance", "pypi-json-2026-10-27"),
                "sha256": chosen["sha256"],
                "status": "downloaded",
                "url": chosen["url"],
            }
        )
    report = {
        "blocked": blocked,
        "closure_complete": not blocked and len({canonicalize_name(str(item["name"])) for item in results}) == len(grouped),
        "download_authorized": True,
        "installable_wheelhouse": False,
        "offline_resolve_pass": False,
        "results": results,
    }
    (destination / "wheelhouse-close.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report["closure_complete"]:
        lines = [f"{item['sha256']}  {item['filename']}\n" for item in sorted(results, key=lambda item: str(item["filename"]))]
        (destination / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    return report


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
    if args == ["materialize"]:
        if os.environ.get("GOLLIE_MATERIALIZE_AUTHORIZED") != "1":
            raise RuntimeError("wheel materialization is not authorized")
        scratch = Path(os.environ["SCRATCH_ROOT"])
        cache = Path(os.environ["GOLLIE_PIP_HTTP_CACHE"])
        manifest = Path(os.environ["GOLLIE_WHEEL_MANIFEST"])
        destination = Path(os.environ["GOLLIE_WHEELHOUSE"])
        assert_exp472_manifest_path(scratch, manifest)
        _emit(
            materialize_accredited(
                manifest,
                cache,
                destination,
                authorized=True,
                expected_sha256=EXP472_MANIFEST_SHA256,
                scratch_root=scratch,
            )
        )
        return 0
    if args == ["assert-closed"]:
        report = assert_wheelhouse_closed(Path(os.environ["GOLLIE_WHEELHOUSE"]))
        _emit(report)
        return 0 if report["closure_complete"] else 1
    if args == ["close"]:
        if os.environ.get("GOLLIE_DOWNLOAD_AUTHORIZED") != "1":
            raise RuntimeError("wheel download is not authorized")
        report = close_wheelhouse(
            Path(os.environ["GOLLIE_PIP_HTTP_CACHE"]),
            Path(os.environ["GOLLIE_WHEELHOUSE"]),
            download_authorized=True,
            scratch_root=Path(os.environ["SCRATCH_ROOT"]),
        )
        _emit(report)
        return 0 if report["closure_complete"] else 1
    if args == ["recover"]:
        if os.environ.get("GOLLIE_RECOVERY_AUTHORIZED") != "1":
            raise RuntimeError("wheelhouse recovery is not authorized")
        _emit(
            reuse_exp48_wheels(
                Path(os.environ["GOLLIE_EXP48_WHEELHOUSE"]),
                Path(os.environ["GOLLIE_WHEELHOUSE"]),
                authorized=True,
                scratch_root=Path(os.environ["SCRATCH_ROOT"]),
            )
        )
        return 0
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
        "usage: gollie_wheelhouse.py [missing|download-plan|download|select-cache|materialize|recover|close|assert-closed]",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
