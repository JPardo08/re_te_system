"""Pinned FlashAttention wheel. The installed copy is a separate identity.

BUILD_PASS records that the preserved wheel was compiled. It does not mean
the package inside gollie-l40s is that wheel. The extension measured in
gollie-l40s was installed at 14:50 UTC on 2026-10-26, before the corrected
wheel finished at 19:36 UTC. Both declare 2.8.3.post2 and they do not share
binary identity. That mismatch is not an ABI pass and it is not an ABI
failure of the corrected wheel. Static GLIBC maxima do not decide ABI.
The corrected wheel has not been executed on L40S.
"""

from __future__ import annotations


FLASH_ATTN_VERSION = "2.8.3.post2"
FLASH_ATTN_WHEEL_FILENAME = "flash_attn-2.8.3.post2-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN_WHEEL_SHA256 = "8238e887fec100cbead2fb662a041bf8486499df00d8d026f234be3a7af2be82"
FLASH_ATTN_EXTENSION = "flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
FLASH_ATTN_PYTHON_TAG = "cp312"
FLASH_ATTN_RECORDED_GLIBC_MAX = "2.14"
WHEEL_BUILD_FINISHED_AT = "2026-10-26T19:36:00Z"
WHEEL_L40S_KERNEL_VALIDATED = False
OBSERVED_BINARY_IDENTITY = "BINARY_MISMATCH"
RECORDED_WHEEL_BUILD_PASS = True

# Bytes currently inside gollie-l40s. This hash must not be treated as the
# extension stored in the corrected wheel, even if a caller claims they match.
INSTALLED_EXTENSION_SHA256 = "1fed48745de90c41071d324ff0bbdf17c3c054c679279e5bc30c640b74007a03"
INSTALLED_EXTENSION_AT = "2026-10-26T14:50:00Z"
INSTALLED_EXTENSION_GLIBC_MAX = "2.32"

EXPECTED_TORCH = "2.5.1+cu121"
EXPECTED_TRANSFORMERS = "4.57.6"
EXPECTED_BLACK = "24.10.0"
# Versions read from the intact gollie-l40s stack. Their wheel files are not
# assumed to be present in the CESVIMA pip cache.
EXPECTED_TRITON = "3.1.0"
EXPECTED_EINOPS = "0.8.2"
EXPECTED_HUGGINGFACE_HUB = "0.36.2"

GOLLIE_J16_DOWNLOAD_AUTHORIZED = False
GOLLIE_J17_SMOKE_AUTHORIZED = False
ISOLATED_RECOVERY_AUTHORIZED = False
STATIC_GLIBC_IS_ABI_PASS = False


def recorded_flash_attention_evidence() -> dict[str, object]:
    """Facts measured on 2026-10-26/27. None of them is ABI_PASS or a kernel result."""
    return {
        "abi_failed": False,
        "abi_pass": False,
        "abi_verdict": "not_decided",
        "binary_identity": OBSERVED_BINARY_IDENTITY,
        "declared_version": FLASH_ATTN_VERSION,
        "installed_at": INSTALLED_EXTENSION_AT,
        "installed_extension_sha256": INSTALLED_EXTENSION_SHA256,
        "installed_glibc_max": INSTALLED_EXTENSION_GLIBC_MAX,
        "same_declared_version": True,
        "static_glibc_is_abi_pass": STATIC_GLIBC_IS_ABI_PASS,
        "wheel_build_pass": RECORDED_WHEEL_BUILD_PASS,
        "wheel_finished_at": WHEEL_BUILD_FINISHED_AT,
        "wheel_l40s_kernel_validated": WHEEL_L40S_KERNEL_VALIDATED,
        "wheel_sha256": FLASH_ATTN_WHEEL_SHA256,
        "wheel_static_glibc_max": FLASH_ATTN_RECORDED_GLIBC_MAX,
    }


def interpret_binary_identity(identity: str) -> dict[str, object]:
    if identity not in {"MATCH", "BINARY_MISMATCH", "NOT_COMPARED"}:
        raise ValueError(f"unknown binary identity {identity}")
    return {
        "abi_failed": False,
        "abi_pass": False,
        "abi_verdict": "not_decided",
        "accredited_for_reuse": identity == "MATCH",
        "automatic_reinstall": False,
        "binary_identity": identity,
        "compilation": False,
        "installed_package_is_verified_wheel": identity == "MATCH",
    }
