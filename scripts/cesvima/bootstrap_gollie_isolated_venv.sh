#!/bin/bash
# R1: inventory or install an isolated GoLLIE venv under SCRATCH.
# The default mode only reports whether the pinned wheel and the named
# stack wheels are present. It does not create a venv, does not contact a
# package index, and does not read or write gollie-l40s site-packages.
# Install requires GOLLIE_ISOLATED_INSTALL=1 and still refuses a package index,
# source builds, and the existing gollie-l40s directory.
# This script does not run the GPU kernel and does not authorize J16 or J17.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

die() {
  echo "FAIL: $*" >&2
  exit 1
}

creator_is_usable() {
  env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
    "$1" -c 'import sys, venv, ensurepip, pip; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' \
    >/dev/null 2>&1
}

# No CESVIMA Lmod Python path is recorded. Prefer the base interpreter named by
# gollie-l40s pyvenv.cfg when it is outside the scientific venvs. Otherwise the
# gollie-l40s executable may create the new venv. It writes only the destination,
# and the destination is created without system site-packages.
resolve_gollie_creator() {
  local original="${GOLLIE_ORIGINAL_VENV:-${SCRATCH_ROOT}/venvs/gollie-l40s}"
  local home candidate
  if [[ -n "${GOLLIE_PYTHON312:-}" && "${GOLLIE_PYTHON312}" != *"/venvs/gollie-l40s/"* ]]; then
    if creator_is_usable "${GOLLIE_PYTHON312}"; then
      return 0
    fi
    die "GOLLIE_PYTHON312 cannot import venv, ensurepip and pip under Python 3.12"
  fi
  home=""
  if [[ -f "${original}/pyvenv.cfg" ]]; then
    home=$(sed -n 's/^home[[:space:]]*=[[:space:]]*//p' "${original}/pyvenv.cfg" | head -n 1)
  fi
  if [[ -n "${home}" && "${home}" != *"/venvs/gollie-l40s"* && "${home}" != *"/venvs/gollie-isolated"* && "${home}" != *"/venvs/uie-train"* && "${home}" != *"/venvs/re_te_system_gpu_312"* && "${home}" != *"/venvs/re_te_system_312"* ]]; then
    for candidate in "${home}/python3.12" "${home}/python3" "${home}/python"; do
      if [[ -x "${candidate}" ]] && creator_is_usable "${candidate}"; then
        export GOLLIE_PYTHON312="${candidate}"
        echo "CREATOR ${GOLLIE_PYTHON312}"
        return 0
      fi
    done
  fi
  candidate="${original}/bin/python"
  if [[ -x "${candidate}" ]] && creator_is_usable "${candidate}"; then
    export GOLLIE_PYTHON312="${candidate}"
    echo "CREATOR ${GOLLIE_PYTHON312}"
    return 0
  fi
  die "no Python 3.12 with venv, ensurepip and pip can create the isolated venv"
}

require_runtime_env() {
  if [[ -n "${GOLLIE_BOOTSTRAP_ALLOW_COMPILE:-}" ]]; then
    die "FlashAttention compilation is forbidden"
  fi
  if [[ -z "${SYSTEM_ROOT:-}" || -z "${SCRATCH_ROOT:-}" ]]; then
    die "SYSTEM_ROOT and SCRATCH_ROOT are required"
  fi
  if [[ -z "${GOLLIE_ISOLATED_VENV:-}" ]]; then
    die "GOLLIE_ISOLATED_VENV is required"
  fi
  export GOLLIE_ORIGINAL_VENV="${GOLLIE_ORIGINAL_VENV:-${SCRATCH_ROOT}/venvs/gollie-l40s}"
  resolve_gollie_creator
  if [[ -z "${GOLLIE_PYTHON312:-}" || ! -x "${GOLLIE_PYTHON312}" ]]; then
    die "GOLLIE_PYTHON312 must be an executable Python 3.12 interpreter"
  fi
  if ! creator_is_usable "${GOLLIE_PYTHON312}"; then
    die "GOLLIE_PYTHON312 cannot import venv, ensurepip and pip under Python 3.12"
  fi
}

require_install_env() {
  require_runtime_env
  if [[ -z "${GOLLIE_FLASH_ATTN_WHEEL:-}" || -z "${GOLLIE_WHEELHOUSE:-}" ]]; then
    die "GOLLIE_FLASH_ATTN_WHEEL and GOLLIE_WHEELHOUSE are required"
  fi
}

assert_isolated_path() {
  "${GOLLIE_PYTHON312}" - <<'PY'
import os
import sys
from pathlib import Path

isolated = Path(os.environ["GOLLIE_ISOLATED_VENV"]).resolve()
scratch = Path(os.environ["SCRATCH_ROOT"]).resolve()
venvs = (scratch / "venvs").resolve()
forbidden = {"gollie-l40s", "uie-train", "re_te_system_gpu_312", "re_te_system_312"}
if isolated.parent != venvs or isolated.name in forbidden:
    print(f"FAIL: refusing isolated path {isolated}; it must be a new directory under {venvs}", file=sys.stderr)
    raise SystemExit(1)
original = os.environ.get("GOLLIE_ORIGINAL_VENV") or os.environ.get("GOLLIE_VENV", "")
if original and Path(original).resolve() == isolated:
    print(f"FAIL: refusing to use the existing venv {isolated} as the isolated target", file=sys.stderr)
    raise SystemExit(1)
protected = []
for key in ("HF_HOME", "P3_MODELS_ROOT", "P3_UPSTREAM_ROOT", "P3_RUNS_ROOT"):
    raw = os.environ.get(key, "")
    if raw:
        protected.append(Path(raw).resolve())
for root in protected:
    if isolated == root or root in isolated.parents:
        print(f"FAIL: isolated venv overlaps protected path {root}", file=sys.stderr)
        raise SystemExit(1)
print(f"ISOLATED_PATH {isolated}")
PY
}

assert_wheel_file() {
  "${GOLLIE_PYTHON312}" - <<'PY'
import hashlib
import os
import sys
from pathlib import Path

from flash_attn_pin import FLASH_ATTN_WHEEL_FILENAME, FLASH_ATTN_WHEEL_SHA256

wheel = Path(os.environ["GOLLIE_FLASH_ATTN_WHEEL"]).resolve()
roots = [Path(os.environ["SCRATCH_ROOT"]).resolve()]
project = os.environ.get("PROJECT_ROOT", "")
if project:
    roots.append(Path(project).resolve())
if not any(wheel == root or root in wheel.parents for root in roots):
    print("FAIL: wheel is outside SCRATCH_ROOT and PROJECT_ROOT", file=sys.stderr)
    raise SystemExit(1)
if wheel.name != FLASH_ATTN_WHEEL_FILENAME or not wheel.is_file():
    print(
        f"FAIL: wheel must be the file {FLASH_ATTN_WHEEL_FILENAME}",
        file=sys.stderr,
    )
    raise SystemExit(1)
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
if digest != FLASH_ATTN_WHEEL_SHA256:
    print(
        f"FAIL: wheel sha256 {digest} != {FLASH_ATTN_WHEEL_SHA256}",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"WHEEL_SHA256 {digest}")
PY
}

classify_wheelhouse() {
  "${GOLLIE_PYTHON312}" - <<'PY'
import os
import sys
from pathlib import Path

from flash_attn_pin import (
    EXPECTED_BLACK,
    EXPECTED_EINOPS,
    EXPECTED_HUGGINGFACE_HUB,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    EXPECTED_TRITON,
)

PINS = (
    ("torch", EXPECTED_TORCH, True),
    ("transformers", EXPECTED_TRANSFORMERS, False),
    ("black", EXPECTED_BLACK, False),
    ("triton", EXPECTED_TRITON, True),
    ("einops", EXPECTED_EINOPS, False),
    ("huggingface_hub", EXPECTED_HUGGINGFACE_HUB, False),
)
wheelhouse = Path(os.environ["GOLLIE_WHEELHOUSE"])
if not wheelhouse.is_dir():
    print(f"FAIL: wheelhouse is not a directory: {wheelhouse}", file=sys.stderr)
    raise SystemExit(1)

def acceptable(filename: str, distribution: str, version: str, binary_only: bool) -> bool:
    prefix = f"{distribution.replace('-', '_')}-{version}-"
    if not filename.startswith(prefix) or not filename.endswith(".whl"):
        return False
    lower = filename.lower()
    if any(token in lower for token in ("macosx", "win32", "win_amd64", "aarch64", "arm64", "i686")):
        return False
    pure = filename.endswith("py3-none-any.whl")
    linux = ("manylinux" in lower or "linux_" in lower) and "x86_64" in lower and "cp312" in filename
    if binary_only:
        return linux and not pure
    return pure or linux

ambiguous = False
missing = False
for distribution, version, binary_only in PINS:
    matches = [
        path
        for path in sorted(wheelhouse.glob("*.whl"))
        if acceptable(path.name, distribution, version, binary_only)
    ]
    if len(matches) == 1:
        print(f"PRESENT {distribution} {matches[0].name}")
    elif len(matches) == 0:
        missing = True
        print(f"MISSING {distribution} {version}")
    else:
        ambiguous = True
        print(f"AMBIGUOUS {distribution} {len(matches)}")
if ambiguous:
    raise SystemExit(1)
if missing:
    raise SystemExit(2)
PY
}

inventory_main() {
  require_install_env
  local path_status=0
  local wheel_status=0
  local class_status=0
  local quiet="${1:-}"
  set +e
  assert_isolated_path
  path_status=$?
  assert_wheel_file
  wheel_status=$?
  classify_wheelhouse
  class_status=$?
  set -e
  if [[ "${quiet}" != "quiet" ]]; then
    echo "INSTALL not performed"
  fi
  if [[ "${path_status}" -ne 0 || "${wheel_status}" -ne 0 || "${class_status}" -eq 1 ]]; then
    return 1
  fi
  return "${class_status}"
}

dry_run_stack() {
  # PYTHONPATH is only this script directory. The creator venv is not inherited.
  env -u PYTHONHOME -u VIRTUAL_ENV \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${SCRIPT_DIR}" \
    PIP_NO_INDEX=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL= \
    PIP_EXTRA_INDEX_URL= \
    PIP_FIND_LINKS="${GOLLIE_WHEELHOUSE}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    "${GOLLIE_PYTHON312}" -m pip install \
      --dry-run \
      --ignore-installed \
      --no-index \
      --find-links "${GOLLIE_WHEELHOUSE}" \
      --only-binary=:all: \
      --disable-pip-version-check \
      "torch==${EXPECTED_TORCH_PIN}" \
      "transformers==${EXPECTED_TRANSFORMERS_PIN}" \
      "black==${EXPECTED_BLACK_PIN}" \
      "triton==${EXPECTED_TRITON_PIN}" \
      "einops==${EXPECTED_EINOPS_PIN}" \
      "huggingface_hub==${EXPECTED_HUGGINGFACE_HUB_PIN}"
}

load_pin_shell_vars() {
  local output line key value
  output="$("${GOLLIE_PYTHON312}" - <<'PY'
from flash_attn_pin import (
    EXPECTED_BLACK,
    EXPECTED_EINOPS,
    EXPECTED_HUGGINGFACE_HUB,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    EXPECTED_TRITON,
)
print(f"EXPECTED_TORCH_PIN={EXPECTED_TORCH}")
print(f"EXPECTED_TRANSFORMERS_PIN={EXPECTED_TRANSFORMERS}")
print(f"EXPECTED_BLACK_PIN={EXPECTED_BLACK}")
print(f"EXPECTED_TRITON_PIN={EXPECTED_TRITON}")
print(f"EXPECTED_EINOPS_PIN={EXPECTED_EINOPS}")
print(f"EXPECTED_HUGGINGFACE_HUB_PIN={EXPECTED_HUGGINGFACE_HUB}")
PY
)"
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    case "${key}" in
      EXPECTED_TORCH_PIN|EXPECTED_TRANSFORMERS_PIN|EXPECTED_BLACK_PIN|EXPECTED_TRITON_PIN|EXPECTED_EINOPS_PIN|EXPECTED_HUGGINGFACE_HUB_PIN)
        printf -v "${key}" '%s' "${value}"
        ;;
      *)
        die "unexpected pin line"
        ;;
    esac
  done <<< "${output}"
}

resolve_main() {
  require_install_env
  assert_isolated_path
  load_pin_shell_vars
  dry_run_stack
}

install_main() {
  if [[ "${GOLLIE_ISOLATED_INSTALL:-}" != "1" ]]; then
    die "R1 install is not authorized"
  fi
  require_install_env
  assert_isolated_path
  assert_wheel_file
  wheelhouse_main assert-closed
  if [[ -e "${GOLLIE_ISOLATED_VENV}" || -L "${GOLLIE_ISOLATED_VENV}" ]]; then
    die "isolated venv already exists; refusing to modify it"
  fi
  load_pin_shell_vars
  set +e
  dry_run_stack
  dry_status=$?
  set -e
  DRY_STATUS="${dry_status}" "${GOLLIE_PYTHON312}" - <<'PY'
import json
import os
from pathlib import Path

destination = Path(os.environ["SCRATCH_ROOT"]) / "recovery" / "gollie-offline-resolve.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    json.dumps(
        {
            "arguments": [
                "--dry-run",
                "--ignore-installed",
                "--no-index",
                "--find-links",
                "--only-binary=:all:",
            ],
            "exit": int(os.environ["DRY_STATUS"]),
            "offline_resolve_pass": os.environ["DRY_STATUS"] == "0",
            "wheelhouse": os.environ["GOLLIE_WHEELHOUSE"],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
  if [[ "${dry_status}" -ne 0 ]]; then
    die "offline dry-run failed"
  fi
  env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
    "${GOLLIE_PYTHON312}" -m venv "${GOLLIE_ISOLATED_VENV}"
  "${GOLLIE_ISOLATED_VENV}/bin/python" - <<'PY'
import os
import sys
from pathlib import Path

isolated = Path(os.environ["GOLLIE_ISOLATED_VENV"]).resolve()
original = Path(os.environ["GOLLIE_ORIGINAL_VENV"]).resolve()
prefix = Path(sys.prefix).resolve()
if prefix != isolated or isolated == original:
    raise SystemExit("isolated interpreter prefix escaped its venv")
config = (isolated / "pyvenv.cfg").read_text(encoding="utf-8")
if "include-system-site-packages = false" not in config:
    raise SystemExit("isolated venv would inherit another environment")
if any(original.as_posix() in path for path in sys.path):
    raise SystemExit("original gollie-l40s is visible on the isolated sys.path")
print("ISOLATED_PYTHON", sys.executable)
PY
  env -u PYTHONHOME -u VIRTUAL_ENV \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${SCRIPT_DIR}" \
    PIP_NO_INDEX=1 \
    PIP_INDEX_URL= \
    PIP_EXTRA_INDEX_URL= \
    PIP_FIND_LINKS="${GOLLIE_WHEELHOUSE}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    "${GOLLIE_ISOLATED_VENV}/bin/python" -m pip install \
      --no-index \
      --find-links "${GOLLIE_WHEELHOUSE}" \
      --only-binary=:all: \
      --disable-pip-version-check \
      "torch==${EXPECTED_TORCH_PIN}" \
      "transformers==${EXPECTED_TRANSFORMERS_PIN}" \
      "black==${EXPECTED_BLACK_PIN}" \
      "triton==${EXPECTED_TRITON_PIN}" \
      "einops==${EXPECTED_EINOPS_PIN}" \
      "huggingface_hub==${EXPECTED_HUGGINGFACE_HUB_PIN}"
  env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV PYTHONNOUSERSITE=1 \
    PIP_NO_INDEX=1 \
    PIP_INDEX_URL= \
    PIP_EXTRA_INDEX_URL= \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    "${GOLLIE_ISOLATED_VENV}/bin/python" -m pip install \
      --no-index \
      --only-binary=:all: \
      --no-deps \
      --disable-pip-version-check \
      "${GOLLIE_FLASH_ATTN_WHEEL}"
  PYTHONPATH="${SCRIPT_DIR}" "${GOLLIE_ISOLATED_VENV}/bin/python" - <<'PY'
import importlib.metadata
import os
import subprocess
from pathlib import Path

from cesvima_common import sha256_file, write_json
from flash_attn_pin import (
    EXPECTED_BLACK,
    EXPECTED_EINOPS,
    EXPECTED_HUGGINGFACE_HUB,
    EXPECTED_TORCH,
    EXPECTED_TRANSFORMERS,
    EXPECTED_TRITON,
    FLASH_ATTN_VERSION,
    FLASH_ATTN_WHEEL_SHA256,
    INSTALLED_EXTENSION_SHA256,
)
from gollie_environment import _find_extension, wheel_extension_sha256

venv = Path(os.environ["GOLLIE_ISOLATED_VENV"]).resolve()
wheel = Path(os.environ["GOLLIE_FLASH_ATTN_WHEEL"]).resolve()
python = venv / "bin" / "python"
expected = {
    "torch": EXPECTED_TORCH,
    "transformers": EXPECTED_TRANSFORMERS,
    "black": EXPECTED_BLACK,
    "triton": EXPECTED_TRITON,
    "einops": EXPECTED_EINOPS,
    "huggingface_hub": EXPECTED_HUGGINGFACE_HUB,
    "flash_attn": FLASH_ATTN_VERSION,
}
versions = {}
mismatch = False
for name, wanted in expected.items():
    try:
        found = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        found = None
    versions[name] = found
    if found != wanted:
        mismatch = True
checked = subprocess.run([str(python), "-m", "pip", "check"], capture_output=True, text=True)
extension = _find_extension(venv)
extension_sha = sha256_file(extension) if extension is not None else None
member_sha = wheel_extension_sha256(wheel)
identity_ok = (
    extension_sha is not None
    and member_sha is not None
    and extension_sha == member_sha
    and extension_sha != INSTALLED_EXTENSION_SHA256
)
status = "PREP_PASS" if not mismatch and checked.returncode == 0 and identity_ok else "PREP_FAIL"
write_json(
    venv / "isolated_prep_record.json",
    {
        "abi_pass": False,
        "gpu_preflight": False,
        "installed_extension_path": None if extension is None else str(extension),
        "installed_extension_sha256": extension_sha,
        "j16_authorized": False,
        "j17_authorized": False,
        "original_gollie_l40s_modified": False,
        "pip_check_exit": checked.returncode,
        "pip_check_output": (checked.stdout + checked.stderr).strip(),
        "preflight_pass": False,
        "status": status,
        "venv": str(venv),
        "versions": versions,
        "wheel_extension_sha256": member_sha,
        "wheel_path": str(wheel),
        "wheel_sha256": FLASH_ATTN_WHEEL_SHA256,
    },
)
print(f"PREP_RECORD {status}", flush=True)
raise SystemExit(0 if status == "PREP_PASS" else 1)
PY
}

assert_prep_main() {
  require_runtime_env
  if [[ -z "${GOLLIE_ORIGINAL_VENV:-}" ]]; then
    die "GOLLIE_ORIGINAL_VENV is required and must remain gollie-l40s"
  fi
  assert_isolated_path
  "${GOLLIE_PYTHON312}" - <<'PY'
import json
import os
import sys
from pathlib import Path

from flash_attn_pin import FLASH_ATTN_EXTENSION, FLASH_ATTN_WHEEL_SHA256, INSTALLED_EXTENSION_SHA256

isolated = Path(os.environ["GOLLIE_ISOLATED_VENV"]).resolve()
original = Path(os.environ["GOLLIE_ORIGINAL_VENV"]).resolve()
selected = Path(os.environ["GOLLIE_VENV"]).resolve()
record_path = isolated / "isolated_prep_record.json"
if selected != isolated or isolated == original:
    print("FAIL: J15 must select the isolated venv after env.sh", file=sys.stderr)
    raise SystemExit(1)
if not (isolated / "bin" / "python").is_file() or not record_path.is_file():
    print("FAIL: isolated prep record is missing", file=sys.stderr)
    raise SystemExit(1)
runs = os.environ.get("P3_RUNS_ROOT", "")
if runs and Path(runs).resolve() in record_path.parents:
    print("FAIL: prep record must not be written into preflight reports", file=sys.stderr)
    raise SystemExit(1)
record = json.loads(record_path.read_text(encoding="utf-8"))
extension = record.get("installed_extension_path")
checks = [
    record.get("status") == "PREP_PASS",
    record.get("venv") == str(isolated),
    record.get("wheel_sha256") == FLASH_ATTN_WHEEL_SHA256,
    record.get("abi_pass") is False,
    record.get("preflight_pass") is False,
    record.get("gpu_preflight") is False,
    record.get("j16_authorized") is False,
    record.get("j17_authorized") is False,
    record.get("installed_extension_sha256") not in {None, INSTALLED_EXTENSION_SHA256},
    isinstance(extension, str) and extension.endswith("/" + FLASH_ATTN_EXTENSION),
    Path(extension).is_file() if isinstance(extension, str) else False,
]
if not all(checks):
    print("FAIL: isolated prep record is not an accredited environment", file=sys.stderr)
    raise SystemExit(1)
print("PREP_ASSERT PASS")
PY
}

assert_modeling_main() {
  if [[ -z "${SYSTEM_ROOT:-}" || -z "${HF_HUB_CACHE:-}" || -z "${GOLLIE_PYTHON312:-}" ]]; then
    die "SYSTEM_ROOT, HF_HUB_CACHE and GOLLIE_PYTHON312 are required"
  fi
  export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  "${GOLLIE_PYTHON312}" - <<'PY'
import os
import sys
from pathlib import Path

import j15_gollie_preflight as j15

cache = Path(os.environ["HF_HUB_CACHE"]).resolve()
folder = "models--" + j15.GOLLIE_REPO.replace("/", "--")
path = cache / folder / "snapshots" / j15.GOLLIE_REVISION / j15.MODELING_FILE
if path.is_symlink() or not path.is_file():
    print(f"FAIL: {j15.MODELING_FILE} is absent at {path}", file=sys.stderr)
    print("REFUSING_DOWNLOAD modeling file and weights", file=sys.stderr)
    raise SystemExit(1)
try:
    j15.audit_modeling_file(path)
except j15.ModelingIdentityError as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    raise SystemExit(1)
print(f"LOCAL_MODELING {path}")
PY
}

wheelhouse_main() {
  require_runtime_env
  "${GOLLIE_PYTHON312}" "${SCRIPT_DIR}/gollie_wheelhouse.py" "$@"
}

main() {
  local mode="${1:-inventory}"
  case "${mode}" in
    inventory) inventory_main ;;
    resolve) resolve_main ;;
    install) install_main ;;
    assert-prep) assert_prep_main ;;
    assert-modeling) assert_modeling_main ;;
    missing|download-plan|download) wheelhouse_main "${mode}" ;;
    # Manifest only. select-cache does not copy a wheelhouse and does not install.
    select-cache) wheelhouse_main select-cache ;;
    # Copy only. Requires GOLLIE_MATERIALIZE_AUTHORIZED=1 and does not install.
    materialize)
      if [[ "${GOLLIE_MATERIALIZE_AUTHORIZED:-}" != "1" ]]; then
        die "wheel materialization is not authorized"
      fi
      wheelhouse_main materialize
      ;;
    # Read EXP.4.8 and copy into a new directory. Does not download or install.
    recover)
      if [[ "${GOLLIE_RECOVERY_AUTHORIZED:-}" != "1" ]]; then
        die "wheelhouse recovery is not authorized"
      fi
      wheelhouse_main recover
      ;;
    # Fill the existing EXP.4.9 wheelhouse. Does not install and does not write EXP.4.8.
    close)
      if [[ "${GOLLIE_DOWNLOAD_AUTHORIZED:-}" != "1" ]]; then
        die "wheel download is not authorized"
      fi
      wheelhouse_main close
      ;;
    assert-closed) wheelhouse_main assert-closed ;;
    fetch-modeling)
      if [[ "${GOLLIE_DOWNLOAD_AUTHORIZED:-}" != "1" ]]; then
        die "wheel download is not authorized"
      fi
      require_runtime_env
      if [[ -z "${HF_HUB_CACHE:-}" ]]; then
        die "HF_HUB_CACHE is required"
      fi
      "${GOLLIE_PYTHON312}" - <<'PY'
import os
import sys

import j15_gollie_preflight as j15

try:
    outcome = j15.ensure_modeling_file(os.environ["HF_HUB_CACHE"], authorized=True)
except j15.ModelingIdentityError as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    raise SystemExit(1)
print(f"MODELING_{outcome}")
PY
      ;;
    *) die "usage: bootstrap_gollie_isolated_venv.sh [inventory|resolve|install|assert-prep|assert-modeling|missing|download-plan|download|select-cache|materialize|recover|close|assert-closed|fetch-modeling]" ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
