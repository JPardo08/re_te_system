#!/bin/bash
# One GoLLIE path: prepare, install, submit.
# prepare downloads only missing official wheels and the single modeling module.
# install creates gollie-isolated after an offline dry-run exit 0.
# submit sends only j15_gollie_isolated_preflight.sbatch.
# prepare and install belong on a CPU node of partition standard.
# The login node only submits that job. Compute-node egress is unverified;
# if the download cannot leave the CPU node, the job stops before install.
# Stop after the first failing phase. This script does not chain them.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BOOTSTRAP="${SCRIPT_DIR}/bootstrap_gollie_isolated_venv.sh"

die() {
  echo "FAIL: $*" >&2
  exit 1
}

require_common() {
  if [[ -n "${GOLLIE_BOOTSTRAP_ALLOW_COMPILE:-}" ]]; then
    die "FlashAttention compilation is forbidden"
  fi
  if [[ -z "${SCRATCH_ROOT:-}" || -z "${SYSTEM_ROOT:-}" || -z "${GOLLIE_PYTHON312:-}" ]]; then
    die "SCRATCH_ROOT, SYSTEM_ROOT and GOLLIE_PYTHON312 are required"
  fi
  if [[ -z "${GOLLIE_WHEELHOUSE:-}" || -z "${GOLLIE_ISOLATED_VENV:-}" || -z "${GOLLIE_FLASH_ATTN_WHEEL:-}" ]]; then
    die "GOLLIE_WHEELHOUSE, GOLLIE_ISOLATED_VENV and GOLLIE_FLASH_ATTN_WHEEL are required"
  fi
  case "${GOLLIE_WHEELHOUSE}" in
    */gollie-wheelhouse-exp48|*/gollie-wheelhouse-exp48/*)
      die "refusing to modify the EXP.4.8 wheelhouse"
      ;;
    */gollie-wheelhouse-exp49) ;;
    *) die "GOLLIE_WHEELHOUSE must be gollie-wheelhouse-exp49" ;;
  esac
  if [[ "${GOLLIE_ISOLATED_VENV}" != "${SCRATCH_ROOT}/venvs/gollie-isolated" ]]; then
    die "the isolated venv must be ${SCRATCH_ROOT}/venvs/gollie-isolated"
  fi
  export GOLLIE_ORIGINAL_VENV="${GOLLIE_ORIGINAL_VENV:-${SCRATCH_ROOT}/venvs/gollie-l40s}"
  # shellcheck disable=SC1091
  source "${BOOTSTRAP}"
  resolve_gollie_creator
}

prepare_main() {
  require_common
  if [[ "${GOLLIE_DOWNLOAD_AUTHORIZED:-}" != "1" ]]; then
    die "wheel download is not authorized"
  fi
  if [[ -z "${GOLLIE_PIP_HTTP_CACHE:-}" || -z "${HF_HUB_CACHE:-}" ]]; then
    die "GOLLIE_PIP_HTTP_CACHE and HF_HUB_CACHE are required"
  fi
  bash "${BOOTSTRAP}" close
  bash "${BOOTSTRAP}" fetch-modeling
}

install_main() {
  require_common
  if [[ "${GOLLIE_ISOLATED_INSTALL:-}" != "1" ]]; then
    die "R1 install is not authorized"
  fi
  bash "${BOOTSTRAP}" install
}

submit_main() {
  require_common
  if [[ "${GOLLIE_R2_PREFLIGHT_AUTHORIZED:-}" != "1" ]]; then
    die "isolated J15 is not authorized"
  fi
  export GOLLIE_ORIGINAL_VENV="${SCRATCH_ROOT}/venvs/gollie-l40s"
  export GOLLIE_VENV="${GOLLIE_ISOLATED_VENV}"
  bash "${BOOTSTRAP}" assert-closed
  bash "${BOOTSTRAP}" assert-prep
  bash "${BOOTSTRAP}" assert-modeling
  "${GOLLIE_PYTHON312}" - <<'PY'
import json
import os
import sys
from pathlib import Path

record = json.loads((Path(os.environ["SCRATCH_ROOT"]) / "recovery" / "gollie-offline-resolve.json").read_text(encoding="utf-8"))
if record.get("offline_resolve_pass") is not True or record.get("wheelhouse") != os.environ["GOLLIE_WHEELHOUSE"]:
    print("FAIL: offline dry-run has not passed for this wheelhouse", file=sys.stderr)
    raise SystemExit(1)
PY
  local sbatch_file="${SYSTEM_ROOT}/jobs/cesvima/j15_gollie_isolated_preflight.sbatch"
  sbatch \
    --export=ALL,GOLLIE_R2_PREFLIGHT_AUTHORIZED=1,GOLLIE_FLASH_ATTN_WHEEL="${GOLLIE_FLASH_ATTN_WHEEL}" \
    "${sbatch_file}"
}

case "${1:-}" in
  prepare) prepare_main ;;
  install) install_main ;;
  submit) submit_main ;;
  *) die "usage: gollie_go.sh [prepare|install|submit]" ;;
esac
