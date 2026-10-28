#!/bin/bash -l
#SBATCH --job-name=p3-j16-gollie-download
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
# Manual download only. Do not submit the train-first chain.
# Source env.sh, then point GOLLIE_VENV at gollie-isolated. Do not edit env.sh.

set -euo pipefail
saved_download="${GOLLIE_J16_DOWNLOAD_AUTHORIZED:-}"
saved_wheel="${GOLLIE_FLASH_ATTN_WHEEL:-}"
if [[ "${saved_download}" != "1" ]]; then
  echo "FAIL: J16 weight download is not authorized" >&2
  exit 1
fi
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
export GOLLIE_J16_DOWNLOAD_AUTHORIZED="${saved_download}"
export GOLLIE_FLASH_ATTN_WHEEL="${saved_wheel}"
original="${GOLLIE_VENV:-}"
if [[ -z "${original}" ]]; then
  echo "FAIL: env.sh did not set GOLLIE_VENV" >&2
  exit 1
fi
if [[ -z "${GOLLIE_FLASH_ATTN_WHEEL:-}" ]]; then
  echo "FAIL: GOLLIE_FLASH_ATTN_WHEEL is required; its CESVIMA path was not assumed" >&2
  exit 1
fi
export GOLLIE_ORIGINAL_VENV="${original}"
export GOLLIE_ISOLATED_VENV="${SCRATCH_ROOT}/venvs/gollie-isolated"
if [[ "${GOLLIE_ISOLATED_VENV}" == "${GOLLIE_ORIGINAL_VENV}" ]]; then
  echo "FAIL: isolated venv collapses onto gollie-l40s" >&2
  exit 1
fi
export GOLLIE_VENV="${GOLLIE_ISOLATED_VENV}"
unset HF_HUB_OFFLINE
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
bash "${SYSTEM_ROOT}/scripts/cesvima/bootstrap_gollie_venv.sh"
exec "${GOLLIE_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/j16_gollie_download.py"
