#!/bin/bash -l
#SBATCH --job-name=p3-j15-gollie-isolated
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Manual R2 only. submit_train_first.sh does not submit this file.
# Source env.sh first so its local adaptations stay in effect, then point
# GOLLIE_VENV at the isolated environment. Do not edit env.sh.
# Do not install packages and do not authorize J16 or J17.

set -euo pipefail
if [[ "${GOLLIE_R2_PREFLIGHT_AUTHORIZED:-}" != "1" ]]; then
  echo "FAIL: isolated J15 is not authorized" >&2
  exit 1
fi
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
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
unset GOLLIE_J17_SMOKE_AUTHORIZED
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
export GOLLIE_PYTHON312="${GOLLIE_VENV}/bin/python"
bash "${SYSTEM_ROOT}/scripts/cesvima/bootstrap_gollie_isolated_venv.sh" assert-prep
bash "${SYSTEM_ROOT}/scripts/cesvima/bootstrap_gollie_isolated_venv.sh" assert-modeling
set +e
bash "${SYSTEM_ROOT}/scripts/cesvima/bootstrap_gollie_venv.sh"
bootstrap_status=$?
set -e
if [[ ! -x "${GOLLIE_VENV}/bin/python" ]]; then
  exit "${bootstrap_status}"
fi
if [[ "${bootstrap_status}" -ne 0 && "${bootstrap_status}" -ne 2 ]]; then
  exit "${bootstrap_status}"
fi
exec "${GOLLIE_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/j15_gollie_preflight.py"
