#!/bin/bash -l
#SBATCH --job-name=p3-j17-gollie-smoke
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Manual smoke only. It stays closed until J16 records a complete checkpoint.
# Source env.sh, then point GOLLIE_VENV at gollie-isolated. Do not edit env.sh.

set -euo pipefail
saved_smoke="${GOLLIE_J17_SMOKE_AUTHORIZED:-}"
if [[ "${saved_smoke}" != "1" ]]; then
  echo "FAIL: J17 native smoke is not authorized" >&2
  exit 1
fi
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
export GOLLIE_J17_SMOKE_AUTHORIZED="${saved_smoke}"
unset GOLLIE_J17_TRACE_AUTHORIZED
original="${GOLLIE_VENV:-}"
if [[ -z "${original}" ]]; then
  echo "FAIL: env.sh did not set GOLLIE_VENV" >&2
  exit 1
fi
export GOLLIE_ORIGINAL_VENV="${original}"
export GOLLIE_ISOLATED_VENV="${SCRATCH_ROOT}/venvs/gollie-isolated"
if [[ "${GOLLIE_ISOLATED_VENV}" == "${GOLLIE_ORIGINAL_VENV}" ]]; then
  echo "FAIL: isolated venv collapses onto gollie-l40s" >&2
  exit 1
fi
export GOLLIE_VENV="${GOLLIE_ISOLATED_VENV}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
exec bash "${SYSTEM_ROOT}/scripts/cesvima/j17_gollie_native_smoke.sh"
