#!/bin/bash -l
#SBATCH --job-name=p3-j15-gollie-preflight
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
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
