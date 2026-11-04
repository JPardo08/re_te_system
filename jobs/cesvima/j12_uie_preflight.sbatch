#!/bin/bash -l
#SBATCH --job-name=p3-j12-uie-preflight
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
saved_load="${UIE_J12_DERIVED_LOAD_AUTHORIZED:-}"
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
if [[ -n "${saved_load}" ]]; then
  export UIE_J12_DERIVED_LOAD_AUTHORIZED="${saved_load}"
else
  unset UIE_J12_DERIVED_LOAD_AUTHORIZED
fi
unset UIE_SAFETENSORS_CONVERT_AUTHORIZED
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
bash "${SYSTEM_ROOT}/scripts/cesvima/bootstrap_uie_train_venv.sh"
exec "${UIE_TRAIN_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/j12_uie_preflight.py"
