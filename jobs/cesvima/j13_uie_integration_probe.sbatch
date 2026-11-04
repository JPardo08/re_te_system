#!/bin/bash -l
#SBATCH --job-name=p3-j13-uie-integration
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
saved_integration="${UIE_J13_INTEGRATION_PROBE_AUTHORIZED:-}"
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
if [[ -n "${saved_integration}" ]]; then
  export UIE_J13_INTEGRATION_PROBE_AUTHORIZED="${saved_integration}"
else
  unset UIE_J13_INTEGRATION_PROBE_AUTHORIZED
fi
unset UIE_J13_RUN_AUTHORIZED
unset UIE_J13_LAUNCHED_BY_GATE
unset UIE_J13_GPU_PROBE_AUTHORIZED
unset UIE_J13_INTEGRATION_CHILD
unset UIE_J12_DERIVED_LOAD_AUTHORIZED
unset UIE_SAFETENSORS_CONVERT_AUTHORIZED
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${P3_UPSTREAM_ROOT}/UIE:${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
bash "${SYSTEM_ROOT}/scripts/cesvima/bootstrap_uie_train_venv.sh"
exec "${UIE_TRAIN_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/j13_uie_integration_probe.py"
