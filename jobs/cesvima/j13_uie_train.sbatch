#!/bin/bash -l
#SBATCH --job-name=p3-j13-uie-train
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
bash "${SYSTEM_ROOT}/scripts/cesvima/bootstrap_uie_train_venv.sh"
exec "${UIE_TRAIN_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/j13_uie_train.py"
