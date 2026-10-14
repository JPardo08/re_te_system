#!/bin/bash -l
#SBATCH --job-name=p3-j09-uie-rebuild
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
exec "${INFER_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/j09_uie_rebuild_data.py"
