#!/bin/bash -l
#SBATCH --job-name=p3-j16-gollie-download
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
bash "${SYSTEM_ROOT}/scripts/cesvima/bootstrap_gollie_venv.sh"
exec "${GOLLIE_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/j16_gollie_download.py"
