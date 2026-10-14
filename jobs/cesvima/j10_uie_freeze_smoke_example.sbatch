#!/bin/bash -l
#SBATCH --job-name=p3-j10-uie-freeze-example
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
exec "${INFER_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/j10_uie_freeze_smoke_example.py"
