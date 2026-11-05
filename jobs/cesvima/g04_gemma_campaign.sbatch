#!/bin/bash -l
# G04 refusal entrypoint. CPU only: this checkout must not reserve a GPU for Gold.
# The Python phase lock remains closed. Do not point this job at annotations.
#SBATCH --job-name=p3-g04-gemma-campaign
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
unset GEMMA_G04_RUN_AUTHORIZED
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/src:${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
output="${P3_RUNS_ROOT}/gemma4/g04/${SLURM_JOB_ID:-local}/refusal.json"
exec "${INFER_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/g04_gemma_campaign.py" \
  --output "${output}"
