#!/bin/bash -l
# G00 readiness. CPU only: do not reserve a GPU and do not download weights.
# The checker may use the existing inference interpreter. It must not install
# into that venv. The Gemma runtime, when present, is GEMMA_VENV.
#SBATCH --job-name=p3-g00-gemma-readiness
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
saved_g00="${GEMMA_G00_READINESS_AUTHORIZED:-}"
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
if [[ "${saved_g00}" == "1" ]]; then
  export GEMMA_G00_READINESS_AUTHORIZED=1
else
  unset GEMMA_G00_READINESS_AUTHORIZED
fi
unset GEMMA_G01_ACQUIRE_AUTHORIZED
unset GEMMA_G02_PREFLIGHT_AUTHORIZED
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/src:${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
output="${P3_RUNS_ROOT}/gemma4/g00/${SLURM_JOB_ID}/readiness.json"
exec "${INFER_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/g00_gemma_readiness.py" \
  --output "${output}"
