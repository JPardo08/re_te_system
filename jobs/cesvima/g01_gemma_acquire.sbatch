#!/bin/bash -l
# G01 acquisition. CPU only. The identity pin is
# 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7. That pin does not download by
# itself: GEMMA_G01_ACQUIRE_AUTHORIZED=1 is still required, and this job
# unsets the G02 grant. Any other revision is refused.
#SBATCH --job-name=p3-g01-gemma-acquire
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail
saved_g01="${GEMMA_G01_ACQUIRE_AUTHORIZED:-}"
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
if [[ "${saved_g01}" == "1" ]]; then
  export GEMMA_G01_ACQUIRE_AUTHORIZED=1
else
  unset GEMMA_G01_ACQUIRE_AUTHORIZED
fi
unset GEMMA_G02_PREFLIGHT_AUTHORIZED
revision="${GEMMA_REQUESTED_REVISION:?GEMMA_REQUESTED_REVISION is required and is not implied by the catalog}"
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/src:${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
destination="${GEMMA_MODEL_DIR}/${revision}"
report="${P3_RUNS_ROOT}/gemma4/g01/${SLURM_JOB_ID}/acquire.json"
exec "${INFER_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/g01_gemma_acquire.py" \
  --model-id google/gemma-4-12B-it \
  --revision "${revision}" \
  --destination "${destination}" \
  --report "${report}"
