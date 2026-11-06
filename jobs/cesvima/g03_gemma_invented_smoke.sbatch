#!/bin/bash -l
# G03 invented C0/C1 smoke on one explicit L40S.
# Do not submit until a revision is approved and G02 has been accredited.
# A technical pass does not authorize G04. This file does not set that grant.
#SBATCH --job-name=p3-g03-gemma-smoke
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
saved_g03="${GEMMA_G03_SMOKE_AUTHORIZED:-}"
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
if [[ "${saved_g03}" == "1" ]]; then
  export GEMMA_G03_SMOKE_AUTHORIZED=1
else
  unset GEMMA_G03_SMOKE_AUTHORIZED
fi
unset GEMMA_G04_RUN_AUTHORIZED
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
revision="${GEMMA_REQUESTED_REVISION:?GEMMA_REQUESTED_REVISION is required and is not implied by the catalog}"
snapshot="${GEMMA_SNAPSHOT_DIR:?GEMMA_SNAPSHOT_DIR is required and must be the accepted local snapshot}"
if [[ ! -x "${GEMMA_VENV}/bin/python" ]]; then
  echo "FAIL: isolated Gemma interpreter is missing: ${GEMMA_VENV}/bin/python" >&2
  exit 2
fi
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/src:${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
output_dir="${P3_RUNS_ROOT}/gemma4/g03/${SLURM_JOB_ID}"
exec "${GEMMA_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/g03_gemma_invented_smoke.py" \
  --execute \
  --revision "${revision}" \
  --snapshot-dir "${snapshot}" \
  --output-dir "${output_dir}"
