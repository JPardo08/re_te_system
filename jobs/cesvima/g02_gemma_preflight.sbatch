#!/bin/bash -l
# G02 preflight on one explicit L40S. Do not submit until G00 reports an
# isolated interpreter that imports Gemma4UnifiedForConditionalGeneration.
# This file does not authorize the run and does not download weights.
#SBATCH --job-name=p3-g02-gemma-preflight
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
saved_g02="${GEMMA_G02_PREFLIGHT_AUTHORIZED:-}"
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
if [[ "${saved_g02}" == "1" ]]; then
  export GEMMA_G02_PREFLIGHT_AUTHORIZED=1
else
  unset GEMMA_G02_PREFLIGHT_AUTHORIZED
fi
unset GEMMA_G01_ACQUIRE_AUTHORIZED
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
revision="${GEMMA_REQUESTED_REVISION:?GEMMA_REQUESTED_REVISION is required and is not implied by the catalog}"
snapshot="${GEMMA_SNAPSHOT_DIR:?GEMMA_SNAPSHOT_DIR is required}"
if [[ ! -x "${GEMMA_VENV}/bin/python" ]]; then
  echo "FAIL: isolated Gemma interpreter is missing: ${GEMMA_VENV}/bin/python" >&2
  exit 2
fi
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/src:${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
output="${P3_RUNS_ROOT}/gemma4/g02/${SLURM_JOB_ID}/preflight.json"
exec "${GEMMA_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/g02_gemma_preflight.py" \
  --execute \
  --revision "${revision}" \
  --snapshot-dir "${snapshot}" \
  --output "${output}"
