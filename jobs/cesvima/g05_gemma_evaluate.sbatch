#!/bin/bash -l
# G05 local scoring entrypoint. It has no default Gold path and does not train.
# Supply synthetic or already-exported files explicitly. Do not pass annotations.jsonl.
#SBATCH --job-name=p3-g05-gemma-evaluate
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
gold="${GEMMA_G05_GOLD:?GEMMA_G05_GOLD is required and must not be inferred}"
predictions="${GEMMA_G05_PREDICTIONS:?GEMMA_G05_PREDICTIONS is required}"
condition="${GEMMA_G05_CONDITION:?GEMMA_G05_CONDITION is required}"
case "${gold}" in
  *annotations.jsonl|*teresia_hohfeld*)
    echo "FAIL: G05 refuses the Hohfeld Gold path in this checkout" >&2
    exit 2
    ;;
esac
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/src:${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
output="${P3_RUNS_ROOT}/gemma4/g05/${SLURM_JOB_ID}/table.json"
exec "${INFER_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/g05_gemma_evaluate.py" \
  --gold "${gold}" \
  --predictions "${predictions}" \
  --condition "${condition}" \
  --output "${output}"
