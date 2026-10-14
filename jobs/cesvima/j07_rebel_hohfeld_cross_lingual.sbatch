#!/bin/bash -l
#SBATCH --job-name=p3-j07-rebel-cross
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
REVISION="$("${INFER_VENV}/bin/python" -c 'import json,os; from pathlib import Path; print(json.loads(Path(os.environ["P3_RUNS_ROOT"], "rebel", "revision.json").read_text())["sha"])')"
exec bash "${SYSTEM_ROOT}/scripts/cesvima/run_seq2seq_baseline.sh" rebel cross-lingual "${REVISION}" "${P3_RUNS_ROOT}/rebel/hohfeld-cross-lingual"
