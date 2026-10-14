#!/bin/bash -l
#SBATCH --job-name=p3-j02-mrebel-smoke3
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
exec bash "${SYSTEM_ROOT}/scripts/cesvima/run_seq2seq_baseline.sh" mrebel smoke3 50e0587ac7ac87d28b9abd069d72333528a5aa09 "${P3_RUNS_ROOT}/mrebel/smoke3"
