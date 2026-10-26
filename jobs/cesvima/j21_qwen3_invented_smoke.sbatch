#!/bin/bash -l
# Qwen3 invented-text smoke on one L40S.
# Operational caps below are not a scientific decoding freeze.
# env.sh supplies shared CESVIMA paths. The code executed here is not SYSTEM_ROOT.
# Do not submit this file from the local Cursor session.
#SBATCH --job-name=p3-j21-qwen3-smoke
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -euo pipefail
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
cesvima_guard
cesvima_prepare_runtime_dirs

# Isolated tree on SCRATCH. It must already contain the a65fb07 archive plus the
# four J21 files. This job does not copy, edit, or submit against SYSTEM_ROOT.
J21_ROOT=/home/y064/SCRATCH/y064644/paper3/isolated/qwen3-j21/re_te_system
export J21_ROOT
if [[ "${J21_ROOT}" == "${SYSTEM_ROOT}" ]]; then
  echo "FAIL: J21_ROOT must not be the canonical SYSTEM_ROOT" >&2
  exit 3
fi
case "${J21_ROOT}" in
  "${PROJECT_ROOT}"*)
    echo "FAIL: J21_ROOT is inside PROJECT_ROOT" >&2
    exit 3
    ;;
esac
if [[ ! -f "${J21_ROOT}/scripts/cesvima/j21_qwen3_invented_smoke.py" ]]; then
  echo "FAIL: isolated J21 script is missing: ${J21_ROOT}" >&2
  exit 3
fi
if [[ ! -f "${J21_ROOT}/src/re_te_system/controlled_experiment/qwen3.py" ]]; then
  echo "FAIL: isolated qwen3.py is missing: ${J21_ROOT}" >&2
  exit 3
fi

cd "${P3_SLURM_LOG_DIR}"
# Replace any login-shell PYTHONPATH. The canonical editable install stays in
# the venv for J00-J20; this process must not import that tree.
export PYTHONPATH="${J21_ROOT}/src:${J21_ROOT}/scripts/cesvima"
export RE_TE_BENCHMARK_SRC="${BENCHMARK_ROOT}/src"
export RE_TE_EVALUATION_SRC="${EVALUATION_ROOT}/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${INFER_VENV}/bin/python" - <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["J21_ROOT"]).resolve()
canonical = Path(os.environ["SYSTEM_ROOT"]).resolve()
if root == canonical or canonical in root.parents:
    sys.stderr.write("FAIL: isolated root resolves to the canonical checkout\n")
    raise SystemExit(3)
import re_te_system
imported = Path(re_te_system.__file__).resolve()
expected = root / "src" / "re_te_system" / "__init__.py"
if imported != expected:
    sys.stderr.write(f"FAIL: re_te_system imported from {imported}, expected {expected}\n")
    raise SystemExit(3)
from re_te_system.controlled_experiment.qwen3 import (
    QWEN3_WEIGHT_EXECUTION_PHASE_LOCK,
    Qwen3InventedSmokePermit,
)
if QWEN3_WEIGHT_EXECUTION_PHASE_LOCK is not True:
    sys.stderr.write("FAIL: phase lock is open\n")
    raise SystemExit(3)
if Qwen3InventedSmokePermit.__name__ != "Qwen3InventedSmokePermit":
    sys.stderr.write("FAIL: invented smoke permit is not the isolated class\n")
    raise SystemExit(3)
print(f"PASS isolated-import {imported}", flush=True)
PY

OUTPUT="${P3_RUNS_ROOT}/qwen3/invented-smoke/${SLURM_JOB_ID}"
exec "${INFER_VENV}/bin/python" "${J21_ROOT}/scripts/cesvima/j21_qwen3_invented_smoke.py" \
  --execute \
  --output-dir "${OUTPUT}"
