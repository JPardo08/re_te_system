#!/bin/bash -l
#SBATCH --job-name=p3-gollie-safetensors
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# CPU conversion of the indexed GoLLIE shards. Do not use a GPU runtime.
# Source env.sh, then run the operator-supplied Python. Do not edit env.sh.

set -euo pipefail
saved_auth="${GOLLIE_SAFETENSORS_CONVERT_AUTHORIZED:-}"
saved_python="${GOLLIE_CONVERT_PYTHON:-}"
if [[ "${saved_auth}" != "1" ]]; then
  echo "FAIL: GoLLIE safetensors conversion is not authorized" >&2
  exit 1
fi
if [[ -z "${saved_python}" || ! -x "${saved_python}" ]]; then
  echo "FAIL: GOLLIE_CONVERT_PYTHON must be an executable Python >= 2.6 outside the GPU runtimes" >&2
  exit 1
fi
case "${saved_python}" in
  *gollie-isolated*|*gollie-l40s*)
    echo "FAIL: refusing to convert bins with the GPU runtime" >&2
    exit 1
    ;;
esac
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
export GOLLIE_SAFETENSORS_CONVERT_AUTHORIZED="${saved_auth}"
export GOLLIE_CONVERT_PYTHON="${saved_python}"
unset HF_HUB_OFFLINE
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/src:${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
exec "${GOLLIE_CONVERT_PYTHON}" "${SYSTEM_ROOT}/scripts/cesvima/gollie_safetensors_conversion.py"
