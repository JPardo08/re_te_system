#!/bin/bash -l
#SBATCH --job-name=p3-uie-safetensors
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=uie-safetensors-%j.out
#SBATCH --error=uie-safetensors-%j.err
# CPU conversion of the pinned UIE pytorch_model.bin. Do not use uie-train.
# A CUDA build of the interpreter is not a GPU request: this job asks for none.
# Source env.sh, then run the operator-supplied Python. Do not edit env.sh.
# This file is not selected by submit_train_first.sh --all.

set -euo pipefail
saved_auth="${UIE_SAFETENSORS_CONVERT_AUTHORIZED:-}"
saved_python="${UIE_CONVERT_PYTHON:-}"
if [[ "${saved_auth}" != "1" ]]; then
  echo "FAIL: UIE safetensors conversion is not authorized" >&2
  exit 1
fi
if [[ -z "${saved_python}" || ! -x "${saved_python}" ]]; then
  echo "FAIL: UIE_CONVERT_PYTHON must be an executable Python >= 2.6 outside uie-train" >&2
  exit 1
fi
case "${saved_python}" in
  *uie-train*)
    echo "FAIL: refusing to convert bins with uie-train" >&2
    exit 1
    ;;
esac
source /home/y064/PROJECT/y064644/paper3/re_te_system/scripts/cesvima/env.sh
export UIE_SAFETENSORS_CONVERT_AUTHORIZED="${saved_auth}"
export UIE_CONVERT_PYTHON="${saved_python}"
unset UIE_J12_DERIVED_LOAD_AUTHORIZED
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
cesvima_guard
cesvima_prepare_runtime_dirs
cd "${P3_SLURM_LOG_DIR}"
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
exec "${UIE_CONVERT_PYTHON}" "${SYSTEM_ROOT}/scripts/cesvima/uie_safetensors_conversion.py"
