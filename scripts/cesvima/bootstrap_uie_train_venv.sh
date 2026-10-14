#!/bin/bash
# Create the UIE training venv only when it is absent.
# An existing venv is reused unchanged. This script does not edit upstream code.
set -euo pipefail

if [[ -x "${UIE_TRAIN_VENV}/bin/python" ]]; then
  echo "PASS reuse ${UIE_TRAIN_VENV}"
  exit 0
fi
if [[ ! -x "${INFER_VENV}/bin/python" ]]; then
  echo "FAIL: inference venv is missing at ${INFER_VENV}" >&2
  exit 1
fi
echo "CREATE ${UIE_TRAIN_VENV}"
"${INFER_VENV}/bin/python" -m venv "${UIE_TRAIN_VENV}"
"${UIE_TRAIN_VENV}/bin/python" -m pip install --upgrade pip
"${UIE_TRAIN_VENV}/bin/python" -m pip install \
  "torch==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu121
"${UIE_TRAIN_VENV}/bin/python" -m pip install \
  "transformers==4.57.6" \
  "sentencepiece>=0.2,<0.3" \
  "numpy" \
  "datasets" \
  "pyyaml" \
  "tqdm"
echo "PASS created ${UIE_TRAIN_VENV}"
