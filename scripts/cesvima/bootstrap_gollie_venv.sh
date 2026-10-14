#!/bin/bash
# Create the GoLLIE L40S venv only when it is absent.
# flash-attn is installed here. GoLLIE-7B weights are not downloaded here.
set -euo pipefail

if [[ -x "${GOLLIE_VENV}/bin/python" ]]; then
  echo "PASS reuse ${GOLLIE_VENV}"
  exit 0
fi
if [[ ! -x "${INFER_VENV}/bin/python" ]]; then
  echo "FAIL: inference venv is missing at ${INFER_VENV}" >&2
  exit 1
fi
echo "CREATE ${GOLLIE_VENV}"
"${INFER_VENV}/bin/python" -m venv "${GOLLIE_VENV}"
"${GOLLIE_VENV}/bin/python" -m pip install --upgrade pip
"${GOLLIE_VENV}/bin/python" -m pip install \
  "torch==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu121
"${GOLLIE_VENV}/bin/python" -m pip install \
  "transformers==4.57.6" \
  "black>=24.8,<25" \
  "sentencepiece>=0.2,<0.3" \
  "huggingface_hub" \
  "packaging"
"${GOLLIE_VENV}/bin/python" -m pip install "flash-attn" --no-build-isolation
echo "PASS created ${GOLLIE_VENV}"
