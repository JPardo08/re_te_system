#!/bin/bash
# J17: local-only GoLLIE native smoke. No fine-tuning and no quantization.
set -euo pipefail

export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima:${SYSTEM_ROOT}/src"
if [[ "${GOLLIE_J17_SMOKE_AUTHORIZED:-}" != "1" ]]; then
  echo "FAIL: J17 native smoke is not authorized" >&2
  exit 1
fi
PYTHON="${GOLLIE_VENV}/bin/python"
OUTPUT_ROOT="${P3_RUNS_ROOT}/gollie/native-smoke"
mkdir -p "${OUTPUT_ROOT}"
RUN_JSON="$("${PYTHON}" -m re_te_system.cli run \
  --extractor gollie \
  --documents "${SYSTEM_ROOT}/tests/fixtures/gollie/official_re_documents.jsonl" \
  --gollie-schema-file "${SYSTEM_ROOT}/tests/fixtures/gollie/official_re_schema.json" \
  --output-root "${OUTPUT_ROOT}" \
  --model-name HiTZ/GoLLIE-7B \
  --model-revision d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9 \
  --local-files-only \
  --device cuda \
  --dtype bfloat16 \
  --input-language en \
  --gollie-max-new-tokens 128 \
  --seed 42 \
  --window-strategy TOKEN \
  --window-overlap 0)"
echo "${RUN_JSON}"
RUN_DIR="$(printf '%s' "${RUN_JSON}" | "${PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["run_dir"])')"
"${PYTHON}" "${SYSTEM_ROOT}/scripts/cesvima/pass_baseline_run.py" \
  "${RUN_DIR}" 1 d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9
