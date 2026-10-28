#!/bin/bash
# J17: local-only GoLLIE native smoke. No fine-tuning and no quantization.
set -euo pipefail

export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima:${SYSTEM_ROOT}/src"
if [[ "${GOLLIE_J17_SMOKE_AUTHORIZED:-}" != "1" ]]; then
  echo "FAIL: J17 native smoke is not authorized" >&2
  exit 1
fi
PYTHON="${GOLLIE_VENV}/bin/python"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "FAIL: J17 requires SLURM_JOB_ID" >&2
  exit 1
fi
OUTPUT_ROOT="$("${PYTHON}" - <<'PY'
import os
from pathlib import Path

from j17_job_paths import j17_job_directory

print(j17_job_directory(Path(os.environ["P3_RUNS_ROOT"]), os.environ["SLURM_JOB_ID"]))
PY
)"
mkdir -p "${OUTPUT_ROOT}"
DERIVED="$("${PYTHON}" - <<'PY'
import os
from pathlib import Path

from gollie_safetensors_conversion import assert_j17_conversion
from j16_gollie_download import assert_j17_prerequisites

runs = Path(os.environ["P3_RUNS_ROOT"])
assert_j17_prerequisites(runs)
print(assert_j17_conversion(runs)["derived_path"])
PY
)"
CLI_LOG="${OUTPUT_ROOT}/cli-stdout.txt"
set +e
"${PYTHON}" -m re_te_system.cli run \
  --extractor gollie \
  --documents "${SYSTEM_ROOT}/tests/fixtures/gollie/official_re_documents.jsonl" \
  --gollie-schema-file "${SYSTEM_ROOT}/tests/fixtures/gollie/official_re_schema.json" \
  --output-root "${OUTPUT_ROOT}" \
  --model-name HiTZ/GoLLIE-7B \
  --model-revision d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9 \
  --gollie-weight-snapshot "${DERIVED}" \
  --local-files-only \
  --device cuda \
  --dtype bfloat16 \
  --input-language en \
  --gollie-max-new-tokens 128 \
  --seed 42 \
  --window-strategy TOKEN \
  --window-overlap 0 > "${CLI_LOG}"
cli_status=$?
set -e
cat "${CLI_LOG}"
if [[ "${cli_status}" -ne 0 ]]; then
  echo "FAIL: GoLLIE CLI exited ${cli_status}" >&2
  exit "${cli_status}"
fi
RUN_DIR="$("${PYTHON}" "${SYSTEM_ROOT}/scripts/cesvima/terminal_run_json.py" "${CLI_LOG}")"
"${PYTHON}" "${SYSTEM_ROOT}/scripts/cesvima/pass_baseline_run.py" \
  "${RUN_DIR}" 1 d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9
