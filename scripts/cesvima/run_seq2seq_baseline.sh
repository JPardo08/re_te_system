#!/bin/bash
# Shared local-only inference for mREBEL and REBEL. Does not train.
set -euo pipefail

EXTRACTOR="${1:?extractor}"
SPLIT="${2:?smoke3 or full}"
REVISION="${3:?sha}"
OUTPUT_ROOT="${4:?output root}"

case "${EXTRACTOR}" in
  mrebel|rebel) ;;
  *)
    echo "FAIL: extractor ${EXTRACTOR} is not a seq2seq baseline job" >&2
    exit 1
    ;;
esac
case "${SPLIT}" in
  smoke3|full|cross-lingual) ;;
  *)
    echo "FAIL: unknown split ${SPLIT}" >&2
    exit 1
    ;;
esac
if [[ "${EXTRACTOR}" == "rebel" && "${SPLIT}" != "cross-lingual" ]]; then
  echo "FAIL: REBEL native English smoke is BLOCKED_FIXTURE" >&2
  echo "FAIL: Hohfeld Spanish is only available as split=cross-lingual" >&2
  exit 2
fi
if [[ "${EXTRACTOR}" != "rebel" && "${SPLIT}" == "cross-lingual" ]]; then
  echo "FAIL: cross-lingual split is only defined for REBEL" >&2
  exit 2
fi
# The historical relation allow-list stays off. This runner never passes it.

PYTHON="${INFER_VENV}/bin/python"
SCRIPT_DIR="${SYSTEM_ROOT}/scripts/cesvima"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"

if [[ "${SPLIT}" == "smoke3" ]]; then
  DOCUMENTS="${OUTPUT_ROOT}/documents.jsonl"
  "${PYTHON}" "${SCRIPT_DIR}/select_smoke3.py" "${DOCUMENTS}"
  EXPECTED=3
else
  DOCUMENTS="${CANONICAL_DOCUMENTS}"
  EXPECTED=36
fi

ARGS=(
  -m re_te_system.cli run
  --extractor "${EXTRACTOR}"
  --documents "${DOCUMENTS}"
  --benchmark-manifest "${CANONICAL_MANIFEST}"
  --output-root "${OUTPUT_ROOT}"
  --model-revision "${REVISION}"
  --local-files-only
  --device cuda
  --dtype float32
  --input-language es
  --max-input-tokens 256
  --num-beams 3
  --seed 42
  --window-strategy TOKEN
  --window-overlap 0
)
if [[ "${EXTRACTOR}" == "mrebel" ]]; then
  ARGS+=(
    --model-name Babelscape/mrebel-large
    --src-lang es_XX
    --target-token tp_XX
    --max-new-tokens 512
  )
else
  ARGS+=(
    --model-name Babelscape/rebel-large
    --max-length 512
  )
fi

RUN_JSON="$("${PYTHON}" "${ARGS[@]}")"
echo "${RUN_JSON}"
RUN_DIR="$(printf '%s' "${RUN_JSON}" | "${PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["run_dir"])')"
"${PYTHON}" "${SCRIPT_DIR}/pass_baseline_run.py" "${RUN_DIR}" "${EXPECTED}" "${REVISION}"
if [[ "${SPLIT}" == "cross-lingual" ]]; then
  "${PYTHON}" - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text(encoding="utf-8"))
model = manifest["model"]
if manifest.get("language_status") != "out_of_documented_language_scope":
    raise SystemExit(f"FAIL: language_status is {manifest.get('language_status')}")
if model.get("scientific_stratum") != "CROSS_LINGUAL_DIAGNOSTIC":
    raise SystemExit(f"FAIL: scientific_stratum is {model.get('scientific_stratum')}")
if model.get("principal_ranking") != "excluded":
    raise SystemExit("FAIL: cross-lingual REBEL must stay out of the principal ranking")
if manifest.get("input_language") != "es":
    raise SystemExit(f"FAIL: input_language is {manifest.get('input_language')}")
print("PASS REBEL cross-lingual diagnostic label", flush=True)
PY
fi
