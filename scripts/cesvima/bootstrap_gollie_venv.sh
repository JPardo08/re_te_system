#!/bin/bash
# Validate the existing gollie-l40s environment.
# Do not create the venv, do not pip install, and do not compile flash-attn.
# Exit 0 only when the installed extension is accredited as the pinned wheel.
# Exit 2 when the venv exists but identity or versions are not accredited.
set -euo pipefail

if [[ -n "${GOLLIE_BOOTSTRAP_ALLOW_COMPILE:-}" ]]; then
  echo "FAIL: FlashAttention compilation is forbidden" >&2
  exit 1
fi
if [[ -z "${GOLLIE_VENV:-}" || ! -x "${GOLLIE_VENV}/bin/python" ]]; then
  echo "FAIL: existing gollie-l40s is required; refusing to create a venv or compile" >&2
  exit 1
fi
if [[ -z "${SYSTEM_ROOT:-}" ]]; then
  echo "FAIL: SYSTEM_ROOT is required" >&2
  exit 1
fi
export PYTHONPATH="${SYSTEM_ROOT}/scripts/cesvima${PYTHONPATH:+:${PYTHONPATH}}"
set +e
"${GOLLIE_VENV}/bin/python" "${SYSTEM_ROOT}/scripts/cesvima/gollie_environment.py" --assess
status=$?
set -e
if [[ "${status}" -eq 0 || "${status}" -eq 2 ]]; then
  exit "${status}"
fi
echo "FAIL: gollie environment assessment exited ${status}" >&2
exit "${status}"
