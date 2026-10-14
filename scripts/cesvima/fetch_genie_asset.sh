#!/bin/bash
# Download and verify one GenIE asset. No checkpoint is loaded.
set -euo pipefail

WHICH="${1:?ckpt or tries}"
case "${WHICH}" in
  ckpt)
    URL="https://zenodo.org/record/6139236/files/genie_r.ckpt"
    NAME="genie_r.ckpt"
    EXPECTED_SIZE=4879373206
    EXPECTED_MD5=c214da56b6e5d5bd259e0cbe826d92f7
    ;;
  tries)
    URL="https://zenodo.org/record/6139236/files/tries.zip"
    NAME="tries.zip"
    EXPECTED_SIZE=94255261
    EXPECTED_MD5=45db4dc2578cdeda891d8ed6ddb3f89f
    ;;
  *)
    echo "FAIL: unknown GenIE asset ${WHICH}" >&2
    exit 1
    ;;
esac

DEST="${P3_MODELS_ROOT}/genie/${NAME}"
mkdir -p "$(dirname "${DEST}")"
verify() {
  local size md5
  size="$(wc -c < "${DEST}" | tr -d ' ')"
  md5="$(md5sum "${DEST}" | awk '{print $1}')"
  if [[ "${size}" != "${EXPECTED_SIZE}" || "${md5}" != "${EXPECTED_MD5}" ]]; then
    echo "FAIL: ${NAME} size=${size} md5=${md5}" >&2
    return 1
  fi
}
if [[ -f "${DEST}" ]] && verify; then
  echo "PASS reuse ${DEST}"
  exit 0
fi
rm -f "${DEST}"
curl --fail --location --retry 3 --output "${DEST}" "${URL}"
verify
echo "PASS downloaded ${DEST}"
