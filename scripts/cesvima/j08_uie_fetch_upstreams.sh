#!/bin/bash
# J08: clone pinned UIE and sincere trees onto SCRATCH. Never copies ignored local trees.
set -euo pipefail

UIE_URL=https://github.com/universal-ie/UIE.git
UIE_REV=c88dd08b8ad0016ec597ebc6f8fff49480367bb2
SINCERE_URL=https://github.com/btaille/sincere.git
SINCERE_REV=dd1c34916ddcdc5ceb2799d64b17e80cdf1a5b31

export GIT_TERMINAL_PROMPT=0

fetch_exact() {
  local url="$1"
  local revision="$2"
  local dest="$3"
  if [[ -d "${dest}/.git" ]]; then
    local head
    head="$(git -C "${dest}" rev-parse HEAD)"
    if [[ "${head}" != "${revision}" ]]; then
      echo "FAIL: ${dest} HEAD ${head} != ${revision}" >&2
      exit 1
    fi
    echo "PASS reuse ${dest} @ ${revision}"
    return 0
  fi
  if [[ -e "${dest}" ]]; then
    echo "FAIL: ${dest} exists and is not a git checkout of ${revision}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dest}")"
  git init "${dest}"
  git -C "${dest}" remote add origin "${url}"
  git -C "${dest}" fetch --depth 1 origin "${revision}"
  git -C "${dest}" checkout --detach FETCH_HEAD
  local head
  head="$(git -C "${dest}" rev-parse HEAD)"
  if [[ "${head}" != "${revision}" ]]; then
    echo "FAIL: fetched HEAD ${head} != ${revision}" >&2
    exit 1
  fi
  echo "PASS clone ${dest} @ ${revision}"
}

fetch_exact "${UIE_URL}" "${UIE_REV}" "${P3_UPSTREAM_ROOT}/UIE"
fetch_exact "${SINCERE_URL}" "${SINCERE_REV}" "${P3_UPSTREAM_ROOT}/sincere"
printf '%s\n' "${UIE_REV}" > "${P3_UPSTREAM_ROOT}/UIE.commit"
printf '%s\n' "${SINCERE_REV}" > "${P3_UPSTREAM_ROOT}/sincere.commit"
