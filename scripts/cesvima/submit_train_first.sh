#!/bin/bash
# Submit the Paper 3 train-first DAG. Default: print usage and submit nothing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_SYSTEM="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

DRY_RUN=0
DO_MREBEL=0
DO_REBEL=0
DO_REBEL_CROSS=0
DO_UIE=0
DO_GOLLIE=0
DO_GENIE=0

usage() {
  echo "Usage: submit_train_first.sh [--dry-run] (--mrebel | --rebel | --rebel-cross-lingual | --uie | --gollie | --genie-assets | --all)" >&2
  echo "No branch is submitted unless one of those options is present." >&2
  echo "--all submits READY native-reproduction and Spanish-main jobs." >&2
  echo "It does not submit J06 BLOCKED_FIXTURE, J07, Pythia, GenIE inference, or Priority B." >&2
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --mrebel) DO_MREBEL=1 ;;
    --rebel) DO_REBEL=1 ;;
    --rebel-cross-lingual)
      DO_REBEL=1
      DO_REBEL_CROSS=1
      ;;
    --uie) DO_UIE=1 ;;
    --gollie) DO_GOLLIE=1 ;;
    --genie-assets) DO_GENIE=1 ;;
    --all)
      DO_MREBEL=1
      DO_REBEL=1
      DO_UIE=1
      DO_GOLLIE=1
      DO_GENIE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "FAIL: unknown option $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ "${DO_MREBEL}${DO_REBEL}${DO_UIE}${DO_GOLLIE}${DO_GENIE}" == "00000" ]]; then
  usage
  exit 2
fi

if [[ "${DRY_RUN}" -eq 0 && "${REPO_SYSTEM}" != "${SYSTEM_ROOT}" ]]; then
  echo "FAIL: real submission must run from ${SYSTEM_ROOT}" >&2
  exit 1
fi

for required in \
  scripts/cesvima/env.sh \
  scripts/cesvima/cesvima_common.py \
  scripts/cesvima/j00_verify_infer_env.py \
  scripts/cesvima/j01_mrebel_cache.py \
  scripts/cesvima/run_seq2seq_baseline.sh \
  scripts/cesvima/j04_rebel_resolve_revision.py \
  scripts/cesvima/j05_rebel_download.py \
  scripts/cesvima/j06_rebel_native_smoke.py \
  scripts/cesvima/j08_uie_fetch_upstreams.sh \
  scripts/cesvima/j09_uie_rebuild_data.py \
  scripts/cesvima/j10_uie_freeze_smoke_example.py \
  scripts/cesvima/j11_uie_cache_base.py \
  scripts/cesvima/j12_uie_preflight.py \
  scripts/cesvima/j13_uie_train.py \
  scripts/cesvima/j14_uie_positive_smoke.py \
  scripts/cesvima/bootstrap_uie_train_venv.sh \
  scripts/cesvima/j15_gollie_preflight.py \
  scripts/cesvima/j16_gollie_download.py \
  scripts/cesvima/j17_gollie_native_smoke.sh \
  scripts/cesvima/bootstrap_gollie_venv.sh \
  scripts/cesvima/fetch_genie_asset.sh \
  scripts/cesvima/j19_genie_tokenizer.py
do
  if [[ ! -f "${REPO_SYSTEM}/${required}" ]]; then
    echo "FAIL: missing required file ${REPO_SYSTEM}/${required}" >&2
    exit 1
  fi
done

require_file() {
  if [[ ! -f "${REPO_SYSTEM}/$1" ]]; then
    echo "FAIL: missing required file ${REPO_SYSTEM}/$1" >&2
    exit 1
  fi
  if printf '%s\n' "$(grep '^#SBATCH' "${REPO_SYSTEM}/$1" || true)" | grep -Eq 'A100|a100|gpu:1([^0-9]|$)'; then
    echo "FAIL: ${1} requests a forbidden GPU" >&2
    exit 1
  fi
}

jobid_set() {
  eval "JOBID_$1=\$2"
}

jobid_get() {
  eval "printf '%s' \"\${JOBID_$1:-}\""
}

submit_job() {
  local key="$1"
  local relative="$2"
  shift 2
  require_file "${relative}"
  local dep_arg=""
  local dep_key dep_id joined=""
  for dep_key in "$@"; do
    dep_id="$(jobid_get "${dep_key}")"
    if [[ -z "${dep_id}" ]]; then
      echo "FAIL: dependency ${dep_key} has no job id" >&2
      exit 1
    fi
    if [[ -n "${joined}" ]]; then
      joined="${joined}:${dep_id}"
    else
      joined="${dep_id}"
    fi
  done
  if [[ -n "${joined}" ]]; then
    dep_arg="--dependency=afterok:${joined}"
  fi
  local script="${SYSTEM_ROOT}/${relative}"
  local cmd=(
    sbatch
    --parsable
    --chdir "${P3_SLURM_LOG_DIR}"
    --output "${P3_SLURM_LOG_DIR}/%x-%j.out"
    --error "${P3_SLURM_LOG_DIR}/%x-%j.err"
  )
  if [[ -n "${dep_arg}" ]]; then
    cmd+=("${dep_arg}")
  fi
  cmd+=("${script}")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'DRY-RUN'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    jobid_set "${key}" "DRY_${key}"
    return 0
  fi
  local job_id
  job_id="$("${cmd[@]}")"
  jobid_set "${key}" "${job_id}"
  echo "SUBMITTED ${key} ${job_id}"
}

echo "DAG"
if [[ "${DO_MREBEL}" -eq 1 ]]; then
  echo "  J00 -> J01 -> J02 -> J03"
fi
if [[ "${DO_REBEL}" -eq 1 ]]; then
  echo "  J00 -> J04 -> J05"
  echo "  J06 native English smoke: BLOCKED_FIXTURE (not submitted)"
fi
if [[ "${DO_REBEL_CROSS}" -eq 1 ]]; then
  echo "  J05 -> J07 CROSS_LINGUAL_DIAGNOSTIC (excluded from the principal ranking)"
fi
if [[ "${DO_UIE}" -eq 1 ]]; then
  echo "  J00 -> J11"
  echo "  J00 + J08 -> J09 -> J10"
  echo "  J09 + J11 -> J12 -> J13"
  echo "  J13 + J10 -> J14"
fi
if [[ "${DO_GOLLIE}" -eq 1 ]]; then
  echo "  J15 -> J16 -> J17"
fi
if [[ "${DO_GENIE}" -eq 1 ]]; then
  echo "  J18"
  echo "  J00 -> J19"
  echo "  J20"
  echo "  GenIE assets: TARGET / BLOCKED_LOADER (no inference job)"
fi
echo "  Pythia: TARGET / BLOCKED_LICENSE"
echo "  Priority B: STANDBY"
echo "  C0/C1: not scheduled"

NEED_J00=0
if [[ "${DO_MREBEL}" -eq 1 || "${DO_REBEL}" -eq 1 || "${DO_UIE}" -eq 1 || "${DO_GENIE}" -eq 1 ]]; then
  NEED_J00=1
fi

SELECTED_JOBS=()
if [[ "${NEED_J00}" -eq 1 ]]; then
  SELECTED_JOBS+=("jobs/cesvima/j00_verify_infer_env.sbatch")
fi
if [[ "${DO_MREBEL}" -eq 1 ]]; then
  SELECTED_JOBS+=(
    "jobs/cesvima/j01_mrebel_cache_verify.sbatch"
    "jobs/cesvima/j02_mrebel_smoke3.sbatch"
    "jobs/cesvima/j03_mrebel_full.sbatch"
  )
fi
if [[ "${DO_REBEL}" -eq 1 ]]; then
  SELECTED_JOBS+=(
    "jobs/cesvima/j04_rebel_resolve_revision.sbatch"
    "jobs/cesvima/j05_rebel_download.sbatch"
  )
fi
if [[ "${DO_REBEL_CROSS}" -eq 1 ]]; then
  SELECTED_JOBS+=("jobs/cesvima/j07_rebel_hohfeld_cross_lingual.sbatch")
fi
if [[ "${DO_UIE}" -eq 1 ]]; then
  SELECTED_JOBS+=(
    "jobs/cesvima/j08_uie_fetch_upstreams.sbatch"
    "jobs/cesvima/j09_uie_rebuild_data.sbatch"
    "jobs/cesvima/j10_uie_freeze_smoke_example.sbatch"
    "jobs/cesvima/j11_uie_cache_base.sbatch"
    "jobs/cesvima/j12_uie_preflight.sbatch"
    "jobs/cesvima/j13_uie_train.sbatch"
    "jobs/cesvima/j14_uie_positive_smoke.sbatch"
  )
fi
if [[ "${DO_GOLLIE}" -eq 1 ]]; then
  SELECTED_JOBS+=(
    "jobs/cesvima/j15_gollie_preflight.sbatch"
    "jobs/cesvima/j16_gollie_download.sbatch"
    "jobs/cesvima/j17_gollie_native_smoke.sbatch"
  )
fi
if [[ "${DO_GENIE}" -eq 1 ]]; then
  SELECTED_JOBS+=(
    "jobs/cesvima/j18_genie_download_ckpt.sbatch"
    "jobs/cesvima/j19_genie_download_tokenizer.sbatch"
    "jobs/cesvima/j20_genie_download_tries.sbatch"
  )
fi
for relative in "${SELECTED_JOBS[@]}"; do
  require_file "${relative}"
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${P3_SLURM_LOG_DIR}"
fi

if [[ "${NEED_J00}" -eq 1 ]]; then
  submit_job J00 jobs/cesvima/j00_verify_infer_env.sbatch
fi
if [[ "${DO_MREBEL}" -eq 1 ]]; then
  submit_job J01 jobs/cesvima/j01_mrebel_cache_verify.sbatch J00
  submit_job J02 jobs/cesvima/j02_mrebel_smoke3.sbatch J01
  submit_job J03 jobs/cesvima/j03_mrebel_full.sbatch J02
fi
if [[ "${DO_REBEL}" -eq 1 ]]; then
  submit_job J04 jobs/cesvima/j04_rebel_resolve_revision.sbatch J00
  submit_job J05 jobs/cesvima/j05_rebel_download.sbatch J04
fi
if [[ "${DO_REBEL_CROSS}" -eq 1 ]]; then
  submit_job J07 jobs/cesvima/j07_rebel_hohfeld_cross_lingual.sbatch J05
fi
if [[ "${DO_UIE}" -eq 1 ]]; then
  submit_job J08 jobs/cesvima/j08_uie_fetch_upstreams.sbatch
  submit_job J11 jobs/cesvima/j11_uie_cache_base.sbatch J00
  submit_job J09 jobs/cesvima/j09_uie_rebuild_data.sbatch J00 J08
  submit_job J10 jobs/cesvima/j10_uie_freeze_smoke_example.sbatch J09
  submit_job J12 jobs/cesvima/j12_uie_preflight.sbatch J09 J11
  submit_job J13 jobs/cesvima/j13_uie_train.sbatch J12
  submit_job J14 jobs/cesvima/j14_uie_positive_smoke.sbatch J13 J10
fi
if [[ "${DO_GOLLIE}" -eq 1 ]]; then
  submit_job J15 jobs/cesvima/j15_gollie_preflight.sbatch
  submit_job J16 jobs/cesvima/j16_gollie_download.sbatch J15
  submit_job J17 jobs/cesvima/j17_gollie_native_smoke.sbatch J16
fi
if [[ "${DO_GENIE}" -eq 1 ]]; then
  submit_job J18 jobs/cesvima/j18_genie_download_ckpt.sbatch
  submit_job J19 jobs/cesvima/j19_genie_download_tokenizer.sbatch J00
  submit_job J20 jobs/cesvima/j20_genie_download_tries.sbatch
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY-RUN complete; sbatch was not executed"
fi
