#!/bin/bash
# Central CESVIMA paths for Paper 3 train-first jobs.
# Every sbatch and the submitter sources this file. Do not duplicate these
# assignments in job bodies.

export PROJECT_ROOT=/home/y064/PROJECT/y064644/paper3
export SCRATCH_ROOT=/home/y064/SCRATCH/y064644/paper3
export SYSTEM_ROOT="${PROJECT_ROOT}/re_te_system"
export BENCHMARK_ROOT="${PROJECT_ROOT}/re_te_benchmark"
export EVALUATION_ROOT="${PROJECT_ROOT}/re_te_evaluation"

export HF_HOME="${SCRATCH_ROOT}/hf-cache"
export HF_HUB_CACHE="${SCRATCH_ROOT}/hf-cache/hub"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/pip-cache"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export PYTHONUNBUFFERED=1

export INFER_VENV="${SCRATCH_ROOT}/venvs/re_te_system_gpu_312"
# re_te_system_312 remains on SCRATCH and is not the GPU runtime.
# Do not point INFER_VENV at it and do not create paper3-infer.
export UIE_TRAIN_VENV="${SCRATCH_ROOT}/venvs/uie-train"
export GOLLIE_VENV="${SCRATCH_ROOT}/venvs/gollie-l40s"

export P3_RUNS_ROOT="${SCRATCH_ROOT}/runs/p3"
export P3_SLURM_LOG_DIR="${P3_RUNS_ROOT}/slurm"
export P3_MODELS_ROOT="${SCRATCH_ROOT}/models"
export P3_UPSTREAM_ROOT="${SCRATCH_ROOT}/upstreams"

# Documented Magerit partitions and typed GRES.
# https://docs.cesvima.upm.es/magerit/jobs/
# CPU: standard. GPU: standard-gpu plus an explicit type.
# Untyped --gres=gpu:N is rejected because it can land on any GPU.
export PARTITION_CPU=standard
export PARTITION_GPU=standard-gpu
export GRES_V100=gpu:v100:1
export GRES_L40S=gpu:l40s:1

export CANONICAL_DOCUMENTS="${BENCHMARK_ROOT}/datasets/teresia_hohfeld/canonical/documents.jsonl"
export CANONICAL_MANIFEST="${BENCHMARK_ROOT}/datasets/teresia_hohfeld/manifests/manifest.json"
export UIE_MANIFEST="${SYSTEM_ROOT}/artifacts/uie_conll04_10shot_seed1_manifest.json"

cesvima_prepare_runtime_dirs() {
  mkdir -p \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${PIP_CACHE_DIR}" \
    "${TMPDIR}" \
    "${P3_RUNS_ROOT}" \
    "${P3_SLURM_LOG_DIR}" \
    "${P3_MODELS_ROOT}" \
    "${P3_UPSTREAM_ROOT}" \
    "${SCRATCH_ROOT}/venvs"
}

cesvima_guard() {
  local haystack
  haystack="${SLURM_JOB_PARTITION:-} ${SLURM_GPUS:-} ${SLURM_JOB_GRES:-} ${CUDA_VISIBLE_DEVICES:-}"
  if printf '%s' "${haystack}" | grep -Eq 'A100|a100'; then
    echo "FAIL: A100 is forbidden for Paper 3 train-first jobs" >&2
    exit 3
  fi
  case "${SCRATCH_ROOT}" in
    "${PROJECT_ROOT}"*)
      echo "FAIL: SCRATCH_ROOT is inside PROJECT_ROOT" >&2
      exit 3
      ;;
  esac
  case "${HF_HOME}" in
    "${SYSTEM_ROOT}"*|"${BENCHMARK_ROOT}"*|"${EVALUATION_ROOT}"*)
      echo "FAIL: HF_HOME points at a repository tree" >&2
      exit 3
      ;;
  esac
}
