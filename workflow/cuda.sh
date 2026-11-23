#!/usr/bin/env bash
# Use the offline toolchain and visible GPU verified on a CUDA node.
set -eu
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.3}"
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-7}"
# Preserve an existing scheduler allocation, including an intentionally empty mask.
exec "${WORKFLOW_PYTHON:-python3}" workflow/run.py "$@"
