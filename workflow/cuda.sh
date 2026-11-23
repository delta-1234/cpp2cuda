#!/usr/bin/env bash
# Use the offline toolchain and visible GPU verified on a CUDA node.
set -eu
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.3}"
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-7}"
# Preserve an existing scheduler allocation, including an intentionally empty mask.
export WORKFLOW_PYTHON="${WORKFLOW_PYTHON:-python3}"
case "${1:-}" in
    --configure-api|--check-api|--agent)
        case "$1" in
            --configure-api) operation=configure ;;
            --check-api) operation=check ;;
            --agent) operation=run; export WITH_TRITON="${WITH_TRITON:-1}" ;;
        esac
        shift
        exec "$WORKFLOW_PYTHON" workflow/api_agent.py "$operation" "$@"
        ;;
    --build-stream)
        shift
        "$WORKFLOW_PYTHON" workflow/run.py --probe-only
        WITH_TRITON="${WITH_TRITON:-1}" exec bash workflow/build_stream.sh "$@"
        ;;
    --stream)
        shift
        exec "${STREAM_REPLAY:-runs/stream/replay}" "$@"
        ;;
    --scale)
        shift
        if [[ ${WITH_TRITON:-1} == 1 ]]; then set -- --with-triton "$@"; fi
        exec "$WORKFLOW_PYTHON" workflow/scale.py "$@"
        ;;
    *) exec "$WORKFLOW_PYTHON" workflow/run.py "$@" ;;
esac
