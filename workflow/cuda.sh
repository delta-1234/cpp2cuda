#!/usr/bin/env bash
# Use the offline toolchain and visible GPU verified on a CUDA node.
set -eu
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.3}"
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-7}"
# The separately installed Nsight reliably exports short CUDA 11.3 kernels on this node.
reference_nsys=/usr/local/cuda-11.7/nsight-systems-2022.1.3/bin/nsys
if [[ -x $reference_nsys ]]; then export NSYS="${NSYS:-$reference_nsys}"; fi
# Preserve an existing scheduler allocation, including an intentionally empty mask.
export WORKFLOW_PYTHON="${WORKFLOW_PYTHON:-python3}"
case "${1:-}" in
    --test-repair)
        shift
        exec "$WORKFLOW_PYTHON" workflow/test_port_agent.py --gpu "$@"
        ;;
    --port-agent)
        shift
        exec "$WORKFLOW_PYTHON" workflow/port_agent.py "$@"
        ;;
    --configure-api|--check-api|--agent)
        case "$1" in
            --configure-api) operation=configure ;;
            --check-api) operation=check ;;
            --agent)
                for option in "$@"; do
                    case "$option" in
                        --source|--spec|--source=*|--spec=*|--inspect)
                            shift; exec "$WORKFLOW_PYTHON" workflow/port_agent.py "$@" ;;
                    esac
                done
                operation=run; export WITH_TRITON="${WITH_TRITON:-1}" ;;
        esac
        shift
        exec "$WORKFLOW_PYTHON" workflow/api_agent.py "$operation" "$@"
        ;;
    --generate-cuda)
        shift
        if [[ ${1:-} == OrderFillRateSpread.cpp && ! -e OrderFillRateSpread.cpp ]]; then
            shift
            set -- example/order-fill-rate-spread/OrderFillRateSpread.cpp "$@"
        fi
        for option in "$@"; do
            case "$option" in
                --per-file|--no-per-file) exec "$WORKFLOW_PYTHON" workflow/generate_cuda.py "$@" ;;
            esac
        done
        exec "$WORKFLOW_PYTHON" workflow/port_agent.py --optimize "$@"
        ;;
    --build-generated)
        shift
        exec bash workflow/build_generated.sh "${1:-example/order-fill-rate-spread/generated}" "${2:-runs/generated}" example/order-fill-rate-spread "${CUDA_SM:-86}"
        ;;
    --generated)
        shift
        exec runs/generated/replay "$@"
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
