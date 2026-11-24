#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
src=example/order-fill-rate-spread/stream
dest=${1:-runs/stream}
mkdir -p "$dest"
nvcc=${CUDA_HOME:+$CUDA_HOME/bin/}nvcc
args=(-std=c++14 -O3 -Xcompiler=-fPIC)
libs=()
if [[ ${WITH_TRITON:-0} == 1 ]]; then
    "${WORKFLOW_PYTHON:-python3}" "$src/aot_tile.py" "$dest/stream_triton.inc"
    args+=(-I"$dest" -DHAVE_TRITON_AOT)
    libs+=(-lcuda)
fi
if [[ -n ${NVCC_CCBIN:-} ]]; then args+=(-ccbin "$NVCC_CCBIN"); fi
# Override with the architecture returned by the inventory on another node.
"$nvcc" "${args[@]}" "-arch=sm_${CUDA_SM:-86}" -c "$src/scan.cu" -o "$dest/scan.o"
"${CXX:-c++}" -std=c++17 -O3 -c "$src/replay.cpp" -o "$dest/replay.o"
toolkit=${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")}
"${CXX:-c++}" "$dest/scan.o" "$dest/replay.o" -L"$toolkit/lib64" -Wl,-rpath,"$toolkit/lib64" -lcudart "${libs[@]}" -o "$dest/replay"
