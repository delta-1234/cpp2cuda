#!/usr/bin/env bash
# Build the model's C++17 host and CUDA C++14 implementation as separate translation units.
set -euo pipefail
src=$1
out=$2
include=$3
arch=$4
toolkit=${CUDA_HOME:?CUDA_HOME is required}
mkdir -p "$out"
nvcc_args=(-std=c++14 -O3 "-arch=sm_$arch")
if [[ -n ${NVCC_CCBIN:-} ]]; then nvcc_args+=(-ccbin "$NVCC_CCBIN"); fi
"$toolkit/bin/nvcc" "${nvcc_args[@]}" -c "$src/kernels.cu" -o "$out/kernels.o"
"${CXX:-c++}" -std=c++17 -O3 -I"$include" -I"$toolkit/include" "$src/host.cpp" "$out/kernels.o" \
    -L"$toolkit/lib64" -Wl,-rpath,"$toolkit/lib64" -lcudart -o "$out/replay"
