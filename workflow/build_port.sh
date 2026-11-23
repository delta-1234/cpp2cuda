#!/usr/bin/env bash
# Compile model-selected source files from the read-only port directory.
set -euo pipefail
toolkit=$1
arch=$2
cuda_std=$3
host_std=$4
src=$5
out=$6
mkdir -p "$out"
nvcc_args=(-std="c++$cuda_std" -O3 "-arch=sm_$arch" -I. -I"$src")
if [[ -n ${NVCC_CCBIN:-} ]]; then nvcc_args+=(-ccbin "$NVCC_CCBIN"); fi
shopt -s nullglob
cuda_files=("$src"/*.cu)
if (( ${#cuda_files[@]} == 0 )); then echo "add a CUDA translation unit" >&2; exit 1; fi
objects=()
for file in "${cuda_files[@]}"; do
    obj="$out/$(basename "$file").o"
    "$toolkit/bin/nvcc" "${nvcc_args[@]}" -c "$file" -o "$obj"
    objects+=("$obj")
done
for file in "$src"/*.cpp "$src"/*.cc "$src"/*.cxx; do
    obj="$out/$(basename "$file").o"
    "${CXX:-c++}" -O3 -std="c++$host_std" -I. -I"$src" -I"$toolkit/include" -c "$file" -o "$obj"
    objects+=("$obj")
done
"${CXX:-c++}" "${objects[@]}" -L"$toolkit/lib64" -Wl,-rpath,"$toolkit/lib64" -lcudart -o "$out/replay"
