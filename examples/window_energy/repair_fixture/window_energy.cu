#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace {

constexpr int R = 2;
constexpr int TILE = 256;

__global__ void energy_simt(const int* __restrict__ in, int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int energy = 0;
#pragma unroll
    for (int k = -R; k <= R; ++k) {
        int j = i + k;
        if (j >= 0 && j < n) energy += in[j];
    }
    out[i] = energy;
}

__global__ void energy_tile(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ int s[TILE + 2 * R];

    const int base = blockIdx.x * TILE;

    for (int idx = threadIdx.x; idx < TILE + 2 * R; idx += blockDim.x) {
        int g = base - R + idx + (base != 0);
        s[idx] = (g >= 0 && g < n) ? in[g] : 0;
    }
    __syncthreads();

    int i = base + threadIdx.x;
    if (i < n) {
        int energy = 0;
#pragma unroll
        for (int k = -R; k <= R; ++k) {
            energy += s[threadIdx.x + R + k];
        }
        out[i] = energy;
    }
}

inline bool cuda_ok(cudaError_t e) { return e == cudaSuccess; }

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) return 1;

    const std::string mode = argv[1];
    if (mode != "simt" && mode != "tile") return 1;

    const int n = std::atoi(argv[2]);
    if (n < 1 || n > 1000000) return 1;

    std::vector<int> h_in(n);
    for (int i = 0; i < n; ++i) {
        int x = i % 17 - 8;
        h_in[i] = x * x;
    }

    if (!cuda_ok(cudaSetDevice(0))) return 1;

    int* d_in = nullptr;
    int* d_out = nullptr;
    if (!cuda_ok(cudaMalloc(&d_in, sizeof(int) * n))) return 1;
    if (!cuda_ok(cudaMalloc(&d_out, sizeof(int) * n))) {
        cudaFree(d_in);
        return 1;
    }

    if (!cuda_ok(cudaMemcpy(d_in, h_in.data(), sizeof(int) * n, cudaMemcpyHostToDevice))) {
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    const int block = TILE;
    const int grid = (n + block - 1) / block;

    if (mode == "simt") {
        energy_simt<<<grid, block>>>(d_in, d_out, n);
    } else {
        energy_tile<<<grid, block>>>(d_in, d_out, n);
    }

    if (!cuda_ok(cudaGetLastError())) {
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    std::vector<int> h_out(n);
    if (!cuda_ok(cudaMemcpy(h_out.data(), d_out, sizeof(int) * n, cudaMemcpyDeviceToHost))) {
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    if (!cuda_ok(cudaFree(d_in))) {
        cudaFree(d_out);
        return 1;
    }
    if (!cuda_ok(cudaFree(d_out))) return 1;

    std::ios::sync_with_stdio(false);
    for (int i = 0; i < n; ++i) {
        std::cout << h_out[i] << '\n';
    }
    return 0;
}
