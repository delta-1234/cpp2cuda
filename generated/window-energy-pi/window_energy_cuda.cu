// GPU port of window_energy.cpp
// For every i in [0,n) print sum over j in [i-2,i+2] intersect [0,n) of ((j%17)-8)^2.
// argv[1] selects the device strategy: "simt" (one thread per output, global window)
// or "tile" (shared-memory tiled stencil with halos).
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA failure %s at %s:%d\n",               \
                         cudaGetErrorString(err__), __FILE__, __LINE__);     \
            std::exit(2);                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------- device side

__device__ __forceinline__ int wsq(int j) {
    const int x = j % 17 - 8;
    return x * x;
}

// materialise the base sequence s[j] = ((j%17)-8)^2 on the device
__global__ void fill_base_kernel(int* __restrict__ s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) s[i] = wsq(i);
}

// SIMT path: each thread owns one output sample. The base sequence is a pure
// function of the index, so instead of loading a materialised array the thread
// regenerates its whole 5-tap window in registers. That removes the separate
// fill launch (and its global store/load round trip) from this path while
// keeping one contiguous output element per lane, so the warp still writes a
// fully coalesced 128-byte segment and stays divergence-free except at the two
// boundary lanes.
__global__ void simt_window_kernel(int* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int energy = 0;
#pragma unroll
    for (int k = -2; k <= 2; ++k) {
        const int j = i + k;
        if (j >= 0 && j < n) energy += wsq(j);
    }
    out[i] = energy;
}

// Tiled path: a block stages a contiguous tile of the base sequence (plus two
// halo samples on each side) into shared memory once, then every thread folds
// its window out of shared memory. Out-of-range halo samples stage as 0 so the
// [0,n) truncation of the reference is reproduced exactly.
#define TILE_OUT 256
#define TILE_HALO 2

__global__ void tile_window_kernel(const int* __restrict__ s,
                                   int* __restrict__ out, int n) {
    __shared__ int sh[TILE_OUT + 2 * TILE_HALO];

    const int base = blockIdx.x * TILE_OUT;
    const int tid = threadIdx.x;

    for (int k = tid; k < TILE_OUT + 2 * TILE_HALO; k += blockDim.x) {
        const int j = base - TILE_HALO + k;
        sh[k] = (j >= 0 && j < n) ? s[j] : 0;
    }
    __syncthreads();

    for (int i = tid; i < TILE_OUT; i += blockDim.x) {
        const int gi = base + i;
        if (gi < n) {
            // window j in [gi-2, gi+2] -> shared indices [i, i+4]
            out[gi] = sh[i] + sh[i + 1] + sh[i + 2] + sh[i + 3] + sh[i + 4];
        }
    }
}

// ------------------------------------------------------------------ host side

namespace {

char g_obuf[1u << 20];

void emit_all(const int* v, int n) {
    std::size_t op = 0;
    char tmp[16];
    for (int i = 0; i < n; ++i) {
        int x = v[i];
        int t = 0;
        if (x == 0) {
            tmp[t++] = '0';
        } else {
            if (x < 0) { tmp[t++] = '-'; x = -x; }
            char digits[12];
            int d = 0;
            while (x > 0) { digits[d++] = char('0' + (x % 10)); x /= 10; }
            while (d > 0) tmp[t++] = digits[--d];
        }
        if (op + static_cast<std::size_t>(t) + 1u > sizeof g_obuf) {
            std::fwrite(g_obuf, 1, op, stdout);
            op = 0;
        }
        std::memcpy(g_obuf + op, tmp, static_cast<std::size_t>(t));
        op += static_cast<std::size_t>(t);
        g_obuf[op++] = '\n';
    }
    if (op) std::fwrite(g_obuf, 1, op, stdout);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) return 1;

    const bool use_simt = (std::strcmp(argv[1], "simt") == 0);
    const bool use_tile = (std::strcmp(argv[1], "tile") == 0);
    if (!use_simt && !use_tile) return 1;

    const int n = std::atoi(argv[2]);
    if (n < 1 || n > 1000000) return 1;

    int* d_s = nullptr;
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_s, sizeof(int) * static_cast<std::size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * static_cast<std::size_t>(n)));

    const int threads = 256;
    const int fill_blocks = (n + threads - 1) / threads;

    if (use_simt) {
        // Fused: the index-generated base sequence is recomputed per window tap,
        // so no fill kernel is launched on this path.
        simt_window_kernel<<<fill_blocks, threads>>>(d_out, n);
    } else {
        fill_base_kernel<<<fill_blocks, threads>>>(d_s, n);
        const int tile_blocks = (n + TILE_OUT - 1) / TILE_OUT;
        tile_window_kernel<<<tile_blocks, TILE_OUT>>>(d_s, d_out, n);
    }
    CUDA_CHECK(cudaGetLastError());

    int* h_out = static_cast<int*>(std::malloc(sizeof(int) * static_cast<std::size_t>(n)));
    if (!h_out) return 1;
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int) * static_cast<std::size_t>(n),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    emit_all(h_out, n);

    std::free(h_out);
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
