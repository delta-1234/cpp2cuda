// Shared batch I/O and event timing for the two native CUDA backends.
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
#define CUDA(call) check((call), #call)

constexpr int width = 128;
constexpr int batch = 4096;
constexpr int warps_per_block = 8;

__device__ void store_rates(double* result, long long av, long long af, long long bv, long long bf) {
    double ask = av ? static_cast<double>(af) / av : -2.0;
    double bid = bv ? static_cast<double>(bf) / bv : -2.0;
    result[0] = ask;
    result[1] = bid;
    result[2] = av && bv ? ask - bid : -2.0;
}

#ifdef SHARED_TILE
#include "shared_tile.cuh"
#else
#include "simt.cuh"
#endif

void launch(int* windows, double* rates, int rows) {
#ifdef SHARED_TILE
    tiled_rate_spread<<<(rows + 7) / 8, dim3(16, 8)>>>(windows, rates, rows);
#else
    simt_rate_spread<<<(rows + warps_per_block - 1) / warps_per_block, warps_per_block * 32>>>(windows, rates, rows);
#endif
    CUDA(cudaGetLastError());
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: backend windows.bin rates.bin device");
        CUDA(cudaSetDevice(std::stoi(argv[3])));
        std::ifstream input(argv[1], std::ios::binary);
        std::ofstream output(argv[2], std::ios::binary);
        if (!input || !output) throw std::runtime_error("cannot open input/output");
        static_assert(sizeof(int) == 4 && sizeof(double) == 8, "unexpected scalar width");
        std::vector<int> host_input(batch * 2 * width);
        std::vector<double> host_output(batch * 3);
        int* windows = nullptr;
        double* rates = nullptr;
        CUDA(cudaMalloc(&windows, host_input.size() * sizeof(int)));
        CUDA(cudaMalloc(&rates, host_output.size() * sizeof(double)));
        cudaEvent_t start, stop;
        CUDA(cudaEventCreate(&start));
        CUDA(cudaEventCreate(&stop));
        double total_ms = 0;
        bool warmed = false;
        while (true) {
            input.read(reinterpret_cast<char*>(host_input.data()), host_input.size() * sizeof(int));
            auto bytes = input.gcount();
            if (input.bad() || bytes % (2 * width * sizeof(int)) != 0)
                throw std::runtime_error("unreadable or incomplete window");
            if (!bytes) break;
            int rows = static_cast<int>(bytes / (2 * width * sizeof(int)));
            CUDA(cudaMemcpy(windows, host_input.data(), bytes, cudaMemcpyHostToDevice));
            if (!warmed) {
                launch(windows, rates, rows);
                CUDA(cudaDeviceSynchronize());
                warmed = true;
            }
            CUDA(cudaEventRecord(start));
            launch(windows, rates, rows);
            CUDA(cudaEventRecord(stop));
            CUDA(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
            auto result_bytes = rows * 3 * sizeof(double);
            CUDA(cudaMemcpy(host_output.data(), rates, result_bytes, cudaMemcpyDeviceToHost));
            output.write(reinterpret_cast<const char*>(host_output.data()), result_bytes);
        }
        output.close();
        if (!output) throw std::runtime_error("output write failed");
        CUDA(cudaEventDestroy(start));
        CUDA(cudaEventDestroy(stop));
        CUDA(cudaFree(rates));
        CUDA(cudaFree(windows));
        std::cout << total_ms << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
