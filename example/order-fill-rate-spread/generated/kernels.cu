#include "gpu.hpp"

#include <cstddef>
#include <cstdint>
#include <new>

#define GPU_RETURN_IF_ERROR(expr)                                        \
    do {                                                                 \
        cudaError_t gpu_return_if_error_code_ = (expr);                  \
        if (gpu_return_if_error_code_ != cudaSuccess)                    \
            return gpu_return_if_error_code_;                            \
    } while (0)

namespace {

constexpr int kSlots = static_cast<int>(gpu::window_slots);
static_assert(kSlots == 128, "The packed row layout reserves 128 order slots.");

constexpr int kSimtThreads = 128;
constexpr int kTileRows = 8;

constexpr unsigned long long kQuietNaNBits = 0x7ff8000000000000ULL;

__device__ __forceinline__ double quiet_nan() {
    return __longlong_as_double(static_cast<long long>(kQuietNaNBits));
}

__device__ __forceinline__ void finish_output_row(gpu::OutputRow& out,
                                                  const gpu::InputRow& in,
                                                  long long ask_volume,
                                                  long long ask_filled,
                                                  long long bid_volume,
                                                  long long bid_filled) {
    out.time_ms = in.time_ms;
    out.trade_count = in.trade_count;

    double ask_rate = quiet_nan();
    if (ask_volume != 0)
        ask_rate = static_cast<double>(ask_filled) / static_cast<double>(ask_volume);

    double bid_rate = quiet_nan();
    if (bid_volume != 0)
        bid_rate = static_cast<double>(bid_filled) / static_cast<double>(bid_volume);

    out.ask_fill_rate = ask_rate;
    out.bid_fill_rate = bid_rate;
    out.alpha = (ask_volume == 0 || bid_volume == 0) ? quiet_nan() : (ask_rate - bid_rate);
}

// One SIMT thread independently computes one output row.
__global__ void simt_kernel(const gpu::InputRow* __restrict__ input,
                            gpu::OutputRow* __restrict__ output,
                            int count) {
    int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= count)
        return;

    const gpu::InputRow& in = input[row];
    long long ask_volume = 0;
    long long ask_filled = 0;
    long long bid_volume = 0;
    long long bid_filled = 0;

    for (int i = 0; i < kSlots; ++i) {
        long long volume = static_cast<long long>(in.features[i]);
        long long filled = static_cast<long long>(in.features[kSlots + i]);

        if (volume > 0) {
            ask_volume += volume;
            ask_filled += filled;
        } else if (volume < 0) {
            bid_volume -= volume;
            bid_filled += filled;
        }
    }

    finish_output_row(output[row], in, ask_volume, ask_filled, bid_volume, bid_filled);
}

// One block processes a small tile of rows. For each row, all threads
// cooperatively reduce the 128 order slots through shared memory.
__global__ void tile_kernel(const gpu::InputRow* __restrict__ input,
                            gpu::OutputRow* __restrict__ output,
                            int count) {
    __shared__ long long shared[4 * kSlots];

    long long* s_ask_volume = shared;
    long long* s_ask_filled = shared + kSlots;
    long long* s_bid_volume = shared + 2 * kSlots;
    long long* s_bid_filled = shared + 3 * kSlots;

    int tile_start = static_cast<int>(blockIdx.x) * kTileRows;
    int tile_end = tile_start + kTileRows;
    if (tile_end > count)
        tile_end = count;

    int t = threadIdx.x;

    for (int row = tile_start; row < tile_end; ++row) {
        const gpu::InputRow& in = input[row];

        long long ask_volume = 0;
        long long ask_filled = 0;
        long long bid_volume = 0;
        long long bid_filled = 0;

        long long volume = static_cast<long long>(in.features[t]);
        long long filled = static_cast<long long>(in.features[kSlots + t]);

        if (volume > 0) {
            ask_volume = volume;
            ask_filled = filled;
        } else if (volume < 0) {
            bid_volume = -volume;
            bid_filled = filled;
        }

        s_ask_volume[t] = ask_volume;
        s_ask_filled[t] = ask_filled;
        s_bid_volume[t] = bid_volume;
        s_bid_filled[t] = bid_filled;
        __syncthreads();

        for (int stride = kSlots / 2; stride > 0; stride >>= 1) {
            if (t < stride) {
                s_ask_volume[t] += s_ask_volume[t + stride];
                s_ask_filled[t] += s_ask_filled[t + stride];
                s_bid_volume[t] += s_bid_volume[t + stride];
                s_bid_filled[t] += s_bid_filled[t + stride];
            }
            __syncthreads();
        }

        if (t == 0) {
            finish_output_row(output[row], in,
                              s_ask_volume[0], s_ask_filled[0],
                              s_bid_volume[0], s_bid_filled[0]);
        }
        __syncthreads();
    }
}

}  // namespace

namespace gpu {

struct Plan {
    Mode mode = Mode::Simt;
    std::size_t capacity = 0;
    InputRow* d_input = nullptr;
    OutputRow* d_output = nullptr;
};

cudaError_t create_plan(Plan** plan, Mode mode, std::size_t capacity) {
    if (plan == nullptr || capacity == 0)
        return cudaErrorInvalidValue;
    if (mode != Mode::Simt && mode != Mode::Tile)
        return cudaErrorInvalidValue;

    if (capacity > static_cast<std::size_t>(-1) / sizeof(InputRow) ||
        capacity > static_cast<std::size_t>(-1) / sizeof(OutputRow)) {
        return cudaErrorInvalidValue;
    }

    GPU_RETURN_IF_ERROR(cudaFree(0));

    Plan* p = new (std::nothrow) Plan();
    if (p == nullptr)
        return cudaErrorMemoryAllocation;

    p->mode = mode;
    p->capacity = capacity;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&p->d_input),
                                 capacity * sizeof(InputRow));
    if (err != cudaSuccess) {
        delete p;
        return err;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&p->d_output),
                     capacity * sizeof(OutputRow));
    if (err != cudaSuccess) {
        cudaFree(p->d_input);
        delete p;
        return err;
    }

    *plan = p;
    return cudaSuccess;
}

void destroy_plan(Plan* plan) {
    if (plan == nullptr)
        return;

    if (plan->d_input != nullptr)
        cudaFree(plan->d_input);
    if (plan->d_output != nullptr)
        cudaFree(plan->d_output);

    delete plan;
}

cudaError_t process_batch(Plan* plan,
                          const InputRow* host_input,
                          std::size_t count,
                          OutputRow* host_output) {
    if (plan == nullptr)
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    if (host_input == nullptr || host_output == nullptr)
        return cudaErrorInvalidValue;
    if (count > plan->capacity)
        return cudaErrorInvalidValue;

    constexpr std::size_t kMaxCount = static_cast<std::size_t>(2147483647);
    if (count > kMaxCount)
        return cudaErrorInvalidValue;

    int n = static_cast<int>(count);

    GPU_RETURN_IF_ERROR(cudaMemcpy(plan->d_input, host_input,
                                   count * sizeof(InputRow),
                                   cudaMemcpyHostToDevice));

    if (plan->mode == Mode::Simt) {
        unsigned int grid = static_cast<unsigned int>(
            (count + kSimtThreads - 1) / kSimtThreads);
        simt_kernel<<<grid, kSimtThreads>>>(plan->d_input, plan->d_output, n);
    } else if (plan->mode == Mode::Tile) {
        unsigned int grid = static_cast<unsigned int>(
            (count + kTileRows - 1) / kTileRows);
        tile_kernel<<<grid, kSlots>>>(plan->d_input, plan->d_output, n);
    } else {
        return cudaErrorInvalidValue;
    }

    GPU_RETURN_IF_ERROR(cudaGetLastError());
    GPU_RETURN_IF_ERROR(cudaDeviceSynchronize());

    GPU_RETURN_IF_ERROR(cudaMemcpy(host_output, plan->d_output,
                                   count * sizeof(OutputRow),
                                   cudaMemcpyDeviceToHost));

    return cudaSuccess;
}

}  // namespace gpu
