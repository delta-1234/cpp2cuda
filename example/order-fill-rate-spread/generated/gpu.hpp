#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace gpu {

// The packed row layout shared by host packing and GPU kernels:
//   features[0..127]   signed order volume: ask > 0, bid < 0
//   features[128..255] corresponding filled volume, same slot order
constexpr std::size_t window_slots = 128;
constexpr std::size_t feature_count = 2 * window_slots;

// A bounded default batch size. Host code may choose another capacity.
constexpr std::size_t default_batch_capacity = 65536;

enum class Mode {
    Simt,
    Tile
};

struct InputRow {
    std::int32_t time_ms;
    std::int64_t trade_count;
    std::int32_t features[feature_count];
};

struct OutputRow {
    std::int32_t time_ms;
    std::int64_t trade_count;
    double ask_fill_rate;
    double bid_fill_rate;
    double alpha;
};

// Opaque device-state handle. Created once per run and reused for batches.
struct Plan;

cudaError_t create_plan(Plan** plan, Mode mode, std::size_t capacity);
void destroy_plan(Plan* plan);

// Processes 0..capacity rows. count may be a partial batch.
// NA values are returned as NaN in the double fields.
cudaError_t process_batch(Plan* plan,
                          const InputRow* host_input,
                          std::size_t count,
                          OutputRow* host_output);

}  // namespace gpu
