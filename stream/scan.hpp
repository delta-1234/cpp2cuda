#pragma once
#include <cstdint>

// Row zero of a segment holds absolute totals; subsequent rows hold differences.
struct Changes {
    std::int64_t ask_volume, ask_filled, bid_volume, bid_filled;
};
struct Rates {
    double ask, bid, spread;
};
constexpr int segment_size = 128;
constexpr int capacity = 8192;  // Chronological outputs in one in-memory batch.

struct Slot {
    Changes* deltas;
    int* destinations;  // Chronological row numbers, grouped by symbol segment.
    int* offsets;       // One start per segment plus a sentinel.
    Rates* rates;
};

// The driver uses one reusable CUDA context, stream and pinned slot per process.
enum class Backend { Simt, Tile, Triton };

class GpuScan {
public:
    GpuScan(Backend backend, int device);
    ~GpuScan();
    GpuScan(const GpuScan&) = delete;
    GpuScan& operator=(const GpuScan&) = delete;
    Slot buffers() const;
    void scan(int segments, int rows);
private:
    struct Impl;
    Impl* impl_;
};
