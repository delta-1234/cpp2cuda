// GPU port of MomentumRank.cpp: per-session cross-sectional momentum rank factor.
// argv[1] selects the device algorithm ("simt" or "tile"); the remaining arguments are the
// original CPU arguments (one quotes CSV path).  Returns and rank scores are computed on the
// device; only CSV parsing/formatting stays on the host (rank_io.hpp is reused verbatim).

#include "rank_io.hpp"

#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;
using momentum::Quote;
using momentum::missing;
using momentum::read_quotes;
using momentum::write_values;

namespace {

constexpr int kTileThreads = 256;      // cooperative block used by the tile kernel
constexpr int kTileCap = 4096;         // shared-memory key capacity (32 KiB of int64)
constexpr std::int64_t kPad = 0x7FFFFFFFFFFFFFFFLL;  // bitonic padding sentinel
constexpr long long kScoreScale = 1000000LL;
constexpr long long kBpScale = 10000LL;

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(std::size_t bytes) {
        if (bytes == 0) return;
        cuda_check(cudaMalloc(&ptr, bytes), "cudaMalloc failed");
    }
    template <typename T> T* as() { return static_cast<T*>(ptr); }
};

// ---------------------------------------------------------------- device helpers

__device__ __forceinline__ std::int64_t device_return(std::int64_t prev, std::int64_t close) {
    if (prev == 0 || close == 0) return missing;            // unusable quote -> NA
    return (close - prev) * kBpScale / prev;                // signed truncation toward zero
}

__device__ __forceinline__ std::int64_t device_score(std::int64_t lower, std::int64_t upper,
                                                      std::int64_t count) {
    if (count == 1) return 0;                                // single peer scores exactly 0
    return (lower + upper - count) * kScoreScale / (count - 1);  // midpoint of zero-based ranks
}

__device__ __forceinline__ std::uint64_t date_key(const char* dates, std::size_t row) {
    const char* p = dates + row * 8;
    std::uint64_t key = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) key = (key << 8) | static_cast<std::uint8_t>(p[b]);
    return key;
}

// flags[i] == 1 when row i opens a new session; flags[n] is a sentinel closing the last one.
__global__ void session_flags_kernel(const char* __restrict__ dates, int n,
                                     int* __restrict__ flags) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return;
    if (i == 0 || i == n) { flags[i] = 1; return; }
    flags[i] = (date_key(dates, static_cast<std::size_t>(i)) !=
                date_key(dates, static_cast<std::size_t>(i - 1))) ? 1 : 0;
}

// starts[s] = first row of session s, starts[nsess] = n (from the sentinel flag).  On entry
// sess[] holds the exclusive prefix of flags[]; because flags[i] is 1 exactly at the first record
// of a group, row i belongs to session exclusive_prefix[i] + flags[i] - 1, so sess[] is rewritten
// in place to that session id (index n keeps holding nsess, since flags[n] is the sentinel 1).
__global__ void session_starts_kernel(const int* __restrict__ flags, int* __restrict__ sess,
                                      int n, int* __restrict__ starts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return;
    const int prefix = sess[i];
    if (flags[i] != 0) starts[prefix] = i;
    sess[i] = prefix + flags[i] - 1;
}

__global__ void momentum_returns_kernel(const std::int64_t* __restrict__ prev,
                                        const std::int64_t* __restrict__ close, int n,
                                        std::int64_t* __restrict__ ret) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ret[i] = device_return(prev[i], close[i]);
}

// ------------------------------------------------------------- SIMT rank kernel
// One thread per quote; the thread walks its whole session in global memory and counts the
// peers that sort strictly below / at-or-below its own return.  No shared memory, no
// inter-thread cooperation: pure SIMT, exactly reproducing the CPU tie-group midpoint.
__global__ void momentum_rank_simt_kernel(const std::int64_t* __restrict__ ret,
                                          const int* __restrict__ sess,
                                          const int* __restrict__ starts, int n,
                                          std::int64_t* __restrict__ score) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::int64_t r = ret[i];
    if (r == missing) { score[i] = missing; return; }
    const int s = sess[i];
    const int first = starts[s];
    const int end = starts[s + 1];
    std::int64_t less = 0, le = 0, count = 0;
    for (int j = first; j < end; ++j) {
        const std::int64_t v = ret[j];
        if (v == missing) continue;
        ++count;
        if (v < r) ++less;
        else if (v == r) ++le;
    }
    score[i] = device_score(less, less + le, count);
}

// ------------------------------------------------------------- Tile rank kernel
// One cooperative block per session.  The session's usable returns are compacted into shared
// memory, padded to a power of two and ordered with a block-wide bitonic network; every thread
// then binary-searches its own return to obtain the tie group bounds [lower, upper).  Sessions
// too large for the shared tile fall back to streaming the session through shared memory in
// tiles, which keeps the result identical.
__device__ __forceinline__ int shared_lower_bound(const std::int64_t* keys, int count,
                                                 std::int64_t r) {
    int lo = 0, hi = count;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (keys[mid] < r) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int shared_upper_bound(const std::int64_t* keys, int count,
                                                 std::int64_t r) {
    int lo = 0, hi = count;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (keys[mid] <= r) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__global__ void momentum_rank_tile_kernel(const std::int64_t* __restrict__ ret,
                                         const int* __restrict__ starts, int nsess,
                                         std::int64_t* __restrict__ score) {
    if (blockIdx.x >= static_cast<unsigned>(nsess)) return;
    cg::thread_block block = cg::this_thread_block();
    const int tid = static_cast<int>(block.thread_rank());
    const int nthr = static_cast<int>(block.size());

    __shared__ std::int64_t keys[kTileCap];
    __shared__ int sh_counter;

    const int first = starts[blockIdx.x];
    const int end = starts[blockIdx.x + 1];
    const int len = end - first;
    if (len <= 0) return;

    // Phase 1: how many quotes of this session are usable?
    if (tid == 0) sh_counter = 0;
    block.sync();
    for (int j = first + tid; j < end; j += nthr)
        if (ret[j] != missing) atomicAdd(&sh_counter, 1);
    block.sync();
    const int count = sh_counter;
    block.sync();   // every thread must finish reading count before thread zero resets it

    if (count <= kTileCap) {
        // Phase 2a: compact into shared memory.
        if (tid == 0) sh_counter = 0;
        block.sync();
        for (int j = first + tid; j < end; j += nthr) {
            const std::int64_t v = ret[j];
            if (v != missing) keys[atomicAdd(&sh_counter, 1)] = v;
        }
        block.sync();
        // Phase 3a: pad to a power of two and run the block-cooperative bitonic network.
        int span = 1;
        while (span < count) span <<= 1;
        for (int i = count + tid; i < span; i += nthr) keys[i] = kPad;
        block.sync();
        for (int k = 2; k <= span; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = tid; i < span; i += nthr) {
                    const int l = i ^ j;
                    if (l < i) continue;
                    const bool ascending = ((i & k) == 0);
                    const std::int64_t a = keys[i];
                    const std::int64_t b = keys[l];
                    if (ascending ? (a > b) : (a < b)) { keys[i] = b; keys[l] = a; }
                }
                block.sync();
            }
        }
        // Phase 4a: rank from the ordered shared tile.
        const int rounds = (len + nthr - 1) / nthr;
        for (int round = 0; round < rounds; ++round) {
            const int idx = round * nthr + tid;
            if (idx >= len) continue;
            const int j = first + idx;
            const std::int64_t r = ret[j];
            if (r == missing) { score[j] = missing; continue; }
            const int lower = shared_lower_bound(keys, count, r);
            const int upper = shared_upper_bound(keys, count, r);
            score[j] = device_score(lower, upper, count);
        }
        return;
    }

    // Phase 2b: oversized session -> stream it through shared memory tile by tile.  The tile
    // loop bounds depend only on the session length, so every thread reaches each barrier.
    const int rounds = (len + nthr - 1) / nthr;
    for (int round = 0; round < rounds; ++round) {
        const int idx = round * nthr + tid;
        const bool active = (idx < len);
        const int j = first + (active ? idx : 0);
        const std::int64_t r = active ? ret[j] : missing;
        std::int64_t less = 0, le = 0, cnt = 0;
        for (int base = first; base < end; base += kTileCap) {
            const int tile = min(kTileCap, end - base);
            for (int t = tid; t < tile; t += nthr) keys[t] = ret[base + t];
            block.sync();
            if (active && r != missing) {
                for (int t = 0; t < tile; ++t) {
                    const std::int64_t v = keys[t];
                    if (v == missing) continue;
                    ++cnt;
                    if (v < r) ++less;
                    else if (v == r) ++le;
                }
            }
            block.sync();
        }
        if (!active) continue;
        score[j] = (r == missing) ? missing : device_score(less, less + le, cnt);
    }
}

// ------------------------------------------------------------------- host driver

enum class Mode { Simt, Tile };

void launch_all(const std::vector<Quote>& rows, Mode mode, std::vector<std::int64_t>& returns,
                std::vector<std::int64_t>& scores) {
    const int n = static_cast<int>(rows.size());
    if (n == 0) return;   // header-only output; nothing to compute

    std::vector<char> dates(rows.size() * 8);
    std::vector<std::int64_t> prev(rows.size()), close(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::memcpy(dates.data() + i * 8, rows[i].date.data(), 8);
        prev[i] = rows[i].prev_close_cent;
        close[i] = rows[i].close_cent;
    }

    DeviceBuffer in_dates, in_prev, in_close, flags, sess, starts, out_ret, out_score, scratch;
    in_dates.allocate(dates.size());
    in_prev.allocate(prev.size() * sizeof(std::int64_t));
    in_close.allocate(close.size() * sizeof(std::int64_t));
    flags.allocate((rows.size() + 1) * sizeof(int));
    sess.allocate((rows.size() + 1) * sizeof(int));
    starts.allocate((rows.size() + 1) * sizeof(int));
    out_ret.allocate(rows.size() * sizeof(std::int64_t));
    out_score.allocate(rows.size() * sizeof(std::int64_t));

    cuda_check(cudaMemcpy(in_dates.ptr, dates.data(), dates.size(), cudaMemcpyHostToDevice),
               "uploading dates failed");
    cuda_check(cudaMemcpy(in_prev.ptr, prev.data(), prev.size() * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice), "uploading previous closes failed");
    cuda_check(cudaMemcpy(in_close.ptr, close.data(), close.size() * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice), "uploading closes failed");

    const int prep_threads = 256;
    const int prep_blocks = (n + 1 + prep_threads - 1) / prep_threads;
    const int rank_blocks = (n + prep_threads - 1) / prep_threads;

    session_flags_kernel<<<prep_blocks, prep_threads>>>(in_dates.as<char>(), n, flags.as<int>());
    cuda_check(cudaGetLastError(), "session flag launch failed");

    // Exclusive scan of the session flags -> session index per row, plus nsess at index n.
    std::size_t scratch_bytes = 0;
    cuda_check(cub::DeviceScan::ExclusiveSum(nullptr, scratch_bytes, flags.as<int>(),
                                             sess.as<int>(), n + 1), "sizing scan failed");
    scratch.allocate(scratch_bytes);
    cuda_check(cub::DeviceScan::ExclusiveSum(scratch.ptr, scratch_bytes, flags.as<int>(),
                                             sess.as<int>(), n + 1), "session scan failed");

    session_starts_kernel<<<prep_blocks, prep_threads>>>(flags.as<int>(), sess.as<int>(), n,
                                                        starts.as<int>());
    cuda_check(cudaGetLastError(), "session start launch failed");

    momentum_returns_kernel<<<rank_blocks, prep_threads>>>(in_prev.as<std::int64_t>(),
                                                          in_close.as<std::int64_t>(), n,
                                                          out_ret.as<std::int64_t>());
    cuda_check(cudaGetLastError(), "return launch failed");

    if (mode == Mode::Simt) {
        momentum_rank_simt_kernel<<<rank_blocks, prep_threads>>>(
            out_ret.as<std::int64_t>(), sess.as<int>(), starts.as<int>(), n,
            out_score.as<std::int64_t>());
        cuda_check(cudaGetLastError(), "SIMT rank launch failed");
    } else {
        int nsess = 0;
        cuda_check(cudaMemcpy(&nsess, sess.as<int>() + n, sizeof(int), cudaMemcpyDeviceToHost),
                   "reading session count failed");
        if (nsess > 0) {
            momentum_rank_tile_kernel<<<nsess, kTileThreads>>>(
                out_ret.as<std::int64_t>(), starts.as<int>(), nsess, out_score.as<std::int64_t>());
            cuda_check(cudaGetLastError(), "tile rank launch failed");
        }
    }

    cuda_check(cudaMemcpy(returns.data(), out_ret.ptr, rows.size() * sizeof(std::int64_t),
                          cudaMemcpyDeviceToHost), "downloading returns failed");
    cuda_check(cudaMemcpy(scores.data(), out_score.ptr, rows.size() * sizeof(std::int64_t),
                          cudaMemcpyDeviceToHost), "downloading scores failed");
    cuda_check(cudaDeviceSynchronize(), "device synchronisation failed");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 || (std::string(argv[1]) != "simt" && std::string(argv[1]) != "tile")) {
        std::cerr << "Usage: momentum_rank <quotes.csv>\n";
        return 2;
    }
    const Mode mode = (std::string(argv[1]) == "simt") ? Mode::Simt : Mode::Tile;
    try {
        const auto rows = read_quotes(argv[2]);
        std::vector<std::int64_t> returns(rows.size(), missing), scores(rows.size(), missing);
        launch_all(rows, mode, returns, scores);
        write_values(rows, returns, scores);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
    return 0;
}
