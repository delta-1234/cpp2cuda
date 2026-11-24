#include "scan.hpp"
#include <cuda_runtime.h>
#ifdef HAVE_TRITON_AOT
#include <cuda.h>
#include "stream_triton.inc"
#endif
#include <stdexcept>
#include <string>

namespace {
void checked(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__device__ Changes add(Changes a, Changes b) {
    return {a.ask_volume + b.ask_volume, a.ask_filled + b.ask_filled,
            a.bid_volume + b.bid_volume, a.bid_filled + b.bid_filled};
}
__device__ void store(const Changes& s, int dst, Rates* out) {
    if (dst < 0) return;
    double a = s.ask_volume ? double(s.ask_filled) / double(s.ask_volume) : -2.;
    double b = s.bid_volume ? double(s.bid_filled) / double(s.bid_volume) : -2.;
    out[dst] = {a, b, s.ask_volume && s.bid_volume ? a - b : -2.};
}
__device__ long long warp_prefix(long long v, int lane) {
    for (int d = 1; d < 32; d *= 2) {
        long long prior = __shfl_up_sync(0xffffffffu, v, d);
        if (lane >= d) v += prior;
    }
    return v;
}
// One warp scans a symbol segment in four stripes. Lane 31 carries the sum.
__global__ void simt_scan(const Changes* steps, const int* dst, const int* offsets, Rates* out, int segments) {
    int segment = blockIdx.x * 4 + threadIdx.x / 32;
    if (segment >= segments) return;
    int lane = threadIdx.x % 32;
    int start = offsets[segment], length = offsets[segment+1] - start;
    Changes carry{};
    for (int stripe = 0; stripe < length; stripe += 32) {
        int row = stripe+lane < length ? dst[start+stripe+lane] : -1;
        Changes d = row >= 0 ? steps[row] : Changes{};
        Changes sum = add(carry, {warp_prefix(d.ask_volume, lane), warp_prefix(d.ask_filled, lane),
                                  warp_prefix(d.bid_volume, lane), warp_prefix(d.bid_filled, lane)});
        store(sum, row, out);
        carry = {__shfl_sync(0xffffffffu, (long long)sum.ask_volume, 31),
                 __shfl_sync(0xffffffffu, (long long)sum.ask_filled, 31),
                 __shfl_sync(0xffffffffu, (long long)sum.bid_volume, 31),
                 __shfl_sync(0xffffffffu, (long long)sum.bid_filled, 31)};
    }
}
// One CUDA C++ tile scans 128 changes with a shared-memory inclusive prefix sum.
__global__ void tile_scan(const Changes* steps, const int* dst, const int* offsets, Rates* out) {
    __shared__ Changes sums[segment_size];
    int col = threadIdx.x, i = offsets[blockIdx.x] + col;
    int row = i < offsets[blockIdx.x+1] ? dst[i] : -1;
    Changes v = row >= 0 ? steps[row] : Changes{};
    for (int stride = 1; stride < segment_size; stride *= 2) {
        sums[col] = v;
        __syncthreads();
        if (col >= stride) v = add(v, sums[col - stride]);
        __syncthreads();
    }
    store(v, row, out);
}
}  // namespace

struct GpuScan::Impl {
    Backend backend = Backend::Simt;
#ifdef HAVE_TRITON_AOT
    CUmodule module{};
    CUfunction function{};
#endif
    cudaStream_t stream{};
    Slot host{}, device{};
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
#ifdef HAVE_TRITON_AOT
        if (module) cuModuleUnload(module);
#endif
        if (device.offsets) cudaFree(device.offsets);
        if (host.offsets) cudaFreeHost(host.offsets);
        if (device.rates) cudaFree(device.rates);
        if (device.destinations) cudaFree(device.destinations);
        if (device.deltas) cudaFree(device.deltas);
        if (host.rates) cudaFreeHost(host.rates);
        if (host.destinations) cudaFreeHost(host.destinations);
        if (host.deltas) cudaFreeHost(host.deltas);
        if (stream) cudaStreamDestroy(stream);
    }
};

GpuScan::GpuScan(Backend backend, int device) : impl_(new Impl{}) {
    try {
        impl_->backend = backend;
        checked(cudaSetDevice(device));
        checked(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking));
        auto& h = impl_->host;
        auto& d = impl_->device;
        checked(cudaMallocHost(&h.deltas, capacity * sizeof(Changes)));
        checked(cudaMallocHost(&h.destinations, capacity * sizeof(int)));
        checked(cudaMallocHost(&h.rates, capacity * sizeof(Rates)));
        checked(cudaMalloc(&d.deltas, capacity * sizeof(Changes)));
        checked(cudaMalloc(&d.destinations, capacity * sizeof(int)));
        checked(cudaMalloc(&d.rates, capacity * sizeof(Rates)));
        checked(cudaMallocHost(&h.offsets, (capacity+1) * sizeof(int)));
        checked(cudaMalloc(&d.offsets, (capacity+1) * sizeof(int)));
        if (backend == Backend::Triton) {
#ifdef HAVE_TRITON_AOT
            if (cuModuleLoadData(&impl_->module, tile_cubin) != CUDA_SUCCESS ||
                cuModuleGetFunction(&impl_->function, impl_->module, tile_symbol) != CUDA_SUCCESS)
                throw std::runtime_error("cannot load the compiled Triton tile for this GPU");
#else
            throw std::runtime_error("rebuild with WITH_TRITON=1 to enable the AOT tile");
#endif
        }
    } catch (...) {
        delete impl_;
        throw;
    }
}
GpuScan::~GpuScan() { delete impl_; }
Slot GpuScan::buffers() const { return impl_->host; }
void GpuScan::scan(int segments, int rows) {
    auto h = impl_->host, d = impl_->device;
    auto s = impl_->stream;
    checked(cudaMemcpyAsync(d.deltas, h.deltas, rows*sizeof(Changes), cudaMemcpyHostToDevice, s));
    checked(cudaMemcpyAsync(d.destinations, h.destinations, rows*sizeof(int), cudaMemcpyHostToDevice, s));
    checked(cudaMemcpyAsync(d.offsets, h.offsets, (segments+1)*sizeof(int), cudaMemcpyHostToDevice, s));
    if (impl_->backend == Backend::Tile) {
        tile_scan<<<segments, segment_size, 0, s>>>(d.deltas, d.destinations, d.offsets, d.rates);
    } else if (impl_->backend == Backend::Simt) {
        simt_scan<<<(segments+3)/4, 128, 0, s>>>(d.deltas, d.destinations, d.offsets, d.rates, segments);
    }
#ifdef HAVE_TRITON_AOT
    else {
        void* args[] = {&d.deltas,&d.destinations,&d.offsets,&d.rates};
        auto status = cuLaunchKernel(impl_->function, segments,1,1, tile_threads,1,1,
                                     tile_shared, s, args, nullptr);
        if (status != CUDA_SUCCESS) throw std::runtime_error("Triton tile launch failed");
    }
#endif
    checked(cudaGetLastError());
    checked(cudaMemcpyAsync(h.rates, d.rates, rows*sizeof(Rates), cudaMemcpyDeviceToHost, s));
    checked(cudaStreamSynchronize(s));
}
