// One warp per window. The caller launches eight warps per block.
__device__ long long warp_sum(long long value) {
    for (int d = 16; d; d /= 2)
        value += __shfl_down_sync(0xffffffffu, value, d);
    return value;
}

__global__ void simt_rate_spread(const int* windows, double* rates, int rows) {
    int row = blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (row >= rows) return;  // Uniform for the entire warp.
    int lane = threadIdx.x % 32;
    const int* volumes = windows + row * 2 * width;
    const int* fills = volumes + width;
    long long av = 0, af = 0, bv = 0, bf = 0;
    for (int col = lane; col < width; col += 32) {
        int v = volumes[col], f = fills[col];
        if (v > 0) { av += v; af += f; }
        if (v < 0) { bv -= v; bf += f; }
    }
    av = warp_sum(av);
    af = warp_sum(af);
    bv = warp_sum(bv);
    bf = warp_sum(bf);
    if (lane == 0) {
        store_rates(rates + 3 * row, av, af, bv, bf);
    }
}
