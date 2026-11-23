// Each block owns an 8-window by 128-order tile. Sixteen columns cooperate
// on each row, then reduce four int64 statistics through shared memory.
__global__ void tiled_rate_spread(const int* windows, double* rates, int rows) {
    __shared__ long long partial[4][8][16];
    int x = threadIdx.x, y = threadIdx.y;
    int row = blockIdx.x * 8 + y;
    long long av = 0, af = 0, bv = 0, bf = 0;
    if (row < rows) {
        const int* v = windows + row * 2 * width;
        const int* f = v + width;
        for (int col = x; col < width; col += 16) {
            if (v[col] > 0) { av += v[col]; af += f[col]; }
            if (v[col] < 0) { bv -= v[col]; bf += f[col]; }
        }
    }
    partial[0][y][x] = av; partial[1][y][x] = af;
    partial[2][y][x] = bv; partial[3][y][x] = bf;
    __syncthreads();
    for (int stride = 8; stride > 0; stride /= 2) {
        if (x < stride) {
            for (int k = 0; k < 4; ++k)
                partial[k][y][x] += partial[k][y][x + stride];
        }
        __syncthreads();
    }
    if (x == 0 && row < rows)
        store_rates(rates + 3 * row, partial[0][y][0], partial[1][y][0],
                    partial[2][y][0], partial[3][y][0]);
}
