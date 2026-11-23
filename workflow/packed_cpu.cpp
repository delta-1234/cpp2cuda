// CPU timing for the same packed batch and outputs as the GPU kernels.
// Allocation, file I/O and CSV conversion stay outside the compute timer.
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: packed_cpu windows.bin rates.bin");
        std::ifstream input(argv[1], std::ios::binary);
        std::ofstream output(argv[2], std::ios::binary);
        if (!input || !output) throw std::runtime_error("cannot open input/output");
        constexpr int width = 128, batch = 4096;
        std::vector<std::int32_t> data(batch * 2 * width);
        std::vector<double> rates(batch * 3);
        std::chrono::duration<double, std::milli> elapsed{};
        while (true) {
            input.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(data[0]));
            auto bytes = input.gcount();
            constexpr int row_bytes = 2 * width * sizeof(data[0]);
            if (input.bad() || bytes % row_bytes) throw std::runtime_error("invalid packed input");
            if (!bytes) break;
            int rows = static_cast<int>(bytes / row_bytes);
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < rows; ++i) {
                const auto* v = data.data() + 2 * width * i;
                const auto* f = v + width;
                std::int64_t av = 0, bv = 0, af = 0, bf = 0;
                for (int j = 0; j < width; ++j) {
                    if (v[j] > 0) { av += v[j]; af += f[j]; }
                    if (v[j] < 0) { bv -= v[j]; bf += f[j]; }
                }
                double ask = av ? static_cast<double>(af) / av : -2;
                double bid = bv ? static_cast<double>(bf) / bv : -2;
                rates[3*i] = ask;
                rates[3*i+1] = bid;
                rates[3*i+2] = av && bv ? ask - bid : -2;
            }
            elapsed += std::chrono::steady_clock::now() - start;
            output.write(reinterpret_cast<const char*>(rates.data()), rows * 3 * sizeof(double));
        }
        output.close();
        if (!output) throw std::runtime_error("output write failed");
        std::cout << elapsed.count() << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
