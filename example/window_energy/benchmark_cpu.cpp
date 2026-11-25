// In-memory version of the scalar reference in window_energy.cpp.
// stdout has the same integer output as the reference; stderr contains samples in us.
#include <atomic>
#include <charconv>
#include <chrono>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    const std::string_view arg(argv[1]);
    int n = 0;
    const auto end = arg.data() + arg.size();
    const auto parsed = std::from_chars(arg.data(), end, n);
    if (parsed.ec != std::errc{} || parsed.ptr != end || n < 1 || n > 1000000) return 1;

    using Clock = std::chrono::steady_clock;
    std::vector<int> values(n);
    auto compute = [&] {
        for (int i = 0; i < n; ++i) {
            int energy = 0;
            for (int j = i - 2; j <= i + 2; ++j) {
                if (j >= 0 && j < n) {
                    const int x = j % 17 - 8;
                    energy += x * x;
                }
            }
            values[i] = energy;
        }
        std::atomic_signal_fence(std::memory_order_seq_cst);
    };

    const auto deadline = Clock::now() + std::chrono::milliseconds(50);
    do {
        compute();
    } while (Clock::now() < deadline);
    double samples[7];
    for (double& us : samples) {
        const auto start = Clock::now();
        compute();
        us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    }
    for (double us : samples) std::cerr << us << '\n';
    for (int energy : values) std::cout << energy << '\n';
}
