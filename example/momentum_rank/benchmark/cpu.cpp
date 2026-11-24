#include "../rank_cpu.hpp"

#include <chrono>

// Compile the reference with the host C++ compiler used by the production CPU CLI.
double time_cpu(const std::vector<momentum::Quote>& rows, std::vector<std::int64_t>& returns,
                std::vector<std::int64_t>& scores, int batch) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < batch; ++i) momentum::compute_ranks(rows, returns, scores);
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
}
