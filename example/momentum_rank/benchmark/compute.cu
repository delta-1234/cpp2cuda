// Reuse the complete, verified Pi launch path, including CUB's internal kernels.
#define main momentum_replay_cli
#include "../generated/replay.cu"
#undef main

#include <chrono>
#include <cuda_profiler_api.h>
#include <iomanip>

double time_cpu(const std::vector<Quote>& rows, std::vector<std::int64_t>& returns,
                std::vector<std::int64_t>& scores, int batch);

int main(int argc, char** argv) {
    if (argc != 5 || (std::string(argv[1]) != "cpu" && std::string(argv[1]) != "gpu")) {
        std::cerr << "Usage: replay cpu|gpu INPUT.csv WARMUP REPEAT\n";
        return 2;
    }
    try {
        const auto rows = momentum::read_quotes(argv[2]);
        const int warmup = std::stoi(argv[3]);
        const int repeats = std::stoi(argv[4]);
        if (rows.empty() || warmup < 1 || repeats < 1)
            throw std::runtime_error("need nonempty input, positive warmup and repeat counts");

        std::vector<std::int64_t> ref_returns, ref_scores;
        if (std::string(argv[1]) == "cpu") {
            using Clock = std::chrono::steady_clock;
            const auto ready = Clock::now() + std::chrono::milliseconds(50);
            for (int i = 0; i < warmup || Clock::now() < ready; ++i)
                time_cpu(rows, ref_returns, ref_scores, 1);
            int batch = 1;
            while (time_cpu(rows, ref_returns, ref_scores, batch) < 500.0) batch *= 2;
            std::vector<double> samples;
            for (int i = 0; i < repeats; ++i)
                samples.push_back(time_cpu(rows, ref_returns, ref_scores, batch) / batch);
            std::cout << batch << '\n';
            for (double us : samples) std::cout << std::setprecision(12) << us << '\n';
            return 0;
        }

        time_cpu(rows, ref_returns, ref_scores, 1);
        std::vector<std::int64_t> returns(rows.size()), scores(rows.size());
        for (int i = -warmup; i < repeats; ++i) {
            if (i == 0) cuda_check(cudaProfilerStart(), "start capture");
            for (int j = 0; j < 2; ++j) {
                const auto mode = (i + j) % 2 == 0 ? Mode::Simt : Mode::Tile;
                launch_all(rows, mode, returns, scores);
                if (returns != ref_returns || scores != ref_scores)
                    throw std::runtime_error("GPU result differs from the CPU reference");
            }
        }
        cuda_check(cudaProfilerStop(), "stop capture");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
