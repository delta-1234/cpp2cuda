#include "factor_replay.hpp"
#include "gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void throw_cuda_error(cudaError_t error) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
}

void write_double_or_na(std::ostream& output, double value) {
    if (std::isnan(value))
        output << "NA";
    else
        output << value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: replay simt|tile INPUT.csv OUTPUT.csv\n";
            return 1;
        }

        const std::string_view mode_arg = argv[1];
        gpu::Mode mode;
        if (mode_arg == "simt") {
            mode = gpu::Mode::Simt;
        } else if (mode_arg == "tile") {
            mode = gpu::Mode::Tile;
        } else {
            std::cerr << "Usage: replay simt|tile INPUT.csv OUTPUT.csv\n";
            return 1;
        }

        const char* input_path = argv[2];
        const char* output_path = argv[3];

        std::ifstream input(input_path);
        if (!input)
            throw std::runtime_error(std::string("cannot open ") + input_path);

        std::ofstream output(output_path);
        if (!output)
            throw std::runtime_error(std::string("cannot open ") + output_path);

        output << "time_ms,symbol,trade_count,ask_fill_rate,bid_fill_rate,alpha\n"
               << std::fixed << std::setprecision(8);
        if (!output)
            throw std::runtime_error("output write failed");

        gpu::Plan* plan_ptr = nullptr;
        const cudaError_t create_error =
            gpu::create_plan(&plan_ptr, mode, gpu::default_batch_capacity);
        if (create_error != cudaSuccess)
            throw_cuda_error(create_error);

        std::unique_ptr<gpu::Plan, decltype(&gpu::destroy_plan)> plan(
            plan_ptr, gpu::destroy_plan);

        constexpr std::size_t batch_capacity = gpu::default_batch_capacity;

        std::vector<gpu::InputRow> batch;
        std::vector<std::string> symbols;
        std::vector<gpu::OutputRow> results;

        batch.reserve(batch_capacity);
        symbols.reserve(batch_capacity);
        results.reserve(batch_capacity);

        auto flush = [&]() {
            if (batch.empty())
                return;

            results.resize(batch.size());

            const cudaError_t process_error =
                gpu::process_batch(plan.get(), batch.data(), batch.size(), results.data());
            if (process_error != cudaSuccess)
                throw_cuda_error(process_error);

            for (std::size_t i = 0; i < results.size(); ++i) {
                const gpu::OutputRow& row = results[i];

                output << row.time_ms << ',' << symbols[i] << ',' << row.trade_count << ',';
                write_double_or_na(output, row.ask_fill_rate);
                output << ',';
                write_double_or_na(output, row.bid_fill_rate);
                output << ',';
                write_double_or_na(output, row.alpha);
                output << '\n';
            }

            if (!output)
                throw std::runtime_error("output write failed");

            batch.clear();
            symbols.clear();
        };

        factor::replay(
            input,
            [&](const factor::Tick& tick,
                std::size_t trades,
                const factor::Window& window,
                const factor::Orders& orders) {
                if (batch.size() == batch_capacity)
                    flush();

                batch.emplace_back();
                symbols.push_back(tick.symbol);

                gpu::InputRow& row = batch.back();
                row.time_ms = tick.time_ms;
                row.trade_count = static_cast<std::int64_t>(trades);
                std::fill(row.features, row.features + gpu::feature_count, 0);

                std::size_t slot = 0;
                for (factor::OrderId id : window) {
                    if (slot >= gpu::window_slots)
                        break;

                    const factor::Order& order = orders.at(id);
                    row.features[slot] = order.side == 'S' ? order.volume : -order.volume;
                    row.features[gpu::window_slots + slot] = order.filled;
                    ++slot;
                }
            });

        flush();

        output.flush();
        if (!output)
            throw std::runtime_error("output write failed");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
