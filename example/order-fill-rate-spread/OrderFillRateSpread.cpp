#include "factor_replay.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

using namespace factor;

// Row format shared by both CUDA backends: 128 signed volumes, then 128 fills.
void pack_windows(const char* input_path, const char* window_path, const char* key_path) {
    std::ifstream input(input_path);
    if (!input)
        throw std::runtime_error(std::string("cannot open ") + input_path);
    std::ofstream windows(window_path, std::ios::binary);
    std::ofstream keys(key_path);
    if (!windows || !keys)
        throw std::runtime_error("cannot create packed output");
    keys << "time_ms,symbol,trade_count\n";
    replay(input, [&](const Tick& tick, std::size_t trades, const Window& window, const Orders& orders) {
        std::array<std::int32_t, 256> row{};
        std::size_t i = 0;
        for (OrderId id : window) {
            const Order& order = orders.at(id);
            row[i] = order.side == 'S' ? order.volume : -order.volume;
            row[128 + i] = order.filled;
            ++i;
        }
        windows.write(reinterpret_cast<const char*>(row.data()), sizeof(row));
        keys << tick.time_ms << ',' << tick.symbol << ',' << trades << '\n';
    });
    windows.close();
    keys.close();
    if (!windows || !keys)
        throw std::runtime_error("packed output write failed");
}

struct RestingOrder {
    OrderId id;
    int price_cent;
    int remaining;
};

struct Book {
    std::map<int,std::deque<RestingOrder>> bids, asks;

    void submit(std::ostream& output, int time, const char* symbol, bool buy, RestingOrder incoming) {
        auto emit = [&](char type, int price, int volume, OrderId bid, OrderId ask) {
            output << time << ',' << symbol << ',' << type << ',' << price << ','
                   << volume << ',' << bid << ',' << ask << '\n';
        };
        emit(buy ? 'B' : 'S', incoming.price_cent, incoming.remaining,
             buy ? incoming.id : 0, buy ? 0 : incoming.id);

        auto& opposite = buy ? asks : bids;
        while (incoming.remaining > 0 && !opposite.empty()) {
            auto level = buy ? opposite.begin() : std::prev(opposite.end());
            auto& queue = level->second;
            auto& best = queue.front();
            if (buy ? incoming.price_cent < best.price_cent : incoming.price_cent > best.price_cent)
                break;
            int quantity = std::min(incoming.remaining, best.remaining);
            emit('T', best.price_cent, quantity, buy ? incoming.id : best.id,
                 buy ? best.id : incoming.id);
            incoming.remaining -= quantity;
            best.remaining -= quantity;
            if (best.remaining == 0) queue.pop_front();
            if (queue.empty()) opposite.erase(level);
        }

        if (incoming.remaining > 0) {
            auto& own = buy ? bids : asks;
            own[incoming.price_cent].push_back(incoming);
        }
    }
};

void generate(std::ostream& output, std::uint32_t seed, int order_count = 2600, bool scaled = false) {
    std::mt19937 rng(seed);
    auto draw = [&](int low, int high) { return std::uniform_int_distribution<int>(low, high)(rng); };
    std::array<Book, 2> books;
    const std::array<const char*, 2> symbols{"000001.SZ", "000002.SZ"};
    const std::array<int, 2> reference_prices{1000, 1500};
    int elapsed = 0;
    constexpr int open = (9 * 60 + 30) * 60 * 1000;
    output << tick_header << '\n';
    for (OrderId id = 1; id <= order_count; ++id) {
        elapsed += scaled ? draw(0, 6) : draw(10, 150);
        if (elapsed >= 4 * 3600000) throw std::runtime_error("generated orders exceed one trading day");
        int time = open + elapsed + (elapsed >= 2 * 3600000 ? 90 * 60000 : 0);
        auto stock = static_cast<std::size_t>(draw(0, 1));
        bool buy = draw(0, 1) == 0;
        int price = reference_prices[stock] + draw(-15, 15);
        int volume = draw(1, 12) * 100;
        books[stock].submit(output, time, symbols[stock], buy, {id, price, volume});
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string_view(argv[1]) == "--generate-scale") {
            generate(std::cout, integer_field<std::uint32_t>(argv[2], "seed"),
                     integer_field<int>(argv[3], "order count"), true);
            std::cout.flush();
            if (!std::cout) throw std::runtime_error("output write failed");
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "--pack") {
            pack_windows(argv[2], argv[3], argv[4]);
            return 0;
        }
        bool generating = argc > 1 && std::string_view(argv[1]) == "--generate";
        if ((generating && argc > 3) || (!generating && argc > 2)) {
            std::cerr << "Usage: " << argv[0] << " [ticks.csv] | --generate [seed] | --generate-scale seed orders | --pack ticks.csv windows.bin keys.csv\n";
            return 1;
        }
        if (generating) {
            auto seed = argc == 3 ? integer_field<std::uint32_t>(argv[2], "seed") : 8001u;
            generate(std::cout, seed);
        } else {
            const char* path = argc == 2 ? argv[1] : "sample_ticks.csv";
            std::ifstream input(path);
            if (!input)
                throw std::runtime_error(std::string("cannot open ") + path);
            std::cout << "time_ms,symbol,trade_count,ask_fill_rate,bid_fill_rate,alpha\n"
                      << std::fixed << std::setprecision(8);
            replay(input, [](const Tick& tick, std::size_t trades, const Window& window, const Orders& orders) {
                write_forecast(std::cout, tick, trades, window, orders);
            });
        }
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("output write failed");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
