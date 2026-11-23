#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

constexpr std::size_t order_window = 100;
constexpr std::size_t trade_warmup = 500;  // Inferred from the screenshot.
constexpr std::string_view tick_header =
    "time_ms,symbol,type,price_cent,volume,bid_order,ask_order";
using OrderId = std::int64_t;

struct Tick {
    int time_ms;
    std::string symbol;
    char type;
    int price_cent;
    int volume;
    OrderId bid_order;
    OrderId ask_order;
};

template<class Integer>
Integer integer_field(std::string_view field, const char* name) {
    Integer value{};
    auto end = field.data() + field.size();
    auto parsed = std::from_chars(field.data(), end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value < 0)
        throw std::runtime_error(std::string("invalid ") + name);
    return value;
}

Tick parse_tick(std::string_view line) {
    std::array<std::string_view, 7> fields;
    for (std::size_t i = 0; i + 1 < fields.size(); ++i) {
        auto comma = line.find(',');
        if (comma == line.npos)
            throw std::runtime_error("expected seven CSV columns");
        fields[i] = line.substr(0, comma);
        line.remove_prefix(comma + 1);
    }
    if (line.find(',') != line.npos)
        throw std::runtime_error("expected seven CSV columns");
    fields.back() = line;

    if (fields[1].empty() || fields[2].size() != 1)
        throw std::runtime_error("invalid symbol or event type");
    Tick tick{integer_field<int>(fields[0], "time_ms"), std::string(fields[1]), fields[2][0],
              integer_field<int>(fields[3], "price_cent"), integer_field<int>(fields[4], "volume"),
              integer_field<OrderId>(fields[5], "bid_order"), integer_field<OrderId>(fields[6], "ask_order")};
    if (tick.time_ms >= 86'400'000 || tick.price_cent == 0 || tick.volume == 0)
        throw std::runtime_error("time outside day or zero price/volume");
    if (!((tick.type == 'B' && tick.bid_order > 0 && tick.ask_order == 0) ||
          (tick.type == 'S' && tick.ask_order > 0 && tick.bid_order == 0) ||
          (tick.type == 'T' && tick.bid_order > 0 && tick.ask_order > 0)))
        throw std::runtime_error("event type and order references disagree");
    return tick;
}

struct Order {
    char side;
    int price_cent;
    int volume;
    int filled = 0;
};

class OrderFillRateSpread {
    // Keep the session's orders so fills can reference IDs that have left the window.
    std::unordered_map<OrderId, Order> orders_;
    std::deque<OrderId> window_;
    std::size_t trades_ = 0;

    Order& trade_order(OrderId id, char side, const Tick& tick) {
        auto it = orders_.find(id);
        if (it == orders_.end() || it->second.side != side)
            throw std::runtime_error("unknown order or incorrect side: " + std::to_string(id));
        Order& order = it->second;
        if (tick.volume > order.volume - order.filled)
            throw std::runtime_error("trade exceeds remaining volume: " + std::to_string(id));
        if ((side == 'B' && tick.price_cent > order.price_cent) ||
            (side == 'S' && tick.price_cent < order.price_cent))
            throw std::runtime_error("trade violates limit price: " + std::to_string(id));
        return order;
    }

public:
    template<class Emit>
    void update(const Tick& tick, Emit& emit) {
        if (tick.type != 'T') {
            OrderId id = tick.type == 'B' ? tick.bid_order : tick.ask_order;
            if (!orders_.emplace(id, Order{tick.type, tick.price_cent, tick.volume}).second)
                throw std::runtime_error("duplicate order: " + std::to_string(id));
            window_.push_back(id);
            if (window_.size() > order_window)
                window_.pop_front();
            return;
        }

        Order& bid = trade_order(tick.bid_order, 'B', tick);
        Order& ask = trade_order(tick.ask_order, 'S', tick);
        bid.filled += tick.volume;
        ask.filled += tick.volume;
        ++trades_;
        if (window_.size() < order_window || trades_ < trade_warmup)
            return;

        emit(tick, trades_, window_, orders_);
    }
};

using Window = std::deque<OrderId>;
using Orders = std::unordered_map<OrderId, Order>;

void write_forecast(std::ostream& output, const Tick& tick, std::size_t trades,
                    const Window& window, const Orders& orders) {
    struct Totals { std::int64_t volume = 0, filled = 0; };
    Totals bids, asks;
    for (OrderId id : window) {
        const Order& order = orders.at(id);
        Totals& totals = order.side == 'B' ? bids : asks;
        totals.volume += order.volume;
        totals.filled += order.filled;
    }

    output << tick.time_ms << ',' << tick.symbol << ',' << trades << ',';
    auto print_rate = [&](const Totals& side) {
        if (side.volume == 0)
            output << "NA";
        else
            output << static_cast<double>(side.filled) / side.volume;
        output << ',';
    };
    print_rate(asks);
    print_rate(bids);
    if (asks.volume == 0 || bids.volume == 0)
        output << "NA\n";
    else
        output << static_cast<double>(asks.filled) / asks.volume
                    - static_cast<double>(bids.filled) / bids.volume << '\n';
}

template<class Emit>
void replay(std::istream& input, Emit emit) {
    std::unordered_map<std::string, OrderFillRateSpread> factors;
    std::string line;
    int last_time = -1;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line_number == 1) {
            if (line != tick_header)
                throw std::runtime_error("unexpected tick CSV header");
        } else {
            try {
                Tick tick = parse_tick(line);
                if (tick.time_ms < last_time)
                    throw std::runtime_error("timestamps out of order");
                factors[tick.symbol].update(tick, emit);
                last_time = tick.time_ms;
            } catch (const std::runtime_error& error) {
                throw std::runtime_error("CSV line " + std::to_string(line_number) + ": " + error.what());
            }
        }
    }
    if (input.bad() || line_number == 0)
        throw std::runtime_error("unreadable or empty input");
}

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
    std::deque<RestingOrder> bids, asks;

    void submit(std::ostream& output, int time, const char* symbol, bool buy, RestingOrder incoming) {
        auto emit = [&](char type, int price, int volume, OrderId bid, OrderId ask) {
            output << time << ',' << symbol << ',' << type << ',' << price << ','
                   << volume << ',' << bid << ',' << ask << '\n';
        };
        emit(buy ? 'B' : 'S', incoming.price_cent, incoming.remaining,
             buy ? incoming.id : 0, buy ? 0 : incoming.id);

        auto& opposite = buy ? asks : bids;
        while (incoming.remaining > 0 && !opposite.empty()) {
            auto& best = opposite.front();
            if (buy ? incoming.price_cent < best.price_cent : incoming.price_cent > best.price_cent)
                break;
            int quantity = std::min(incoming.remaining, best.remaining);
            emit('T', best.price_cent, quantity, buy ? incoming.id : best.id,
                 buy ? best.id : incoming.id);
            incoming.remaining -= quantity;
            best.remaining -= quantity;
            if (best.remaining == 0)
                opposite.pop_front();
        }

        if (incoming.remaining > 0) {
            auto& own = buy ? bids : asks;
            // Insert after all orders with better or equal prices (FIFO at each price).
            auto pos = std::find_if(own.begin(), own.end(), [&](const RestingOrder& resting) {
                return buy ? resting.price_cent < incoming.price_cent : resting.price_cent > incoming.price_cent;
            });
            own.insert(pos, incoming);
        }
    }
};

void generate(std::ostream& output, std::uint32_t seed) {
    std::mt19937 rng(seed);
    auto draw = [&](int low, int high) { return std::uniform_int_distribution<int>(low, high)(rng); };
    std::array<Book, 2> books;
    const std::array<const char*, 2> symbols{"000001.SZ", "000002.SZ"};
    const std::array<int, 2> reference_prices{1000, 1500};
    int time = (9 * 60 + 30) * 60 * 1000;
    output << tick_header << '\n';
    for (OrderId id = 1; id <= 2600; ++id) {
        time += draw(10, 150);
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
        if (argc == 5 && std::string_view(argv[1]) == "--pack") {
            pack_windows(argv[2], argv[3], argv[4]);
            return 0;
        }
        bool generating = argc > 1 && std::string_view(argv[1]) == "--generate";
        if ((generating && argc > 3) || (!generating && argc > 2)) {
            std::cerr << "Usage: " << argv[0] << " [sample_ticks.csv] | --generate [seed]\n";
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
