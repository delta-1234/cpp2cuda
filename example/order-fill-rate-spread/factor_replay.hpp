#pragma once
#include <array>
#include <charconv>
#include <cstdint>
#include <deque>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace factor {
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

inline Tick parse_tick(std::string_view line) {
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
    bool active = true;
};

inline Order& trade_order(std::unordered_map<OrderId, Order>& orders, OrderId id, char side, const Tick& tick) {
    auto it = orders.find(id);
    if (it == orders.end() || it->second.side != side)
        throw std::runtime_error("unknown order or incorrect side: " + std::to_string(id));
    Order& order = it->second;
    if (tick.volume > order.volume - order.filled)
        throw std::runtime_error("trade exceeds remaining volume: " + std::to_string(id));
    if ((side == 'B' && tick.price_cent > order.price_cent) ||
        (side == 'S' && tick.price_cent < order.price_cent))
        throw std::runtime_error("trade violates limit price: " + std::to_string(id));
    return order;
}

class OrderFillRateSpread {
    // Keep the session's orders so fills can reference IDs that have left the window.
    std::unordered_map<OrderId, Order> orders_;
    std::deque<OrderId> window_;
    std::size_t trades_ = 0;

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

        Order& bid = trade_order(orders_, tick.bid_order, 'B', tick);
        Order& ask = trade_order(orders_, tick.ask_order, 'S', tick);
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

inline void write_forecast(std::ostream& output, const Tick& tick, std::size_t trades,
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

template<class Visit>
void read_ticks(std::istream& input, Visit visit) {
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
                visit(tick);
                last_time = tick.time_ms;
            } catch (const std::runtime_error& error) {
                throw std::runtime_error("CSV line " + std::to_string(line_number) + ": " + error.what());
            }
        }
    }
    if (input.bad() || line_number == 0)
        throw std::runtime_error("unreadable or empty input");
}

template<class Emit>
void replay(std::istream& input, Emit emit) {
    std::unordered_map<std::string, OrderFillRateSpread> factors;
    read_ticks(input, [&](const Tick& tick) { factors[tick.symbol].update(tick, emit); });
}

}  // namespace factor
