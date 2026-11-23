#include "../factor_replay.hpp"
#include "scan.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {
using namespace factor;
using Clock = std::chrono::steady_clock;
constexpr char header[] = "time_ms,symbol,trade_count,ask_fill_rate,bid_fill_rate,alpha\n";

struct Key { int time; const std::string* symbol; std::size_t trade; };

void write_row(std::ostream& out, const Key& key, const Rates& rates) {
    // Both optimized paths use exactly the same formatter and file buffering.
    char buf[120];
    auto p = std::to_chars(buf, buf + sizeof(buf), key.time).ptr;
    *p++ = ',';
    out.write(buf, p-buf);
    out.write(key.symbol->data(), key.symbol->size());
    buf[0] = ',';
    p = std::to_chars(buf+1, buf+sizeof(buf), key.trade).ptr;
    for (double v : {rates.ask, rates.bid, rates.spread}) {
        *p++ = ',';
        if (v == -2.) { *p++ = 'N'; *p++ = 'A'; }
        else p = std::to_chars(p, buf+sizeof(buf), v, std::chars_format::fixed, 8).ptr;
    }
    *p++ = '\n';
    out.write(buf, p-buf);
}

Rates calculate(const Changes& s) {
    double a = s.ask_volume ? double(s.ask_filled)/s.ask_volume : -2.;
    double b = s.bid_volume ? double(s.bid_filled)/s.bid_volume : -2.;
    return {a,b,s.ask_volume && s.bid_volume ? a-b : -2.};
}

struct LiveSymbol {
    Orders orders;
    // References to unordered_map elements survive rehash; the ledger never erases.
    std::array<Order*, order_window> ring{};
    std::size_t inserted = 0, trades = 0;
    Changes changes{};
    std::size_t epoch = 0;
    int tile = -1, length = segment_size;

    static void include(Changes& d, const Order& o, int sign) {
        if (o.side == 'S') {
            d.ask_volume += sign * o.volume;
            d.ask_filled += sign * o.filled;
        } else {
            d.bid_volume += sign * o.volume;
            d.bid_filled += sign * o.filled;
        }
    }
    bool update(const Tick& tick) {
        if (tick.type != 'T') {
            OrderId id = tick.type == 'B' ? tick.bid_order : tick.ask_order;
            auto entry = orders.emplace(id, Order{tick.type, tick.price_cent, tick.volume});
            if (!entry.second) throw std::runtime_error("duplicate order: " + std::to_string(id));
            Order*& slot = ring[inserted % order_window];
            if (slot) { slot->active = false; include(changes, *slot, -1); }
            slot = &entry.first->second;
            include(changes, *slot, 1);
            ++inserted;
            return false;
        }
        Order& bid = trade_order(orders, tick.bid_order, 'B', tick);
        Order& ask = trade_order(orders, tick.ask_order, 'S', tick);
        bid.filled += tick.volume;
        ask.filled += tick.volume;
        if (bid.active) changes.bid_filled += tick.volume;
        if (ask.active) changes.ask_filled += tick.volume;
        ++trades;
        return trades >= trade_warmup && inserted >= order_window;
    }
    Changes anchor() const {
        Changes absolute{};
        for (const Order* order : ring) include(absolute, *order, 1);
        return absolute;
    }
};

struct Stats { std::size_t rows=0, batches=0, segments=0; };

class OutputBatch {
    GpuScan& gpu_;
    Slot slot_;
    std::ostream& out_;
    Stats& stats_;
    std::size_t epoch_ = 1;
    int segments_ = 0, rows_ = 0;
    std::array<int, capacity> lengths_{};
    std::array<int, capacity> row_segment_;
    std::array<int, capacity> cursor_;
    std::array<Key, capacity> keys_;
public:
    OutputBatch(GpuScan& gpu, std::ostream& out, Stats& stats)
        : gpu_(gpu), slot_(gpu.buffers()), out_(out), stats_(stats) {}
    void append(LiveSymbol& symbol, const Key& key) {
        if (rows_ == capacity) flush();
        bool first = symbol.epoch != epoch_ || symbol.length == segment_size;
        if (first) {
            symbol.epoch = epoch_;
            symbol.tile = segments_++;
            symbol.length = 0;
        }
        ++symbol.length;
        slot_.deltas[rows_] = first ? symbol.anchor() : symbol.changes;
        symbol.changes = {};
        lengths_[symbol.tile] = symbol.length;
        keys_[rows_] = key;
        row_segment_[rows_++] = symbol.tile;
    }
    void flush() {
        if (!rows_) return;
        // Reorder only the four-byte indices. Quantities and output stay in arrival order.
        int offset = 0;
        for (int s = 0; s < segments_; ++s) {
            slot_.offsets[s] = cursor_[s] = offset;
            offset += lengths_[s];
        }
        slot_.offsets[segments_] = offset;
        for (int row = 0; row < rows_; ++row)
            slot_.destinations[cursor_[row_segment_[row]]++] = row;
        gpu_.scan(segments_, rows_);
        for (int i=0; i<rows_; ++i) write_row(out_, keys_[i], slot_.rates[i]);
        ++stats_.batches;
        stats_.segments += segments_;
        ++epoch_;
        rows_ = segments_ = 0;
    }
};

Stats process(const std::string& source, const std::string& target, bool legacy, GpuScan* gpu) {
    std::ifstream input(source);
    if (!input) throw std::runtime_error("cannot open input " + source);
    std::error_code ec;
    if (std::filesystem::equivalent(source, target, ec)) throw std::runtime_error("input and output coincide");
    std::ofstream out(target);
    if (!out) throw std::runtime_error("cannot create output " + target);
    out << header;
    Stats stats;
    if (legacy) {
        out << std::fixed << std::setprecision(8);
        replay(input, [&](const Tick& t, std::size_t n, const Window& w, const Orders& o) {
            write_forecast(out,t,n,w,o);
            ++stats.rows;
        });
    } else {
        std::unordered_map<std::string,LiveSymbol> symbols;
        std::unique_ptr<OutputBatch> batch;
        if (gpu) batch = std::make_unique<OutputBatch>(*gpu,out,stats);
        read_ticks(input,[&](const Tick& t) {
            auto it = symbols.try_emplace(t.symbol).first;
            auto& state = it->second;
            if (!state.update(t)) return;
            Key key{t.time_ms,&it->first,state.trades};
            if (batch) batch->append(state,key);
            else write_row(out,key,calculate(state.changes));
            ++stats.rows;
        });
        if (batch) batch->flush();
    }
    out.close();
    if (!out) throw std::runtime_error("output write failed: " + target);
    return stats;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 3)
            throw std::runtime_error("usage: replay cpu|legacy|simt|tile|triton INPUT OUTPUT, or replay MODE --serve");
        std::string mode = argv[1];
        bool serving = argc == 3 && std::string_view(argv[2]) == "--serve";
        if ((!serving && argc != 4) || (mode!="cpu" && mode!="legacy" && mode!="simt" && mode!="tile" && mode!="triton"))
            throw std::runtime_error("invalid mode or arguments");
        auto start = Clock::now();
        std::unique_ptr<GpuScan> gpu;
        if (mode=="simt") gpu = std::make_unique<GpuScan>(Backend::Simt,0);
        if (mode=="tile") gpu = std::make_unique<GpuScan>(Backend::Tile,0);
        if (mode=="triton") gpu = std::make_unique<GpuScan>(Backend::Triton,0);
        double init_ms = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        auto run = [&](const std::string& source, const std::string& target) {
            auto stats = process(source,target,mode=="legacy",gpu.get());
            std::cout << "OK\t" << stats.rows << '\t' << stats.batches << '\t' << stats.segments << '\t' << init_ms << std::endl;
        };
        if (serving) {
            std::cout << "READY\t" << init_ms << std::endl;
            std::string request;
            while (std::getline(std::cin,request)) {
                auto sep = request.find('\t');
                if (sep == request.npos) throw std::runtime_error("expected INPUT<TAB>OUTPUT");
                run(request.substr(0,sep), request.substr(sep+1));
            }
        } else {
            run(argv[2],argv[3]);
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
