#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace momentum {

constexpr std::int64_t missing = std::numeric_limits<std::int64_t>::min();

struct Quote {
    std::string date;
    std::string symbol;
    std::int64_t prev_close_cent;
    std::int64_t close_cent;
};

inline std::int64_t parse_price(const std::string& field) {
    if (field.empty() || field.size() > 9 ||
        !std::all_of(field.begin(), field.end(), [](char c) { return c >= '0' && c <= '9'; }))
        throw std::runtime_error("expected an integer price in cents (0..999999999)");
    return std::stoll(field);
}

inline std::vector<Quote> read_quotes(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input CSV");
    std::string line;
    auto next_line = [&]() {
        if (!std::getline(in, line)) return false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
    };
    if (!next_line() || line != "date,symbol,prev_close_cent,close_cent")
        throw std::runtime_error("unexpected CSV header");

    std::vector<Quote> rows;
    while (next_line()) {
        std::istringstream fields(line);
        std::string date, symbol, previous, close;
        if (!std::getline(fields, date, ',') || date.size() != 8 ||
            !std::all_of(date.begin(), date.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
            !std::getline(fields, symbol, ',') || symbol.empty() ||
            !std::getline(fields, previous, ',') || !std::getline(fields, close))
            throw std::runtime_error("malformed CSV row");
        if (!rows.empty() && rows.back().date > date)
            throw std::runtime_error("dates must be in ascending order");
        rows.push_back({date, symbol, parse_price(previous), parse_price(close)});
    }
    if (!in.eof()) throw std::runtime_error("error reading CSV");
    return rows;
}

inline void write_values(const std::vector<Quote>& rows,
                         const std::vector<std::int64_t>& returns,
                         const std::vector<std::int64_t>& scores) {
    std::cout << "date,symbol,return_bp,rank_ppm\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::cout << rows[i].date << ',' << rows[i].symbol << ',';
        if (returns[i] == missing) std::cout << "NA,NA\n";
        else std::cout << returns[i] << ',' << scores[i] << '\n';
    }
}

} // namespace momentum
