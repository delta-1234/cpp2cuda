#pragma once

#include "rank_io.hpp"

#include <utility>

namespace momentum {

inline void rank_session(const std::vector<Quote>& rows, std::size_t first, std::size_t end,
                         std::vector<std::int64_t>& returns, std::vector<std::int64_t>& scores) {
    std::vector<std::pair<std::int64_t, std::size_t>> peers;
    for (auto i = first; i < end; ++i) {
        const auto& quote = rows[i];
        if (quote.prev_close_cent == 0 || quote.close_cent == 0) continue;
        returns[i] = (quote.close_cent - quote.prev_close_cent) * 10000 / quote.prev_close_cent;
        peers.emplace_back(returns[i], i);
    }
    std::sort(peers.begin(), peers.end());
    const auto count = static_cast<std::int64_t>(peers.size());
    for (std::int64_t lower = 0; lower < count;) {
        auto upper = lower + 1;
        while (upper < count && peers[upper].first == peers[lower].first) ++upper;
        // [lower, upper) shares the midpoint of its zero-based ranks.
        const auto score = count == 1 ? 0 : (lower + upper - count) * 1000000 / (count - 1);
        for (auto j = lower; j < upper; ++j) scores[peers[j].second] = score;
        lower = upper;
    }
}

inline void compute_ranks(const std::vector<Quote>& rows, std::vector<std::int64_t>& returns,
                          std::vector<std::int64_t>& scores) {
    returns.assign(rows.size(), missing);
    scores.assign(rows.size(), missing);
    for (std::size_t first = 0; first < rows.size();) {
        auto end = first + 1;
        while (end < rows.size() && rows[end].date == rows[first].date) ++end;
        rank_session(rows, first, end, returns, scores);
        first = end;
    }
}

} // namespace momentum
