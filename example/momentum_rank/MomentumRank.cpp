#include "rank_cpu.hpp"

#include <exception>

using namespace momentum;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: momentum_rank <quotes.csv>\n";
        return 2;
    }
    try {
        const auto rows = read_quotes(argv[1]);
        std::vector<std::int64_t> returns, scores;
        compute_ranks(rows, returns, scores);
        write_values(rows, returns, scores);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
