#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    const int n = std::atoi(argv[1]);
    if (n < 1 || n > 1000000) return 1;
    for (int i = 0; i < n; ++i) {
        int energy = 0;
        for (int j = i - 2; j <= i + 2; ++j) {
            if (j >= 0 && j < n) {
                const int x = j % 17 - 8;
                energy += x * x;
            }
        }
        std::cout << energy << '\n';
    }
}
