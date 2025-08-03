#include <iostream>
#include <vector>
#include "sparse_matrix.h"
#include <chrono>
#include <omp.h>

int main() {
    int rows = 1000000, cols = 1000000;
    SparseMatrixCSR mat(rows, cols);
    // Sparse identity matrix: only diagonal elements
    for (int i = 0; i < rows; ++i) {
        mat.addValue(i, i, 1.0);
    }
    // Fix row_ptr for CSR
    for (int i = 1; i <= rows; ++i) {
        mat.row_ptr[i] += mat.row_ptr[i - 1];
    }
    std::vector<double> x(rows);
    for (int i = 0; i < rows; ++i) x[i] = i + 1;
    std::vector<int> thread_counts = {1, 2, 4, 6, 12};
    for (int tc : thread_counts) {
        omp_set_num_threads(tc);
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> y = mat.multiply(x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> elapsed = end - start;
        std::cout << "Threads: " << tc << ", Time (ns): " << elapsed.count() << std::endl;
        if (tc == thread_counts[0]) {
            std::cout << "Result (first 10): ";
            for (int i = 0; i < 10; ++i) std::cout << y[i] << " ";
            std::cout << "...\n";
            std::cout << "nnz: " << mat.values.size() << std::endl;
        }
    }
    std::cout << "Max OpenMP threads: " << omp_get_max_threads() << std::endl;
    // KPIs can be calculated here
    return 0;
}
