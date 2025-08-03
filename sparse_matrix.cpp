#include "sparse_matrix.h"
SparseMatrixCSR::SparseMatrixCSR(int r, int c) : rows(r), cols(c) {
    row_ptr.resize(r + 1, 0);
}
void SparseMatrixCSR::addValue(int r, int c, double val) {
    // For simplicity, just append (not efficient for real CSR construction)
    values.push_back(val);
    col_indices.push_back(c);
    row_ptr[r + 1]++;
}
std::vector<double> SparseMatrixCSR::multiply(const std::vector<double> &x) const {
    std::vector<double> result(rows, 0.0);
    #pragma omp parallel for schedule(static,16384)
    for (int i = 0; i < rows; ++i) {
        for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            result[i] += values[idx] * x[col_indices[idx]];
        }
    }
    return result;
}

void spmv(const double *__restrict val,
          const int *__restrict col,
          const int *__restrict rowptr,
          const double *__restrict x,
          double *__restrict y,
          std::size_t N) {
    #pragma omp parallel for schedule(static,16384)
    for (std::size_t r = 0; r < N; ++r) {
        double sum = 0.0;
        for (int idx = rowptr[r]; idx < rowptr[r + 1]; ++idx) {
            sum += val[idx] * x[col[idx]];
        }
        y[r] = sum;
    }
}
