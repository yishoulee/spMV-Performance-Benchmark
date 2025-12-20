#include "sparse_matrix.h"

#ifndef SPMV_ENABLE_PREFETCH
#define SPMV_ENABLE_PREFETCH 1
#endif

// Prefetch distance in *inner-loop iterations*.
// Tune via: -DSPMV_PREFETCH_DISTANCE=32 (or 0 to effectively disable).
#ifndef SPMV_PREFETCH_DISTANCE
#define SPMV_PREFETCH_DISTANCE 16
#endif

#if defined(__clang__) || defined(__GNUC__)
static inline void prefetch_read_l1(const void* ptr) {
#if SPMV_ENABLE_PREFETCH
    __builtin_prefetch(ptr, 0, 3);
#else
    (void)ptr;
#endif
}
#else
static inline void prefetch_read_l1(const void*) {}
#endif

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
            const int pref_idx = idx + SPMV_PREFETCH_DISTANCE;
            if (SPMV_ENABLE_PREFETCH && SPMV_PREFETCH_DISTANCE > 0 && pref_idx < row_ptr[i + 1]) {
                const int pref_col = col_indices[pref_idx];
                prefetch_read_l1(&x[pref_col]);
                prefetch_read_l1(&values[pref_idx]);
                prefetch_read_l1(&col_indices[pref_idx]);
            }
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
            const int pref_idx = idx + SPMV_PREFETCH_DISTANCE;
            if (SPMV_ENABLE_PREFETCH && SPMV_PREFETCH_DISTANCE > 0 && pref_idx < rowptr[r + 1]) {
                const int pref_col = col[pref_idx];
                prefetch_read_l1(&x[pref_col]);
                prefetch_read_l1(&val[pref_idx]);
                prefetch_read_l1(&col[pref_idx]);
            }
            sum += val[idx] * x[col[idx]];
        }
        y[r] = sum;
    }
}
