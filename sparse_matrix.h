#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H
#include <vector>
class SparseMatrixCSR {
public:
    int rows, cols;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    SparseMatrixCSR(int r, int c);
    void addValue(int r, int c, double val);
    std::vector<double> multiply(const std::vector<double>& x) const;
    void spmv(const double *__restrict val,
              const int *__restrict col,
              const int *__restrict rowptr,
              const double *__restrict x,
              double *__restrict y,
              std::size_t N);
};
#endif // SPARSE_MATRIX_H
