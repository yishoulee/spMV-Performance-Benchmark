all:
	/usr/local/opt/llvm/bin/clang++ \
    -O3 -march=native -ffast-math -funroll-loops \
    -fopenmp main.cpp sparse_matrix.cpp \
    -L/usr/local/opt/llvm/lib -lomp \
    -o spmv
	./spmv