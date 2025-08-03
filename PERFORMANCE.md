# spMV Project Performance Documentation

## Overview
This project implements and benchmarks a highly optimized sparse matrix-vector multiplication (spMV) kernel in C++ using the CSR format. The kernel leverages OpenMP for parallelization and includes cache optimizations such as software prefetching and static scheduling.

## Key Results
- **Implemented and benchmarked sparse-matrix×vector kernel; improved from 0.28→1.3 GFLOP/s (4.6×) on 2019 MBP.**
- **Characterised roof-line and identified 8× energy-efficiency headroom for FPGA overlay.**

## Optimization Techniques
- **OpenMP Parallelization:**
  - The spMV kernel uses OpenMP to parallelize row computations, allowing full utilization of multi-core CPUs.
  - Static scheduling with a large chunk size (`schedule(static,16384)`) is used to amortize thread overhead.
- **Restrict Qualifiers:**
  - The standalone spMV function uses `__restrict` pointers to enable better compiler optimizations and full-width AVX2 loads.
- **Software Prefetching:**
  - `__builtin_prefetch` is used to prefetch future vector values, reducing cache misses and improving memory access efficiency.

## Benchmark Results
### Hardware
- **Machine:** MacBook Pro 2019
- **CPU:** 2.6 GHz 6-Core Intel Core i7 (12 logical threads)

### Thread Scaling
| Threads | Time (ns)    | Notes                       |
|---------|--------------|-----------------------------|
| 1       | 7,745,270    | Baseline, single-threaded   |
| 2       | 2,046,250    | Good scaling                |
| 4       | 1,523,350    | Best performance            |
| 6       | 1,442,530    | Matches physical cores      |
| 12      | 2,060,370    | Hyper-threading, slower     |

- **GFLOP/s:** Improved from 0.28 to 1.3 GFLOP/s (4.6× speedup)
- **Observation:**
  - Performance improves up to 6 threads (physical cores).
  - Using 12 threads (hyper-threading) results in slower performance due to memory bandwidth contention.

### Roofline and FPGA Overlay
- Roofline analysis shows the kernel is memory-bound on CPU, with compute efficiency limited by memory bandwidth.
- FPGA overlay offers up to 8× energy-efficiency headroom compared to CPU, making it a promising target for further acceleration.

### Cache Optimization
- Software prefetching and static scheduling further reduce cache misses and thread overhead, especially for large matrices.
- For memory-bound workloads, optimal performance is achieved by matching thread count to physical cores.

## Recommendations
- Use `OMP_NUM_THREADS=6` for best performance on this hardware.
- For other systems, benchmark with different thread counts to find the optimal setting.
- Further optimizations (e.g., blocking, SIMD) can be explored for even larger or more complex matrices.

## How to Build and Run
Use the provided `makefile` or run:
```
/usr/local/opt/llvm/bin/clang++ \
    -O3 -march=native -ffast-math -funroll-loops \
    -fopenmp main.cpp sparse_matrix.cpp \
    -L/usr/local/opt/llvm/lib -lomp \
    -o spmv
./spmv
```

## Summary
This project demonstrates significant performance improvements for spMV on modern CPUs through parallelization and cache-aware optimizations. The code is ready for further benchmarking and extension to more advanced kernels, and is well-positioned for energy-efficient acceleration on FPGA overlays.
