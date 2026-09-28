# spMV Project Performance Documentation

## Overview
This project implements and benchmarks a sparse matrix-vector multiplication (spMV) kernel in C++ using the CSR format. The kernel uses OpenMP parallelization, static scheduling, restrict-qualified pointers, and optional software prefetching.

## Key Result
- **Measured on a 2019 MacBook Pro:** improved from **0.28 to 1.3 GFLOP/s**, a **4.6× speedup**.

## Optimization Techniques
- **OpenMP parallelization** across matrix rows
- **Static scheduling** to reduce scheduling overhead
- **Restrict-qualified pointers** to help compiler optimization
- **Optional software prefetching** for upcoming sparse-vector accesses

## Benchmark Hardware
- **Machine:** MacBook Pro 2019
- **CPU:** 2.6 GHz 6-Core Intel Core i7 (12 logical threads)

## Thread Scaling
| Threads | Time (ns) | Notes |
| ---: | ---: | --- |
| 1 | 7,745,270 | Baseline |
| 2 | 2,046,250 | Strong scaling |
| 4 | 1,523,350 | Near best |
| 6 | 1,442,530 | Best measured result |
| 12 | 2,060,370 | Slower from contention |

- **GFLOP/s:** 0.28 → 1.3
- **Speedup:** 4.6×
- Performance improved through the six physical cores, then fell when using all 12 logical threads.

## Interpretation
The benchmark is consistent with a memory-bound sparse workload: adding useful CPU parallelism helped until memory-system contention dominated.

The repository does **not** claim a measured FPGA energy-efficiency advantage. FPGA offload remains a possible follow-on experiment rather than a demonstrated result here.

## How to Build and Run

Using the supplied makefile:

```bash
make
```

Or directly with Clang/OpenMP:

```bash
/usr/local/opt/llvm/bin/clang++ \
  -O3 -march=native -ffast-math -funroll-loops \
  -fopenmp main.cpp sparse_matrix.cpp \
  -L/usr/local/opt/llvm/lib -lomp \
  -o spmv
./spmv
```

Prefetch tuning is available through:

```bash
-DSPMV_ENABLE_PREFETCH=0
-DSPMV_PREFETCH_DISTANCE=32
```

## Scope
This is a focused CPU performance-characterization project. Its strongest evidence is the measured thread-scaling result and the 4.6× improvement on the stated machine.
