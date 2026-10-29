# C reference benchmark

Times the upstream C implementations (`lapjv.cpp` and `lapmod.cpp` from
gatagat/lap 0.5.13) on the same deterministic LCG matrices as the Go
benchmarks in `bench_test.go`, for an apples-to-apples comparison. The sparse
harness also prints the assignment cost, which must match the Go solver's on
the same matrix.

The sparse harness takes `n per_row [span]`: without a span it builds the
banded layout of `BenchmarkLapmodSparse{1k,4k}`; with a span it builds the
random integer-cost layout of `BenchmarkLapmodSparsePath1k`, whose ties make
the solve reach the `find_path` augmentation phase.

```bash
./run.sh
```

First run downloads the lap 0.5.13 sdist from PyPI (~200 KB) and compiles
it with `-O2`. Requires a C compiler.
