# go-lap

Go port of Python's [`lap`](https://github.com/gatagat/lap) (v0.5.13) for solving linear assignment problems using the Jonker-Volgenant algorithm.

This package includes both the dense (`Lapjv`) and sparse (`Lapmod`) solvers from the original library, commonly used in multi-object tracking (MOT) to match detections to tracklets. It aims for 1:1 behavioral parity with the Python/Cython upstream, producing the exact same assignments and tie-breaking results for identical inputs.

## Installation

```bash
go get github.com/PloyBox/go-lap
```

## Quick Start

```go
package main

import (
	"fmt"

	"github.com/PloyBox/go-lap"
)

func main() {
	// cost[i][j]: cost of assigning row i to column j
	cost := [][]float64{
		{1000, 2, 11},
		{6, 1000, 1},
		{5, 12, 1000},
	}

	// Drop assignments with a cost > 4.99; unmatched indices are set to -1
	res, err := lap.Lapjv(cost, lap.WithCostLimit(4.99))
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Cost) // 3
	fmt.Println(res.X)    // [1 2 -1]  row i -> col res.X[i] (-1 = unmatched)
	fmt.Println(res.Y)    // [-1 0 1]  col j -> row res.Y[j]
}
```

By default, `lap.Lapjv(cost)` works just like `lap.lapjv(cost)` in Python and returns a complete matching. If you have a non-square matrix, pass `lap.WithExtendCost(true)`—any unassigned rows or columns will be set to `-1`.

### Sparse Matrices (`Lapmod`)

`lap.Lapmod` ports `lap.lapmod` for square cost matrices in CSR format, storing only the finite entries:

```go
// cost[i][kk[ii[i]+k]] == cc[ii[i]+k]
// Note: kk must be sorted within each row.
res, err := lap.Lapmod(n, cc, ii, kk)
```

Just like upstream `lapmod`, it doesn't take `extend_cost` or `cost_limit` options directly. If you need them, you'll have to build the extended CSR matrix yourself (see how upstream's `arr-loop` test does it).

A few constraints to keep in mind for `Lapmod`:

- Stored costs must be in `[0, 1e6)`.
- Don't store `+Inf` entries; just omit them from the CSR arrays.
- If no feasible complete assignment exists, `res.Cost` will be `+Inf`.

## Python vs. Go API

**Dense Solver (`lapjv` -> `Lapjv`)**

| Python (`lap.lapjv`) | Go (`lap.Lapjv`) |
| --- | --- |
| `cost` (`N×N` `ndarray`) | `cost [][]float64` |
| `extend_cost=False` | `WithExtendCost(false)` (default) |
| `cost_limit=np.inf` | `WithCostLimit(math.Inf(1))` (default) |
| `return_cost=True` | `WithReturnCost(true)` (default) |
| Returns `(opt, x, y)` | Returns `Result{Cost, X, Y}` |
| Returns `(x, y)` when `return_cost=False` | `Result.Cost` is set to `NaN` |

**Sparse Solver (`lapmod` -> `Lapmod`)**

| Python (`lap.lapmod`) | Go (`lap.Lapmod`) |
| --- | --- |
| `n, cc, ii, kk` (CSR arrays) | `n int, cc []float64, ii, kk []int` |
| `fast=True` (Cython path) | Always enabled (only implementation) |
| `return_cost=True` | `WithSparseReturnCost(true)` (default) |
| `fp_version=FP_DYNAMIC` | `WithFPVersion(FPDynamic)` (default) |
| Returns `(opt, x, y)` | Returns `Result{Cost, X, Y}` |
| Raises `ValueError` on empty input | Returns `ErrNoRows` / `ErrNoCols` |

## Notes & Edge Cases

- **Square matrices**: Solved directly. `X` and `Y` are inverse permutations of each other.
- **Non-square matrices**: Requires `WithExtendCost(true)`, which pads the matrix with zeros up to `max(r, c)²` before solving. Without it, `Lapjv` returns `ErrNotSquare`.
- **Cost limits (`WithCostLimit(l)`)**: Internally expands the matrix to `(r+c)×(r+c)`, filling off-diagonal blocks with `l/2` and the bottom-right block with `0`. Assignments with `cost > l` are dropped (`-1`), `< l` are kept, and exact `= l` ties depend on solver tie-breaking.
- **`Inf` and negative costs**: Supported out of the box in `Lapjv`. Unsolvable matrices return `Cost = +Inf`.
- **Sparse bounds checking**: The original C code for `lapmod` doesn't do bounds checking and can read out of bounds on unsolvable inputs. This port validates that `kk` is strictly sorted per row, returns `Cost = +Inf` on unsolvable inputs, and safely marks remaining rows as `-1`.
- **Empty / malformed inputs**: Passing a `0×0` matrix to `Lapjv` returns `Cost = 0` with empty slices. Ragged 2D slices (rows of different lengths) return `ErrRagged`. `Lapmod` returns `ErrNoRows` or `ErrNoCols` on empty inputs, matching upstream's `check_cost`.

## Testing & Verification

To make sure the port is solid, it's tested against:

- All fixed test cases ported directly from the upstream repo (matching assignments and tie-breakers).
- 700 brute-force reference cases for `Lapjv` and 300 for `Lapmod`.
- Optimality certificate checks for `n = 40, 80, 150` (dense) and `n = 100, 300` at 5–40% density (sparse).
- Upstream's `368×368` benchmark matrix, matching the known optimal cost (`224.8899507294651`) within `1e-9` on both solvers.

## Benchmarks

Tested against the upstream C implementation (`gcc -O2`) on the same hardware with fixed inputs (median of 100 runs).

| Dense matrix | C (`gcc -O2`) | Go |
| --- | --- | --- |
| 1000×1000 | ~56 ms | ~55 ms |
| 2000×2000 | ~481 ms | ~476 ms |
| 4000×4000 | ~1281 ms | ~1340 ms |

| Sparse matrix | C (`gcc -O2`) | Go |
| --- | --- | --- |
| 1000×1000, 50 entries/row (banded) | ~7.6 ms | ~11.1 ms |
| 4000×4000, 40 entries/row (banded) | ~71.6 ms | ~69.0 ms |
| 1000×1000, 10 entries/row (random, integer costs) | ~2.4 ms | ~1.3 ms |

A few takeaways (YMMV depending on your CPU):
- **Dense**: Pretty much neck-and-neck with C (within ~5%).
- **Sparse (banded)**: Go lags behind by ~1.5x at 1000×1000, then evens out at 4000×4000. These spend most of their time in the augmenting-row-reduction phase. The two-minima scan has a lot of branches that `gcc` optimizes into branchless assembly, whereas the Go compiler struggles a bit.
- **Sparse (random, integer costs)**: Go is nearly 2x faster here. Tie-heavy inputs push the solver into the `find_path` phase. The C version rebuilds a reverse index for every scanned column, while we just do a binary search over the sorted row.

To run the Go benchmarks yourself:

```bash
go test -bench . -benchtime 1x -count 100
```

The C benchmark harness lives in `bench/` and covers both solvers. `sh run.sh` runs 20 invocations per config (100 samples, matching the numbers above); `sh run.sh 1` gives you a quick pass.

## License

Apache-2.0 (Copyright 2026 PloyBox). See [LICENSE](LICENSE) for details.

The core algorithms are ported from [gatagat/lap](https://github.com/gatagat/lap) by Tomas Kazmar (BSD-2-Clause). Original copyright notices and disclaimers are kept in [NOTICE](NOTICE).
