// Package lap solves the linear assignment problem using the Jonker-Volgenant
// algorithm. It is a faithful port of Python's "lap" package (v0.5.13) and
// provides both solvers: Lapjv for dense cost matrices (lap.lapjv) and Lapmod
// for sparse CSR matrices (lap.lapmod).
//
// Given identical inputs, both solvers return the same assignment as the
// Python reference implementation, down to tie-breaking. Lapjv exposes the
// Python keyword arguments as functional options:
//
//	lapjv(cost, extend_cost=False, cost_limit=np.inf, return_cost=True)
//
// See README.md for a complete mapping between the two APIs.
package lap

import (
	"errors"
	"math"
)

// Result holds the assignment computed by a solver.
//
// X[i] is the column assigned to row i, and Y[j] is the row assigned to
// column j. For square inputs with a full assignment, X and Y are inverse
// permutations. Unassigned rows or columns (resulting from matrix padding or
// a cost limit) have their index set to -1.
type Result struct {
	// Cost is the total cost of the assignment. It is NaN if cost calculation
	// was disabled via WithReturnCost or WithSparseReturnCost, and +Inf if the
	// assignment uses a row-column pair with no finite cost (an infeasible
	// problem).
	Cost float64
	// X maps each row index to its assigned column index (or -1 if unassigned).
	X []int
	// Y maps each column index to its assigned row index (or -1 if unassigned).
	Y []int
}

// config holds options for Lapjv.
type config struct {
	extendCost bool
	costLimit  float64
	returnCost bool
}

// Option configures a Lapjv solver call.
type Option func(*config)

// WithExtendCost permits non-square matrices by padding them to square with
// zero-cost dummy rows or columns.
func WithExtendCost(v bool) Option {
	return func(c *config) { c.extendCost = v }
}

// WithCostLimit sets an upper bound v on the cost of any individual match.
// Pairs with cost > v are excluded and left unassigned (-1). Pairs with cost
// == v may be kept or dropped depending on tie-breaking. Defaults to +Inf.
func WithCostLimit(v float64) Option {
	return func(c *config) { c.costLimit = v }
}

// WithReturnCost controls whether Result.Cost is computed. When false, cost
// summation is skipped and Result.Cost is set to NaN. Defaults to true.
func WithReturnCost(v bool) Option {
	return func(c *config) { c.returnCost = v }
}

// ErrNotSquare is returned when a non-square matrix is passed without
// WithExtendCost(true).
var ErrNotSquare = errors.New(
	"lap: square cost array expected; if intentional, pass WithExtendCost(true)")

// ErrRagged is returned when the input slice has rows of differing lengths.
var ErrRagged = errors.New("lap: cost must be a rectangular 2D matrix")

// Lapjv computes the minimum-cost assignment for a dense cost matrix, where
// cost[i][j] is the cost of matching row i to column j.
//
// Its behavior matches gatagat/lap v0.5.13:
//
//   - Square matrices are solved directly.
//   - Non-square matrices require WithExtendCost(true), which zero-pads the
//     matrix to max(rows, cols) before solving.
//   - When a finite cost limit l is set, the matrix is augmented to size
//     (rows+cols)×(rows+cols) with l/2 in the off-diagonal blocks and 0 in
//     the bottom-right block. Matches exceeding l are discarded (-1); exact
//     ties at l depend on tie-breaking.
//   - An empty (0×0) matrix returns Cost 0 and empty slices. (The C reference
//     reads out of bounds on 0×0 inputs.)
func Lapjv(cost [][]float64, opts ...Option) (Result, error) {
	cfg := &config{costLimit: math.Inf(1), returnCost: true}
	for _, o := range opts {
		o(cfg)
	}

	nRows := len(cost)
	nCols := 0
	if nRows > 0 {
		nCols = len(cost[0])
		for _, row := range cost {
			if len(row) != nCols {
				return Result{}, ErrRagged
			}
		}
	}

	var n int
	if nRows == nCols {
		n = nRows
	} else if !cfg.extendCost {
		return Result{}, ErrNotSquare
	}

	work := cost
	if cfg.costLimit < math.Inf(1) {
		n = nRows + nCols
		ext := make([]float64, n*n)
		half := cfg.costLimit / 2
		for i := range ext {
			ext[i] = half
		}
		for r := nRows; r < n; r++ {
			row := ext[r*n : (r+1)*n]
			for c := nCols; c < n; c++ {
				row[c] = 0
			}
		}
		for r := 0; r < nRows; r++ {
			copy(ext[r*n:(r*n)+nCols], cost[r])
		}
		work = rowViews(ext, n)
	} else if cfg.extendCost {
		if nRows > nCols {
			n = nRows
		} else {
			n = nCols
		}
		ext := make([]float64, n*n)
		for r := 0; r < nRows; r++ {
			copy(ext[r*n:(r*n)+nCols], cost[r])
		}
		work = rowViews(ext, n)
	}

	x := make([]int, n)
	y := make([]int, n)
	lapjvInternal(n, work, x, y)

	res := Result{Cost: math.NaN(), X: x, Y: y}
	if cfg.costLimit < math.Inf(1) || cfg.extendCost {
		for i, xi := range x {
			if xi >= nCols {
				x[i] = -1
			}
		}
		for j, yj := range y {
			if yj >= nRows {
				y[j] = -1
			}
		}
		x = x[:nRows]
		y = y[:nCols]
		res.X, res.Y = x, y
		if cfg.returnCost {
			sum := 0.0
			for i, xi := range x {
				if xi != -1 {
					sum += work[i][xi]
				}
			}
			res.Cost = sum
		}
	} else if cfg.returnCost {
		sum := 0.0
		for i := 0; i < nRows; i++ {
			sum += work[i][x[i]]
		}
		res.Cost = sum
	}
	return res, nil
}

// rowViews partitions a flat n*n backing slice into n row slices without
// copying the underlying elements.
func rowViews(flat []float64, n int) [][]float64 {
	rows := make([][]float64, n)
	for i := range rows {
		rows[i] = flat[i*n : (i+1)*n]
	}
	return rows
}

// Core dense solver, ported from _lapjv_cpp/lapjv.cpp in gatagat/lap v0.5.13.
// Indexing and tie-breaking semantics match the C reference exactly. Each
// function notes its corresponding upstream symbol.

// large corresponds to LARGE in the C reference, serving as the initial
// column dual variable and a sentinel for infinity during searches.
const large = 1e6

// ccrrtDense runs the initial column reduction and reduction transfer pass
// (upstream _ccrrt_dense).
//
// Column reduction sets each column dual v[j] to its minimum cost and records
// the minimizing row in y[j]. Iterating columns in reverse, each column is
// tentatively matched to its minimizing row; if that row is already matched,
// it is marked contested and the additional column remains unassigned. For rows
// matched to a single column, reduction transfer lowers v[j] by the row's
// second-smallest reduced cost, creating a zero-slack tie with the runner-up
// column to facilitate subsequent augmenting swaps.
//
// Populates freeRows with the remaining unassigned rows and returns their count.
func ccrrtDense(n int, cost [][]float64, freeRows []int, x, y []int, v []float64) int {
	for i := 0; i < n; i++ {
		x[i] = -1
		v[i] = large
		y[i] = 0
	}
	for i := 0; i < n; i++ {
		ci := cost[i]
		_ = ci[n-1] // BCE hint
		_ = v[n-1]
		for j := 0; j < n; j++ {
			if c := ci[j]; c < v[j] {
				v[j] = c
				y[j] = i
			}
		}
	}
	unique := make([]bool, n)
	for i := range unique {
		unique[i] = true
	}
	j := n
	for {
		j--
		i := y[j]
		if x[i] < 0 {
			x[i] = j
		} else {
			unique[i] = false
			y[j] = -1
		}
		if j <= 0 {
			break
		}
	}
	nFreeRows := 0
	for i := 0; i < n; i++ {
		if x[i] < 0 {
			freeRows[nFreeRows] = i
			nFreeRows++
		} else if unique[i] {
			col := x[i]
			ci := cost[i]
			_ = ci[n-1] // BCE hint
			_ = v[n-1]
			min := large
			for j2 := 0; j2 < n; j2++ {
				if j2 == col {
					continue
				}
				if c := ci[j2] - v[j2]; c < min {
					min = c
				}
			}
			v[col] -= min
		}
	}
	return nFreeRows
}

// carrDense runs the augmenting row reduction pass (upstream _carr_dense) over
// the first nFreeRows entries in freeRows.
//
// For each free row, the pass finds the columns with the smallest (j1) and
// second-smallest (j2) reduced costs. If reducing v[j1] by the gap
// (v2 - v1) strictly lowers v[j1], the price is updated and the evicted row
// is re-examined immediately; otherwise, the row falls back to j2 if
// available and the evicted row is deferred to the next pass. The
// rrCnt < current*n bound prevents unbounded immediate requeues.
//
// Compacts the remaining unassigned rows into freeRows and returns the new count.
func carrDense(n int, cost [][]float64, nFreeRows int, freeRows, x, y []int, v []float64) int {
	current := 0
	newFreeRows := 0
	rrCnt := 0
	for current < nFreeRows {
		rrCnt++
		freeI := freeRows[current]
		current++
		ci := cost[freeI]
		_ = ci[n-1] // BCE hint
		_ = v[n-1]
		j1 := 0
		v1 := ci[0] - v[0]
		j2 := -1
		v2 := large
		// Unroll 2x to reduce loop overhead; comparisons stay in order.
		j := 1
		for ; j+1 < n; j += 2 {
			c := ci[j] - v[j]
			if c < v2 {
				if c >= v1 {
					v2 = c
					j2 = j
				} else {
					v2 = v1
					v1 = c
					j2 = j1
					j1 = j
				}
			}
			c = ci[j+1] - v[j+1]
			if c < v2 {
				if c >= v1 {
					v2 = c
					j2 = j + 1
				} else {
					v2 = v1
					v1 = c
					j2 = j1
					j1 = j + 1
				}
			}
		}
		if j < n {
			c := ci[j] - v[j]
			if c < v2 {
				if c >= v1 {
					v2 = c
					j2 = j
				} else {
					v2 = v1
					v1 = c
					j2 = j1
					j1 = j
				}
			}
		}
		i0 := y[j1]
		v1New := v[j1] - (v2 - v1)
		v1Lowers := v1New < v[j1]
		if rrCnt < current*n {
			if v1Lowers {
				v[j1] = v1New
			} else if i0 >= 0 && j2 >= 0 {
				j1 = j2
				i0 = y[j2]
			}
			if i0 >= 0 {
				if v1Lowers {
					current--
					freeRows[current] = i0
				} else {
					freeRows[newFreeRows] = i0
					newFreeRows++
				}
			}
		} else {
			if i0 >= 0 {
				freeRows[newFreeRows] = i0
				newFreeRows++
			}
		}
		x[freeI] = j1
		y[j1] = freeI
	}
	return newFreeRows
}

// findDense partitions unscanned columns in cols[lo:n], moving those that
// achieve the minimum distance d into the scan window cols[lo:hi] (upstream
// _find_dense). Returns the updated upper bound hi.
func findDense(n, lo int, d []float64, cols, y []int) int {
	hi := lo + 1
	mind := d[cols[lo]]
	for k := hi; k < n; k++ {
		j := cols[k]
		if d[j] <= mind {
			if d[j] < mind {
				hi = lo
				mind = d[j]
			}
			cols[k] = cols[hi]
			cols[hi] = j
			hi++
		}
	}
	return hi
}

// scanDense relaxes unscanned columns cols[hi:n] from the current scan window
// cols[lo:hi] (upstream _scan_dense).
//
// For each column j in the window (matched to row i = y[j]), it updates the
// shortest-path distances d and predecessors pred of remaining columns via i.
// Any column whose distance drops to mind is appended to the scan window; if
// that column is unassigned, it is returned immediately without writing back
// lo or hi. Returns -1 once the scan window is exhausted.
func scanDense(n int, cost [][]float64, plo, phi *int, d []float64, cols, pred, y []int, v []float64) int {
	lo := *plo
	hi := *phi
	for lo != hi {
		j := cols[lo]
		lo++
		i := y[j]
		mind := d[j]
		h := cost[i][j] - v[j] - mind
		for k := hi; k < n; k++ {
			j = cols[k]
			credIJ := cost[i][j] - v[j] - h
			if credIJ < d[j] {
				d[j] = credIJ
				pred[j] = i
				if credIJ == mind {
					if y[j] < 0 {
						return j
					}
					cols[k] = cols[hi]
					cols[hi] = j
					hi++
				}
			}
		}
	}
	*plo = lo
	*phi = hi
	return -1
}

// findPathDense runs a Dijkstra-style shortest augmenting path search over
// reduced costs starting from startI to any unassigned column (upstream
// find_path_dense), updating dual potentials v for settled columns before
// returning. Returns the terminal unassigned column, with pred populated to
// backtrack to startI.
func findPathDense(n int, cost [][]float64, startI int, y []int, v []float64, pred []int) int {
	lo, hi := 0, 0
	finalJ := -1
	nReady := 0
	cols := make([]int, n)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		cols[i] = i
		pred[i] = startI
		d[i] = cost[startI][i] - v[i]
	}
	for finalJ == -1 {
		if lo == hi {
			nReady = lo
			hi = findDense(n, lo, d, cols, y)
			for k := lo; k < hi; k++ {
				if j := cols[k]; y[j] < 0 {
					finalJ = j
				}
			}
		}
		if finalJ == -1 {
			finalJ = scanDense(n, cost, &lo, &hi, d, cols, pred, y, v)
		}
	}
	mind := d[cols[lo]]
	for k := 0; k < nReady; k++ {
		j := cols[k]
		v[j] += d[j] - mind
	}
	return finalJ
}

// caDense matches the remaining nFreeRows unassigned rows via shortest
// augmenting paths (upstream _ca_dense), flipping x and y along each path.
func caDense(n int, cost [][]float64, nFreeRows int, freeRows, x, y []int, v []float64) int {
	pred := make([]int, n)
	for _, freeI := range freeRows[:nFreeRows] {
		i := -1
		k := 0
		j := findPathDense(n, cost, freeI, y, v, pred)
		for i != freeI {
			i = pred[j]
			y[j] = i
			j, x[i] = x[i], j
			k++
			if k >= n {
				// Invariant violation: an augmenting path cannot exceed n edges.
				panic("lap: augmenting path longer than n")
			}
		}
	}
	return 0
}

// lapjvInternal executes the full Jonker-Volgenant dense pipeline (upstream
// lapjv_internal): column reduction and reduction transfer, up to two passes
// of augmenting row reduction, and shortest-path augmentation for any
// remaining unassigned rows. Populates x and y in place.
func lapjvInternal(n int, cost [][]float64, x, y []int) {
	if n == 0 {
		return
	}
	freeRows := make([]int, n)
	v := make([]float64, n)
	ret := ccrrtDense(n, cost, freeRows, x, y, v)
	it := 0
	for ret > 0 && it < 2 {
		ret = carrDense(n, cost, ret, freeRows, x, y, v)
		it++
	}
	if ret > 0 {
		caDense(n, cost, ret, freeRows, x, y, v)
	}
}
