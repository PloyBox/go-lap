package lap

import (
	"errors"
	"math"
)

// Sparse Jonker-Volgenant solver for CSR cost matrices, ported from
// _lapjv_cpp/lapmod.cpp in gatagat/lap v0.5.13. This port preserves the exact
// C tie-breaking semantics; comments note the upstream name for each
// function.

// FPVersion selects the shortest augmenting path strategy for Lapmod,
// corresponding to fp_version in lap.lapmod. All strategies reach the same
// optimal cost, though assignments may differ when multiple optima exist due
// to tie-breaking.
type FPVersion int

const (
	// FP1 rescans all n columns to find the minimum each round (upstream FP_1).
	// Best suited for dense rows.
	FP1 FPVersion = 1
	// FP2 maintains todo/scan/ready lists to visit only stored entries
	// (upstream FP_2). Best suited for sparse rows.
	FP2 FPVersion = 2
	// FPDynamic picks a strategy per matrix based on overall sparsity
	// (upstream FP_DYNAMIC). This is the default in lap.lapmod.
	FPDynamic FPVersion = 3
)

// Input validation errors, ported from check_cost in Python's lapmod.
var (
	// ErrNoRows is returned when n == 0. Unlike Lapjv, lap.lapmod rejects
	// empty matrices rather than treating 0x0 as a no-op.
	ErrNoRows = errors.New("lap: cost matrix has zero rows")
	// ErrNoCols is returned when the matrix has no finite entries.
	ErrNoCols = errors.New("lap: cost matrix has zero columns")
	// ErrNegativeCost is returned when any stored cost is negative.
	ErrNegativeCost = errors.New("lap: cost matrix values must be non-negative")
	// ErrCostTooLarge is returned when any stored cost is >= large (1e6).
	ErrCostTooLarge = errors.New("lap: cost matrix values must be less than 1000000")
)

// CSR structure validation errors. The C implementation indexes into cc, ii,
// and kk without bounds checks, making malformed CSR inputs UB; the Go port
// validates them up front.
var (
	// ErrBadII indicates a malformed row-pointer slice.
	ErrBadII = errors.New("lap: ii must have length n+1, start at 0, end at len(cc), and be non-decreasing")
	// ErrBadKK indicates out-of-bounds or unsorted column indices within a row.
	ErrBadKK = errors.New("lap: kk must hold one column index in [0, n) per entry of cc, strictly sorted within each row")
	// ErrBadFPVersion indicates an unrecognized FPVersion value.
	ErrBadFPVersion = errors.New("lap: fp version must be FP1, FP2, or FPDynamic")
)

// sparseConfig holds options for Lapmod.
type sparseConfig struct {
	returnCost bool
	fpVersion  FPVersion
}

// SparseOption configures Lapmod.
type SparseOption func(*sparseConfig)

// WithSparseReturnCost sets whether to compute the total assignment cost.
// If false, cost summation is skipped and Result.Cost is set to NaN.
// Defaults to true.
func WithSparseReturnCost(v bool) SparseOption {
	return func(c *sparseConfig) { c.returnCost = v }
}

// WithFPVersion sets the shortest augmenting path search strategy, matching
// the fp_version parameter in lap.lapmod. Defaults to FPDynamic.
func WithFPVersion(v FPVersion) SparseOption {
	return func(c *sparseConfig) { c.fpVersion = v }
}

// Lapmod solves the minimum-cost assignment problem for a sparse n×n cost
// matrix in CSR format. It ports the fast=True path of lap.lapmod from the
// Python "lap" library (v0.5.13).
//
// Only finite entries are stored: cc holds the cell costs, ii holds row
// offsets such that row i spans cc[ii[i]:ii[i+1]], and kk holds the column
// indices for each entry (strictly sorted per row), so that
// cost[i][kk[ii[i]+k]] == cc[ii[i]+k].
//
// Behavior matches gatagat/lap v0.5.13:
//
//   - Only square matrices are supported. There are no extend_cost or
//     cost_limit options; pad the CSR matrix directly if needed (as upstream's
//     own tests do).
//   - Stored costs must be in [0, 1e6). Implicitly omitted entries are
//     treated as +Inf.
//   - If no complete assignment exists, Cost is +Inf. Where the C code would
//     read out of bounds on infeasible inputs, this port safely leaves
//     unmatched rows as x[i] = -1.
//   - n == 0 is rejected, matching upstream's check_cost.
func Lapmod(n int, cc []float64, ii, kk []int, opts ...SparseOption) (Result, error) {
	cfg := &sparseConfig{returnCost: true, fpVersion: FPDynamic}
	for _, o := range opts {
		o(cfg)
	}
	switch cfg.fpVersion {
	case FP1, FP2, FPDynamic:
	default:
		return Result{}, ErrBadFPVersion
	}
	if err := validateSparse(n, cc, ii, kk); err != nil {
		return Result{}, err
	}

	x := make([]int, n)
	y := make([]int, n)
	lapmodInternal(n, cc, ii, kk, x, y, cfg.fpVersion)

	res := Result{Cost: math.NaN(), X: x, Y: y}
	if cfg.returnCost {
		res.Cost = getCostSparse(n, cc, ii, kk, x)
	}
	return res, nil
}

// validateSparse ports check_cost from Python's lapmod and adds structural
// invariants that the C code assumes without checking.
func validateSparse(n int, cc []float64, ii, kk []int) error {
	if n == 0 {
		return ErrNoRows
	}
	if len(kk) == 0 {
		return ErrNoCols
	}
	if len(ii) != n+1 || ii[0] != 0 || ii[n] != len(cc) {
		return ErrBadII
	}
	for i := 0; i < n; i++ {
		if ii[i+1] < ii[i] {
			return ErrBadII
		}
	}
	if len(kk) != len(cc) {
		return ErrBadKK
	}
	for i := 0; i < n; i++ {
		for k := ii[i]; k < ii[i+1]; k++ {
			if j := kk[k]; j < 0 || j >= n {
				return ErrBadKK
			}
		}
		for k := ii[i] + 1; k < ii[i+1]; k++ {
			if kk[k] <= kk[k-1] {
				return ErrBadKK
			}
		}
	}
	for _, c := range cc {
		if c < 0 {
			return ErrNegativeCost
		}
		if c >= large {
			return ErrCostTooLarge
		}
	}
	return nil
}

// ccrrtSparse runs the initial column reduction and reduction transfer pass
// (upstream _ccrrt_sparse).
//
// Column reduction finds each column's minimum cost v[j] and the row y[j] that
// achieves it. A reverse scan over columns then tentatively assigns each
// column to its minimizing row; if a row is claimed by multiple columns, it is
// marked non-unique and the extra columns remain unassigned. Finally, for
// rows that uniquely win their column and have at least one other entry,
// reduction transfer lowers v[j] by the row's second-smallest reduced cost,
// creating a tie with the runner-up so later passes can pivot through it.
//
// Unassigned rows are written to freeRows, and the function returns their count.
func ccrrtSparse(n int, cc []float64, ii, kk []int, freeRows, x, y []int, v []float64) int {
	for i := 0; i < n; i++ {
		x[i] = -1
		v[i] = large
		y[i] = 0
	}
	for i := 0; i < n; i++ {
		lo, hi := ii[i], ii[i+1]
		if hi > lo {
			_ = kk[hi-1] // BCE hint for the inner loop
			_ = cc[hi-1]
		}
		for k := lo; k < hi; k++ {
			j := kk[k]
			if c := cc[k]; c < v[j] {
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
		} else if unique[i] && ii[i+1]-ii[i] > 1 {
			col := x[i]
			lo, hi := ii[i], ii[i+1]
			_ = kk[hi-1] // BCE hint for the inner loop
			_ = cc[hi-1]
			min := large
			for k := lo; k < hi; k++ {
				if j2 := kk[k]; j2 != col {
					if c := cc[k] - v[j2]; c < min {
						min = c
					}
				}
			}
			v[col] -= min
		}
	}
	return nFreeRows
}

// carrSparse runs the augmenting row reduction pass (upstream _carr_sparse)
// over the first nFreeRows entries in freeRows.
//
// Each free row picks its cheapest column j1 and runner-up j2. If lowering
// v[j1] by the gap (v2 - v1) actually decreases v[j1], the price is updated
// and the evicted row is re-processed immediately; otherwise, the row falls
// back to j2 if available and the evicted row is deferred to the next pass.
// The rrCnt < current*n check caps immediate retries to prevent infinite
// loops.
//
// Empty rows fall back to column 0 with cost `large`, matching the C code's
// dummy assignment. This renders the problem infeasible and surfaces as +Inf
// in getCostSparse.
//
// Remaining unassigned rows are compacted into freeRows, and the new count is
// returned.
func carrSparse(n int, cc []float64, ii, kk []int, nFreeRows int, freeRows, x, y []int, v []float64) int {
	current := 0
	newFreeRows := 0
	rrCnt := 0
	for current < nFreeRows {
		rrCnt++
		freeI := freeRows[current]
		current++
		lo, hi := ii[freeI], ii[freeI+1]
		var j1 int
		var v1 float64
		if hi > lo {
			j1 = kk[lo]
			v1 = cc[lo] - v[j1]
		} else {
			j1 = 0
			v1 = large
		}
		j2 := -1
		v2 := large
		// Unroll 2x to reduce loop overhead, matching carrDense. Reslicing
		// up front also lets the compiler elide bounds checks without
		// changing the comparison order.
		rowKK := kk[lo:hi]
		rowCC := cc[lo:hi]
		t := 1
		for ; t+1 < len(rowKK); t += 2 {
			j := rowKK[t]
			c := rowCC[t] - v[j]
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
			j = rowKK[t+1]
			c = rowCC[t+1] - v[j]
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
		if t < len(rowKK) {
			j := rowKK[t]
			c := rowCC[t] - v[j]
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

// findSparse2 partitions the minimum-distance columns from the todo list into
// the scan list, skipping columns already marked done (upstream _find_sparse_2).
// Returns the number of columns added to scan.
func findSparse2(d []float64, scan []int, nTodo int, todo []int, done []bool) int {
	hi := 0
	mind := large
	for k := 0; k < nTodo; k++ {
		j := todo[k]
		if done[j] {
			continue
		}
		if d[j] <= mind {
			if d[j] < mind {
				hi = 0
				mind = d[j]
			}
			scan[hi] = j
			hi++
		}
	}
	return hi
}

// scanSparse1 relaxes unscanned columns from the current scan window (upstream
// _scan_sparse_1). For each scanned column j (assigned to row i = y[j]), it
// updates distances d for remaining columns via row i, using a per-column
// reverse index into row i's CSR entries. Columns that tie the current minimum
// distance are moved into the scan window. If an unassigned column is reached
// at the minimum distance, it is returned immediately without updating *plo or
// *phi; otherwise, returns -1 once the scan window is exhausted.
func scanSparse1(n int, cc []float64, ii, kk []int, plo, phi *int, d []float64, cols, pred, y []int, v []float64) int {
	lo := *plo
	hi := *phi
	revKk := make([]int, n)
	for lo != hi {
		j := cols[lo]
		lo++
		i := y[j]
		mind := d[j]
		for k := 0; k < n; k++ {
			revKk[k] = -1
		}
		for k := ii[i]; k < ii[i+1]; k++ {
			revKk[kk[k]] = k
		}
		kj := revKk[j]
		if kj == -1 {
			continue
		}
		h := cc[kj] - v[j] - mind
		for k := hi; k < n; k++ {
			j = cols[k]
			if kj = revKk[j]; kj == -1 {
				continue
			}
			credIJ := cc[kj] - v[j] - h
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

// scanSparse2 relaxes todo columns from the current scan window (upstream
// _scan_sparse_2). For each scanned column j (assigned to row i = y[j]), it
// relaxes non-done columns using only row i's stored entries. Columns hitting
// the minimum distance join the scan window and are marked done; others are
// queued into todo once. Processed columns are pushed to ready for the dual
// update. Returns the first unassigned column reached at the minimum distance
// (leaving caller pointers untouched), or -1 when the scan window is drained.
func scanSparse2(n int, cc []float64, ii, kk []int, plo, phi *int, d []float64, pred []int, done []bool, pnReady *int, ready, scan []int, pnTodo *int, todo []int, added []bool, y []int, v []float64) int {
	lo := *plo
	hi := *phi
	nTodo := *pnTodo
	nReady := *pnReady
	for lo != hi {
		j := scan[lo]
		lo++
		i := y[j]
		ready[nReady] = j
		nReady++
		mind := d[j]
		// Upstream rebuilds a full reverse index (rev_kk) per scanned column
		// and asserts that j exists in row i. Since kk is strictly sorted per
		// row, binary search finds the same offset directly. Note that the
		// assertion fails when carrSparse assigns an empty row to dummy column 0
		// (causing an OOB read in C); relaxation is simply skipped here, as the
		// instance is infeasible and getCostSparse will return +Inf anyway.
		if kj := binarySearch(kk[ii[i]:ii[i+1]], j); kj != -1 {
			rowLo, rowHi := ii[i], ii[i+1]
			_ = kk[rowHi-1] // BCE hint for the inner loop
			_ = cc[rowHi-1]
			h := cc[rowLo+kj] - v[j] - mind
			for k := rowLo; k < rowHi; k++ {
				jj := kk[k]
				if done[jj] {
					continue
				}
				credIJ := cc[k] - v[jj] - h
				if credIJ < d[jj] {
					d[jj] = credIJ
					pred[jj] = i
					if credIJ <= mind {
						if y[jj] < 0 {
							return jj
						}
						scan[hi] = jj
						hi++
						done[jj] = true
					} else if !added[jj] {
						todo[nTodo] = jj
						nTodo++
						added[jj] = true
					}
				}
			}
		}
	}
	*pnTodo = nTodo
	*pnReady = nReady
	*plo = lo
	*phi = hi
	return -1
}

// findPathSparse1 runs a Dijkstra-style shortest augmenting path search from
// startI to an unassigned column over reduced costs (upstream find_path_sparse_1),
// updating dual potentials v for settled columns. It rescans all n columns each
// round, making it faster on denser rows. Returns the terminal column index
// and populates pred for path reconstruction. (Upstream's _find_sparse_1 is
// identical to _find_dense, so findDense is reused directly.)
func findPathSparse1(n int, cc []float64, ii, kk []int, startI int, y []int, v []float64, pred []int) int {
	lo, hi := 0, 0
	finalJ := -1
	nReady := 0
	cols := make([]int, n)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		cols[i] = i
		d[i] = large
		pred[i] = startI
	}
	for k := ii[startI]; k < ii[startI+1]; k++ {
		j := kk[k]
		d[j] = cc[k] - v[j]
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
			finalJ = scanSparse1(n, cc, ii, kk, &lo, &hi, d, cols, pred, y, v)
		}
	}
	mind := d[cols[lo]]
	for k := 0; k < nReady; k++ {
		j := cols[k]
		v[j] += d[j] - mind
	}
	return finalJ
}

// findPathSparse2 runs a Dijkstra-style shortest augmenting path search from
// startI to an unassigned column over reduced costs (upstream find_path_sparse_2),
// updating dual potentials v for settled columns. Unlike findPathSparse1, it
// visits only stored entries via todo/scan/ready lists, which is faster for
// very sparse rows. Returns the terminal column, or -1 if no augmenting path
// exists (an infeasible state where the C code reads out of bounds).
func findPathSparse2(n int, cc []float64, ii, kk []int, startI int, y []int, v []float64, pred []int) int {
	lo, hi := 0, 0
	finalJ := -1
	nReady := 0
	nTodo := ii[startI+1] - ii[startI]
	done := make([]bool, n)
	added := make([]bool, n)
	ready := make([]int, n)
	scan := make([]int, n)
	todo := make([]int, n)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = large
		pred[i] = startI
	}
	for k := ii[startI]; k < ii[startI+1]; k++ {
		j := kk[k]
		d[j] = cc[k] - v[j]
		todo[k-ii[startI]] = j
		added[j] = true
	}
	for finalJ == -1 {
		if lo == hi {
			lo = 0
			hi = findSparse2(d, scan, nTodo, todo, done)
			if hi == 0 {
				// Infeasible: no unassigned column is reachable. Pick the last
				// remaining free column to keep state valid for the caller;
				// getCostSparse will catch the missing edge and return +Inf.
				for j := 0; j < n; j++ {
					if !done[j] && y[j] < 0 {
						finalJ = j
					}
				}
				break
			}
			for k := lo; k < hi; k++ {
				j := scan[k]
				if y[j] < 0 {
					finalJ = j
				} else {
					done[j] = true
				}
			}
		}
		if finalJ == -1 {
			finalJ = scanSparse2(n, cc, ii, kk, &lo, &hi, d, pred, done, &nReady, ready, scan, &nTodo, todo, added, y, v)
		}
	}
	mind := d[scan[lo]]
	for k := 0; k < nReady; k++ {
		j := ready[k]
		v[j] += d[j] - mind
	}
	return finalJ
}

// caSparse matches the remaining nFreeRows unassigned rows by augmenting along
// shortest paths from findPathSparse1 or findPathSparse2 (upstream _ca_sparse),
// flipping x and y assignments along each path. The search variant is chosen
// once per call: forced by FP1/FP2, or selected by matrix sparsity for FPDynamic.
//
// If path search returns -1 (infeasible input; OOB in C), the row is left
// unassigned and getCostSparse will return +Inf.
func caSparse(n int, cc []float64, ii, kk []int, nFreeRows int, freeRows, x, y []int, v []float64, fp FPVersion) {
	useSparse1 := false
	switch fp {
	case FP1:
		useSparse1 = true
	case FP2:
		useSparse1 = false
	case FPDynamic:
		// Matches upstream get_better_find_path. Note that C computes n*n in
		// uint32, which overflows for n > 65535; the Go port uses float64 to
		// avoid this.
		sparsity := float64(ii[n]) / (float64(n) * float64(n))
		useSparse1 = sparsity > 0.25
	}
	pred := make([]int, n)
	for _, freeI := range freeRows[:nFreeRows] {
		var j int
		if useSparse1 {
			j = findPathSparse1(n, cc, ii, kk, freeI, y, v, pred)
		} else {
			j = findPathSparse2(n, cc, ii, kk, freeI, y, v, pred)
		}
		if j < 0 {
			continue
		}
		i := -1
		k := 0
		for i != freeI {
			i = pred[j]
			y[j] = i
			j, x[i] = x[i], j
			k++
			if k >= n {
				// Sanity check: simple paths cannot exceed n steps.
				panic("lap: augmenting path longer than n")
			}
		}
	}
}

// lapmodInternal executes the full sparse JV pipeline (upstream lapmod_internal):
// column reduction and reduction transfer, up to two passes of augmenting row
// reduction, and finally shortest-path augmentation for any remaining free rows.
// Populates x and y in place.
func lapmodInternal(n int, cc []float64, ii, kk []int, x, y []int, fp FPVersion) {
	freeRows := make([]int, n)
	v := make([]float64, n)
	ret := ccrrtSparse(n, cc, ii, kk, freeRows, x, y, v)
	it := 0
	for ret > 0 && it < 2 {
		ret = carrSparse(n, cc, ii, kk, ret, freeRows, x, y, v)
		it++
	}
	if ret > 0 {
		caSparse(n, cc, ii, kk, ret, freeRows, x, y, v, fp)
	}
}

// getCostSparse computes the total cost of the assignment in x (upstream
// get_cost). Returns +Inf if any row is matched to a column with no stored
// entry (indicating an infeasible assignment).
func getCostSparse(n int, cc []float64, ii, kk []int, x []int) float64 {
	ret := 0.0
	for i := 0; i < n; i++ {
		kj := binarySearch(kk[ii[i]:ii[i+1]], x[i])
		if kj == -1 {
			return math.Inf(1)
		}
		ret += cc[ii[i]+kj]
	}
	return ret
}

// binarySearch returns the index of key in the strictly sorted slice data,
// or -1 if not found. Ports the bisect_left-based binary_search from Python's
// lapmod.
func binarySearch(data []int, key int) int {
	lo, hi := 0, len(data)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if data[mid] < key {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < len(data) && data[lo] == key {
		return lo
	}
	return -1
}
