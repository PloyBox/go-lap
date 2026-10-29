package lap

import (
	"compress/gzip"
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"
)

// Cross-check the port against independent oracles:
//   - Brute-force enumeration on small instances
//   - Brute-force enumeration of the padded objective for extend_cost and
//     cost_limit semantics
//   - Negative-alternating-cycle optimality certificates for larger instances
//   - The upstream Python package's eps test fixture and reference optimum
//
// TestAPIEdges additionally covers Go-specific API edge cases without an
// upstream counterpart.

// permCost returns the total cost of the assignment given by perm.
func permCost(cost [][]float64, perm []int) float64 {
	sum := 0.0
	for i := range perm {
		sum += cost[i][perm[i]]
	}
	return sum
}

// bruteForceMin finds the minimum assignment cost via Heap's algorithm.
func bruteForceMin(cost [][]float64) float64 {
	n := len(cost)
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	best := permCost(cost, perm)
	p := make([]int, n)
	copy(p, perm)
	c := make([]int, n)
	for i := 0; i < n; {
		if c[i] < i {
			if i%2 == 0 {
				p[0], p[i] = p[i], p[0]
			} else {
				p[c[i]], p[i] = p[i], p[c[i]]
			}
			if s := permCost(cost, p); s < best {
				best = s
			}
			c[i]++
			i = 0
			continue
		}
		c[i] = 0
		i++
	}
	return best
}

// checkConsistency verifies that r.X and r.Y form a valid, mutually inverse matching.
func checkConsistency(t *testing.T, r Result, nRows, nCols int) {
	t.Helper()
	seenCol := make([]bool, nCols)
	for i, j := range r.X {
		if j == -1 {
			continue
		}
		if j < 0 || j >= nCols {
			t.Fatalf("x[%d] = %d out of range", i, j)
		}
		if seenCol[j] {
			t.Fatalf("column %d assigned twice", j)
		}
		seenCol[j] = true
		if r.Y[j] != i {
			t.Fatalf("y[%d] = %d, want %d", j, r.Y[j], i)
		}
	}
	seenRow := make([]bool, nRows)
	for j, i := range r.Y {
		if i == -1 {
			continue
		}
		if i < 0 || i >= nRows {
			t.Fatalf("y[%d] = %d out of range", j, i)
		}
		if seenRow[i] {
			t.Fatalf("row %d assigned twice", i)
		}
		seenRow[i] = true
	}
}

// hasNegativeCycle reports whether the residual alternating graph for matching x
// contains a negative cycle. By LP duality, a perfect matching on a square matrix
// is optimal iff no such cycle exists (where edge i -> k has weight cost[i][x[k]] - cost[i][x[i]]).
func hasNegativeCycle(cost [][]float64, x []int) bool {
	n := len(cost)
	dist := make([]float64, n)
	relax := func() bool {
		updated := false
		for i := 0; i < n; i++ {
			ci := cost[i]
			base := ci[x[i]]
			di := dist[i]
			for k := 0; k < n; k++ {
				if k == i {
					continue
				}
				if w := ci[x[k]] - base; di+w < dist[k] {
					dist[k] = di + w
					updated = true
				}
			}
		}
		return updated
	}
	for pass := 0; ; pass++ {
		if !relax() {
			return false
		}
		if pass >= n {
			return true
		}
	}
}

// TestBruteForceSquare checks Lapjv against brute-force enumeration on small
// random square matrices, including cases with heavy ties and negative costs.
func TestBruteForceSquare(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 300; trial++ {
		n := 1 + rng.Intn(8)
		span := []int{1, 2, 3, 10}[rng.Intn(4)]
		neg := rng.Intn(2) == 0
		cost := make([][]float64, n)
		for i := range cost {
			cost[i] = make([]float64, n)
			for j := range cost[i] {
				v := float64(rng.Intn(span))
				if neg && rng.Intn(2) == 0 {
					v = -v
				}
				cost[i][j] = v
			}
		}
		got, err := Lapjv(cost)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		want := bruteForceMin(cost)
		if got.Cost != want {
			t.Fatalf("trial %d n=%d: cost = %v, want %v\n%v", trial, n, got.Cost, want, cost)
		}
		checkConsistency(t, got, n, n)
		if hasNegativeCycle(cost, got.X) {
			t.Fatalf("trial %d n=%d: negative alternating cycle remains", trial, n)
		}
	}
}

// extendedTotal computes the effective objective value of perm under lapjv's
// matrix padding scheme without explicitly constructing the padded matrix.
// With a finite limit, dummy edges contribute (r+c-2m)*limit/2 across an (r+c)-sized
// permutation; with an infinite limit, the problem reduces to max(r, c) with zero
// padding cost, so only the real edges are summed.
func extendedTotal(cost [][]float64, r, c int, limit float64, perm []int) float64 {
	m := 0
	realSum := 0.0
	for i := 0; i < r; i++ {
		if j := perm[i]; j < c {
			realSum += cost[i][j]
			m++
		}
	}
	if math.IsInf(limit, 1) {
		return realSum
	}
	return realSum + float64(r+c-2*m)*limit/2
}

// bruteForceExtended finds the minimum padded objective value by exhausting all
// permutations of the virtual padded matrix.
func bruteForceExtended(cost [][]float64, r, c int, limit float64) float64 {
	n := r + c
	if math.IsInf(limit, 1) {
		n = r
		if c > n {
			n = c
		}
	}
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	p := make([]int, n)
	copy(p, perm)
	cCnt := make([]int, n)
	best := extendedTotal(cost, r, c, limit, p)
	for i := 0; i < n; {
		if cCnt[i] < i {
			if i%2 == 0 {
				p[0], p[i] = p[i], p[0]
			} else {
				p[cCnt[i]], p[i] = p[i], p[cCnt[i]]
			}
			if t := extendedTotal(cost, r, c, limit, p); t < best {
				best = t
			}
			cCnt[i]++
			i = 0
			continue
		}
		cCnt[i] = 0
		i++
	}
	return best
}

// TestBruteForceExtended verifies WithExtendCost and WithCostLimit against
// brute-force enumeration of the equivalent padded problem.
func TestBruteForceExtended(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 400; trial++ {
		r := 1 + rng.Intn(4)
		c := 1 + rng.Intn(4)
		cost := make([][]float64, r)
		for i := range cost {
			cost[i] = make([]float64, c)
			for j := range cost[i] {
				cost[i][j] = float64(rng.Intn(6))
			}
		}
		var opts []Option
		var limit float64
		switch rng.Intn(3) {
		case 0: // rectangular padding only
			opts = []Option{WithExtendCost(true)}
			limit = math.Inf(1)
		case 1: // cost limit on square input
			if r != c {
				continue
			}
			limit = 1.0 + float64(rng.Intn(8))
			opts = []Option{WithCostLimit(limit)}
		default: // both rectangular padding and cost limit
			limit = 1.0 + float64(rng.Intn(8))
			opts = []Option{WithExtendCost(true), WithCostLimit(limit)}
		}
		got, err := Lapjv(cost, opts...)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		checkConsistency(t, got, r, c)
		// Compare the full padded objective rather than realSum alone, as ties
		// in the padded matrix may yield different cardinalities (m) with the
		// same overall padded cost.
		m := 0
		realSum := 0.0
		for i, j := range got.X {
			if j != -1 {
				realSum += cost[i][j]
				m++
				if cost[i][j] > limit {
					t.Fatalf("trial %d: matched cost %v exceeds limit %v", trial, cost[i][j], limit)
				}
			}
		}
		if got.Cost != realSum {
			t.Fatalf("trial %d: cost %v != matched sum %v", trial, got.Cost, realSum)
		}
		totalGot := realSum
		if !math.IsInf(limit, 1) {
			totalGot += float64(r+c-2*m) * limit / 2
		}
		if want := bruteForceExtended(cost, r, c, limit); math.Abs(totalGot-want) > 1e-9 {
			t.Fatalf("trial %d r=%d c=%d limit=%v: total = %v, want %v\n%v",
				trial, r, c, limit, totalGot, want, cost)
		}
	}
}

// TestCertificateMedium verifies optimality via negative-cycle detection on
// medium-sized matrices where brute-force enumeration is intractable.
func TestCertificateMedium(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for _, n := range []int{40, 80, 150} {
		cost := make([][]float64, n)
		for i := range cost {
			cost[i] = make([]float64, n)
			for j := range cost[i] {
				cost[i][j] = float64(rng.Intn(30))
			}
		}
		got, err := Lapjv(cost)
		if err != nil {
			t.Fatal(err)
		}
		checkConsistency(t, got, n, n)
		if hasNegativeCycle(cost, got.X) {
			t.Fatalf("n=%d: negative alternating cycle remains", n)
		}
	}
}

// TestEpsReference replicates the upstream Python test suite's eps test
// against its known optimal cost.
func TestEpsReference(t *testing.T) {
	cost, err := loadEps("testdata/cost_eps.csv.gz")
	if err != nil {
		t.Fatal(err)
	}
	got, err := Lapjv(cost)
	if err != nil {
		t.Fatal(err)
	}
	const want = 224.8899507294651
	if math.Abs(got.Cost-want) > 1e-9 {
		t.Fatalf("cost = %.13f, want %.13f", got.Cost, want)
	}
	checkConsistency(t, got, len(cost), len(cost[0]))
}

func loadEps(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()
	rows, err := csv.NewReader(gz).ReadAll()
	if err != nil {
		return nil, err
	}
	cost := make([][]float64, len(rows))
	for i, row := range rows {
		cost[i] = make([]float64, len(row))
		for j, cell := range row {
			v, err := strconv.ParseFloat(cell, 64)
			if err != nil {
				return nil, fmt.Errorf("row %d col %d: %w", i, j, err)
			}
			cost[i][j] = v
		}
	}
	return cost, nil
}

// TestAPIEdges covers Go-specific API edge cases and invariants not present
// in the upstream Python test suite.
func TestAPIEdges(t *testing.T) {
	// 0x0 matrix should be a no-op returning (0, [], []), whereas the upstream C code triggers UB.
	got, err := Lapjv(nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Cost != 0 || len(got.X) != 0 || len(got.Y) != 0 {
		t.Fatalf("empty input: got %+v", got)
	}

	// Reject ragged matrices.
	if _, err := Lapjv([][]float64{{1, 2}, {3}}); err != ErrRagged {
		t.Fatalf("ragged: got %v, want ErrRagged", err)
	}

	// WithReturnCost(false) should set Cost to NaN while still populating X and Y.
	cost := [][]float64{{1, 2}, {2, 1}}
	got, err = Lapjv(cost, WithReturnCost(false))
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(got.Cost) {
		t.Fatalf("return_cost=false: cost = %v, want NaN", got.Cost)
	}
	assertIntSlice(t, "x", got.X, []int{0, 1})
	assertIntSlice(t, "y", got.Y, []int{0, 1})

	// Ensure the input matrix is not mutated in-place.
	before := fmt.Sprint(cost)
	if _, err := Lapjv(cost, WithCostLimit(1.5)); err != nil {
		t.Fatal(err)
	}
	if after := fmt.Sprint(cost); after != before {
		t.Fatalf("input mutated:\nbefore %s\nafter  %s", before, after)
	}
}
