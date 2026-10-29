package lap

import (
	"math"
	"math/rand"
	"testing"
)

// This file tests the sparse solver against several independent baselines:
// brute-force enumeration on small instances, negative-alternating-cycle
// optimality certificates on larger ones, cross-checks against the dense
// solver on random matrices, and the reference eps matrix from the Python
// package. It also covers fp_version agreement and the input validation
// beyond upstream's check_cost. Instead of bundling upstream's large numpy
// test fixtures, we generate deterministic sparse matrices directly in Go.

// genSparse generates a random sparse cost matrix with a planted perfect
// matching to guarantee a finite optimal solution.
func genSparse(rng *rand.Rand, n int, density float64, span int) ([][]float64, [][]bool) {
	cost := make([][]float64, n)
	mask := make([][]bool, n)
	for i := range cost {
		cost[i] = make([]float64, n)
		mask[i] = make([]bool, n)
		for j := range cost[i] {
			if rng.Float64() < density {
				mask[i][j] = true
				cost[i][j] = float64(rng.Intn(span))
			}
		}
	}
	for i, j := range rng.Perm(n) {
		mask[i][j] = true
		cost[i][j] = float64(rng.Intn(span))
	}
	return cost, mask
}

// maskToCSR converts the masked entries of cost into CSR format; kk comes
// out strictly sorted within each row, as Lapmod requires.
func maskToCSR(cost [][]float64, mask [][]bool) (int, []float64, []int, []int) {
	n := len(cost)
	var cc []float64
	ii := make([]int, n+1)
	var kk []int
	for i := range cost {
		for j := range cost[i] {
			if mask[i][j] {
				cc = append(cc, cost[i][j])
				kk = append(kk, j)
			}
		}
		ii[i+1] = len(cc)
	}
	return n, cc, ii, kk
}

// densify expands a masked sparse matrix into a full dense matrix,
// filling missing entries with +Inf.
func densify(cost [][]float64, mask [][]bool) [][]float64 {
	dense := make([][]float64, len(cost))
	for i := range dense {
		dense[i] = make([]float64, len(cost))
		for j := range dense[i] {
			dense[i][j] = math.Inf(1)
			if mask[i][j] {
				dense[i][j] = cost[i][j]
			}
		}
	}
	return dense
}

// TestLapmodBruteForce checks Lapmod against brute-force enumeration on
// small random sparse matrices. The density mix occasionally defers rows
// into the find_path phase, always via find_path_sparse_1; the sparse_2
// variant is covered by TestLapmodCertificate and TestLapmodFPVersions.
func TestLapmodBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	densities := []float64{0.9, 0.5, 0.2, 0.08}
	for trial := 0; trial < 300; trial++ {
		n := 1 + rng.Intn(8)
		density := densities[rng.Intn(len(densities))]
		cost, mask := genSparse(rng, n, density, 1+rng.Intn(10))
		nn, cc, ii, kk := maskToCSR(cost, mask)
		got, err := Lapmod(nn, cc, ii, kk)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		dense := densify(cost, mask)
		want := bruteForceMin(dense)
		if got.Cost != want {
			t.Fatalf("trial %d n=%d density=%v: cost = %v, want %v", trial, n, density, got.Cost, want)
		}
		checkConsistency(t, got, n, n)
		if hasNegativeCycle(dense, got.X) {
			t.Fatalf("trial %d n=%d: negative alternating cycle remains", trial, n)
		}
	}
}

// TestLapmodCertificate verifies optimality on larger sparse matrices by
// checking for negative alternating cycles across both path-finding regimes.
func TestLapmodCertificate(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	cases := []struct {
		n       int
		density float64
	}{
		{100, 0.4},  // hits find_path_sparse_1
		{300, 0.05}, // hits find_path_sparse_2
	}
	for _, tc := range cases {
		cost, mask := genSparse(rng, tc.n, tc.density, 100)
		nn, cc, ii, kk := maskToCSR(cost, mask)
		got, err := Lapmod(nn, cc, ii, kk)
		if err != nil {
			t.Fatalf("n=%d: %v", tc.n, err)
		}
		checkConsistency(t, got, tc.n, tc.n)
		if hasNegativeCycle(densify(cost, mask), got.X) {
			t.Fatalf("n=%d: negative alternating cycle remains", tc.n)
		}
	}
}

// TestLapmodMatchesDense verifies that Lapmod and Lapjv find the same
// optimal cost on random dense matrices; assignment-level parity is pinned
// by the fixed cases in lapmod_test.go, mirroring upstream's test_square.
func TestLapmodMatchesDense(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 100; trial++ {
		n := 2 + rng.Intn(30)
		span := 1 + rng.Intn(50)
		cost := make([][]float64, n)
		for i := range cost {
			cost[i] = make([]float64, n)
			for j := range cost[i] {
				cost[i][j] = float64(rng.Intn(span))
			}
		}
		nn, cc, ii, kk := sparseFromDense(cost)
		got, err := Lapmod(nn, cc, ii, kk)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		denseGot, err := Lapjv(cost)
		if err != nil {
			t.Fatalf("trial %d: dense: %v", trial, err)
		}
		if got.Cost != denseGot.Cost {
			t.Fatalf("trial %d n=%d: sparse cost %v != dense cost %v", trial, n, got.Cost, denseGot.Cost)
		}
		checkConsistency(t, got, n, n)
		if hasNegativeCycle(cost, got.X) {
			t.Fatalf("trial %d n=%d: negative alternating cycle remains", trial, n)
		}
	}
}

// TestLapmodFPVersions ensures all three FPVersion strategies yield the
// exact same optimal cost.
func TestLapmodFPVersions(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	densities := []float64{0.6, 0.2, 0.05}
	for trial := 0; trial < 30; trial++ {
		n := 2 + rng.Intn(40)
		density := densities[rng.Intn(len(densities))]
		cost, mask := genSparse(rng, n, density, 100)
		nn, cc, ii, kk := maskToCSR(cost, mask)
		costs := make([]float64, 3)
		for i, fp := range []FPVersion{FP1, FP2, FPDynamic} {
			got, err := Lapmod(nn, cc, ii, kk, WithFPVersion(fp))
			if err != nil {
				t.Fatalf("trial %d fp %d: %v", trial, fp, err)
			}
			costs[i] = got.Cost
			checkConsistency(t, got, n, n)
		}
		if costs[0] != costs[1] || costs[1] != costs[2] {
			t.Fatalf("trial %d n=%d density=%v: costs diverge: %v", trial, n, density, costs)
		}
	}
}

// TestLapmodEpsReference checks the sparse solver against the reference eps
// fixture and known optimum from the Python package.
func TestLapmodEpsReference(t *testing.T) {
	cost, err := loadEps("testdata/cost_eps.csv.gz")
	if err != nil {
		t.Fatal(err)
	}
	n, cc, ii, kk := sparseFromDense(cost)
	got, err := Lapmod(n, cc, ii, kk)
	if err != nil {
		t.Fatal(err)
	}
	const want = 224.8899507294651
	if math.Abs(got.Cost-want) > 1e-9 {
		t.Fatalf("cost = %.13f, want %.13f", got.Cost, want)
	}
	checkConsistency(t, got, n, n)
}

// TestLapmodValidation tests the structural CSR checks and option handling
// that upstream does not cover; its check_cost errors are already pinned by
// the fixed cases in lapmod_test.go.
func TestLapmodValidation(t *testing.T) {
	// Empty dimension matches check_cost's zero-rows error.
	if _, err := Lapmod(0, nil, nil, nil); err != ErrNoRows {
		t.Fatalf("n=0: got %v, want ErrNoRows", err)
	}
	// Empty matrix matches check_cost's zero-columns error.
	if _, err := Lapmod(2, nil, []int{0, 0, 0}, nil); err != ErrNoCols {
		t.Fatalf("empty: got %v, want ErrNoCols", err)
	}
	// Reject malformed CSR inputs (which would otherwise cause UB in C).
	if _, err := Lapmod(2, []float64{1}, []int{0, 1}, []int{0}); err != ErrBadII {
		t.Fatalf("short ii: got %v, want ErrBadII", err)
	}
	if _, err := Lapmod(1, []float64{1}, []int{0, 0}, []int{0}); err != ErrBadII {
		t.Fatalf("ii[n] != len(cc): got %v, want ErrBadII", err)
	}
	if _, err := Lapmod(2, []float64{1}, []int{0, 2, 1}, []int{0}); err != ErrBadII {
		t.Fatalf("decreasing ii: got %v, want ErrBadII", err)
	}
	if _, err := Lapmod(2, []float64{1, 1}, []int{0, 1, 2}, []int{0, 5}); err != ErrBadKK {
		t.Fatalf("column out of range: got %v, want ErrBadKK", err)
	}
	if _, err := Lapmod(2, []float64{1, 1}, []int{0, 2, 2}, []int{1, 0}); err != ErrBadKK {
		t.Fatalf("unsorted kk: got %v, want ErrBadKK", err)
	}
	if _, err := Lapmod(1, []float64{-1}, []int{0, 1}, []int{0}); err != ErrNegativeCost {
		t.Fatalf("negative: got %v, want ErrNegativeCost", err)
	}
	if _, err := Lapmod(1, []float64{1e6}, []int{0, 1}, []int{0}); err != ErrCostTooLarge {
		t.Fatalf("too large: got %v, want ErrCostTooLarge", err)
	}
	if _, err := Lapmod(1, []float64{1}, []int{0, 1}, []int{0}, WithFPVersion(FPVersion(9))); err != ErrBadFPVersion {
		t.Fatalf("bad fp: got %v, want ErrBadFPVersion", err)
	}

	// Minimal valid 1x1 input should succeed.
	got, err := Lapmod(1, []float64{2}, []int{0, 1}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, got, 2, []int{0}, []int{0})

	// Disabling cost calculation should set Cost to NaN while still returning the assignment.
	got, err = Lapmod(2, []float64{1, 2, 2, 1}, []int{0, 2, 4}, []int{0, 1, 0, 1}, WithSparseReturnCost(false))
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(got.Cost) {
		t.Fatalf("return_cost=false: cost = %v, want NaN", got.Cost)
	}
	assertIntSlice(t, "x", got.X, []int{0, 1})
	assertIntSlice(t, "y", got.Y, []int{0, 1})
}
