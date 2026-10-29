package lap

import (
	"math"
	"sort"
	"testing"
)

// Test cases in this file are ported from lap/tests/test_lapmod.py and
// test_arr_loop.py in gatagat/lap (v0.5.13). Expected assignments reflect
// upstream tie-breaking behavior.

// sparseFromDense converts a dense matrix to CSR format (port of
// test_utils.sparse_from_dense). It retains all entries, including +Inf,
// which input validation is expected to reject.
func sparseFromDense(cost [][]float64) (int, []float64, []int, []int) {
	n := len(cost)
	cc := make([]float64, 0, n*n)
	ii := make([]int, n+1)
	kk := make([]int, 0, n*n)
	for i, row := range cost {
		for j, c := range row {
			cc = append(cc, c)
			kk = append(kk, j)
		}
		ii[i+1] = ii[i] + len(row)
	}
	return n, cc, ii, kk
}

// sparseFromMasked converts a dense matrix to CSR format while filtering out
// infinite values (port of test_utils.sparse_from_masked).
func sparseFromMasked(cost [][]float64) (int, []float64, []int, []int) {
	n := len(cost)
	var cc []float64
	ii := make([]int, n+1)
	var kk []int
	for i, row := range cost {
		for j, c := range row {
			if !math.IsInf(c, 0) {
				cc = append(cc, c)
				kk = append(kk, j)
			}
		}
		ii[i+1] = len(cc)
	}
	return n, cc, ii, kk
}

// TestLapmodSquare ports the parameterized test_square suite and cross-checks
// each result against the dense Lapjv solver, matching upstream tests.
func TestLapmodSquare(t *testing.T) {
	cases := []struct {
		cost  [][]float64
		cost_ float64
		x, y  []int
	}{
		{dense8x8, 17.0, []int{1, 2, 0, 4, 5, 3, 7, 6}, []int{2, 0, 1, 5, 3, 4, 7, 6}},
		{[][]float64{{1000, 4, 1}, {1, 1000, 3}, {5, 1, 1000}}, 3., []int{2, 0, 1}, []int{1, 2, 0}},
		{[][]float64{{5, 1000, 3}, {1000, 2, 2}, {1, 5, 1000}}, 6., []int{2, 1, 0}, []int{2, 1, 0}},
		{[][]float64{{1000, 1001, 1000}, {1000, 1000, 1001}, {1, 2, 3}}, 2001., []int{2, 1, 0}, []int{2, 1, 0}},
		{[][]float64{{10, 10, 13}, {4, 8, 8}, {8, 5, 8}}, 22., []int{2, 0, 1}, []int{1, 2, 0}},
		{[][]float64{{11, 10, 6}, {10, 11, 11}, {11, 12, 15}}, 28., []int{2, 0, 1}, []int{1, 2, 0}},
		{[][]float64{{12, 4, 9}, {16, 15, 14}, {19, 13, 17}}, 37., []int{1, 0, 2}, []int{1, 0, 2}},
		{[][]float64{{2, 5, 7}, {7, 10, 12}, {1, 5, 9}}, 18., []int{2, 1, 0}, []int{2, 1, 0}},
		{[][]float64{{10, 6, 14, 1}, {17, 18, 17, 15}, {14, 17, 15, 8}, {11, 13, 11, 4}}, 41., []int{1, 2, 0, 3}, []int{2, 0, 1, 3}},
	}
	for i, tc := range cases {
		n, cc, ii, kk := sparseFromDense(tc.cost)
		got, err := Lapmod(n, cc, ii, kk)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		assertResult(t, got, tc.cost_, tc.x, tc.y)
		dense, err := Lapjv(tc.cost)
		if err != nil {
			t.Fatalf("case %d: dense cross-check: %v", i, err)
		}
		assertResult(t, dense, tc.cost_, tc.x, tc.y)
	}
}

// Port of test_sparse_square.
func TestLapmodSparseSquare(t *testing.T) {
	inf := math.Inf(1)
	cost := [][]float64{
		{11., 20., inf, inf, inf},
		{12., inf, 12., inf, inf},
		{inf, 11., 10., 15., 9.},
		{15., inf, inf, 22., inf},
		{13., inf, inf, inf, 15.},
	}
	n, cc, ii, kk := sparseFromMasked(cost)
	got, err := Lapmod(n, cc, ii, kk)
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, got, 71., []int{0, 2, 1, 3, 4}, []int{0, 2, 1, 3, 4})
}

// Port of test_infs_unsolvable (lapmod subset).
func TestLapmodInfsUnsolvable(t *testing.T) {
	inf := math.Inf(1)
	cases := [][][]float64{
		{
			{0., 0., 0., inf, inf},
			{inf, inf, inf, 0., 0.},
			{inf, inf, inf, 0., 0.},
			{inf, inf, inf, 0., 0.},
			{0., 0., 0., inf, inf},
		},
		{
			{19., 22., 16., inf, inf},
			{inf, inf, inf, 4., 13.},
			{inf, inf, inf, 3., 14.},
			{inf, inf, inf, 10., 12.},
			{11., 14., 13., inf, inf},
		},
	}
	for i, cost := range cases {
		n, cc, ii, kk := sparseFromMasked(cost)
		got, err := Lapmod(n, cc, ii, kk)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if !math.IsInf(got.Cost, 1) {
			t.Fatalf("case %d: cost = %v, want +Inf", i, got.Cost)
		}
	}
}

// Port of test_inf_unique. Passing explicit +Inf entries in the CSR payload
// should fail validation, whereas omitting them should yield a valid assignment.
func TestLapmodInfUnique(t *testing.T) {
	inf := math.Inf(1)
	costExt := [][]float64{
		{1000, 4, 1, inf},
		{1, 1000, 3, inf},
		{5, 1, 1000, inf},
		{inf, inf, inf, 0},
	}
	n, cc, ii, kk := sparseFromDense(costExt)
	if _, err := Lapmod(n, cc, ii, kk); err != ErrCostTooLarge {
		t.Fatalf("sparse_from_dense: got %v, want ErrCostTooLarge", err)
	}
	n, cc, ii, kk = sparseFromMasked(costExt)
	got, err := Lapmod(n, cc, ii, kk)
	if err != nil {
		t.Fatal(err)
	}
	if got.Cost != 3. {
		t.Fatalf("cost = %v, want 3", got.Cost)
	}
	assertIntSlice(t, "x", got.X, []int{2, 0, 1, 3})
}

// Port of test_inf_col.
func TestLapmodInfCol(t *testing.T) {
	inf := math.Inf(1)
	cost := [][]float64{
		{0., inf, 0., 0., inf},
		{inf, inf, 0., 0., 0.},
		{inf, inf, inf, 0., inf},
		{inf, inf, inf, 0., 0.},
		{0., inf, 0., inf, inf},
	}
	n, cc, ii, kk := sparseFromDense(cost)
	if _, err := Lapmod(n, cc, ii, kk); err != ErrCostTooLarge {
		t.Fatalf("sparse_from_dense: got %v, want ErrCostTooLarge", err)
	}
	n, cc, ii, kk = sparseFromMasked(cost)
	got, err := Lapmod(n, cc, ii, kk)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got.Cost, 1) {
		t.Fatalf("cost = %v, want +Inf", got.Cost)
	}
}

// Port of test_inf_row.
func TestLapmodInfRow(t *testing.T) {
	inf := math.Inf(1)
	cost := [][]float64{
		{0., 0., 0., 0., inf},
		{inf, inf, 0., 0., 0.},
		{inf, inf, inf, inf, inf},
		{inf, inf, inf, 0., 0.},
		{0., 0., 0., inf, inf},
	}
	n, cc, ii, kk := sparseFromDense(cost)
	if _, err := Lapmod(n, cc, ii, kk); err != ErrCostTooLarge {
		t.Fatalf("sparse_from_dense: got %v, want ErrCostTooLarge", err)
	}
	n, cc, ii, kk = sparseFromMasked(cost)
	got, err := Lapmod(n, cc, ii, kk)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got.Cost, 1) {
		t.Fatalf("cost = %v, want +Inf", got.Cost)
	}
}

// Port of test_all_inf.
func TestLapmodAllInf(t *testing.T) {
	cost := make([][]float64, 5)
	for i := range cost {
		cost[i] = []float64{math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)}
	}
	n, cc, ii, kk := sparseFromDense(cost)
	if _, err := Lapmod(n, cc, ii, kk); err != ErrCostTooLarge {
		t.Fatalf("sparse_from_dense: got %v, want ErrCostTooLarge", err)
	}
	n, cc, ii, kk = sparseFromMasked(cost)
	if _, err := Lapmod(n, cc, ii, kk); err != ErrNoCols {
		t.Fatalf("sparse_from_masked: got %v, want ErrNoCols", err)
	}
}

// prepareSparseCost pads a rectangular COO matrix into a square (n+m)x(n+m)
// CSR matrix using dummy edges weighted by costLimit (port of
// test_arr_loop.prepare_sparse_cost).
func prepareSparseCost(n, m int, cc []float64, ii, jj []int, costLimit float64) ([]float64, []int, []int) {
	type entry struct {
		i, j int
		c    float64
	}
	entries := make([]entry, 0, 2*len(cc)+n+m)
	for k := range cc {
		entries = append(entries, entry{ii[k], jj[k], cc[k]})
	}
	for i := 0; i < n; i++ {
		entries = append(entries, entry{i, m + i, costLimit})
	}
	for j := 0; j < m; j++ {
		entries = append(entries, entry{n + j, j, costLimit})
	}
	for k := range cc {
		entries = append(entries, entry{n + jj[k], m + ii[k], 0})
	}
	sort.Slice(entries, func(a, b int) bool {
		if entries[a].i != entries[b].i {
			return entries[a].i < entries[b].i
		}
		return entries[a].j < entries[b].j
	})
	cc2 := make([]float64, len(entries))
	kk := make([]int, len(entries))
	ii2 := make([]int, n+m+1)
	for k, e := range entries {
		cc2[k] = e.c
		kk[k] = e.j
		ii2[e.i+1]++
	}
	for i := 0; i < n+m; i++ {
		ii2[i+1] += ii2[i]
	}
	return cc2, ii2, kk
}

// Port of test_lapmod_arr_loop.
func TestLapmodArrLoop(t *testing.T) {
	cc0 := []float64{
		2.593883482138951146e-01, 3.080381437461217620e-01,
		1.976243020727339317e-01, 2.462740976049606068e-01,
		4.203993396282833528e-01, 4.286184525458427985e-01,
		1.706431415909629434e-01, 2.192929371231896185e-01,
		2.117769622802734286e-01, 2.604267578125001315e-01,
	}
	ii0 := []int{0, 0, 1, 1, 2, 2, 5, 5, 6, 6}
	jj0 := []int{0, 1, 0, 1, 1, 2, 0, 1, 0, 1}
	cc, ii, kk := prepareSparseCost(7, 3, cc0, ii0, jj0, 1e3)
	got, err := Lapmod(len(ii)-1, cc, ii, kk)
	if err != nil {
		t.Fatal(err)
	}
	y := append([]int(nil), got.Y...)
	for j, i := range y {
		if i >= 7 {
			y[j] = -1
		}
	}
	y = y[:3]
	if rel := math.Abs(got.Cost/4000.8455356917416 - 1); rel > 1e-10 {
		t.Fatalf("cost = %v, want ~4000.8455356917416", got.Cost)
	}
	if !equalInts(y, []int{5, 1, 2}) && !equalInts(y, []int{1, 5, 2}) {
		t.Fatalf("y = %v, want [5 1 2] or [1 5 2]", y)
	}
}
