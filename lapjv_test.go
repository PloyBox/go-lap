package lap

import (
	"math"
	"testing"
)

// Test cases ported from gatagat/lap (v0.5.13).
// Expected assignments match upstream tie-breaking behavior.

var dense8x8 = [][]float64{
	{1000, 2, 11, 10, 8, 7, 6, 5},
	{6, 1000, 1, 8, 8, 4, 6, 7},
	{5, 12, 1000, 11, 8, 12, 3, 11},
	{11, 9, 10, 1000, 1, 9, 8, 10},
	{11, 11, 9, 4, 1000, 2, 10, 9},
	{12, 8, 5, 2, 11, 1000, 11, 9},
	{10, 11, 12, 10, 9, 12, 1000, 3},
	{10, 10, 10, 10, 6, 3, 1, 1000},
}

func assertResult(t *testing.T, got Result, wantCost float64, wantX, wantY []int) {
	t.Helper()
	if math.IsNaN(wantCost) {
		if !math.IsNaN(got.Cost) {
			t.Fatalf("cost = %v, want NaN", got.Cost)
		}
	} else if got.Cost != wantCost {
		t.Fatalf("cost = %v, want %v", got.Cost, wantCost)
	}
	assertIntSlice(t, "x", got.X, wantX)
	assertIntSlice(t, "y", got.Y, wantY)
}

func assertIntSlice(t *testing.T, name string, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s = %v, want %v (length %d != %d)", name, got, want, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s = %v, want %v (mismatch at %d)", name, got, want, i)
		}
	}
}

// Non-square matrices must be rejected unless cost extension is explicitly enabled.
func TestNonSquareFails(t *testing.T) {
	cost := make([][]float64, 3)
	for i := range cost {
		cost[i] = make([]float64, 2)
	}
	if _, err := Lapjv(cost); err == nil {
		t.Fatal("expected error for non-square input without extend_cost")
	}
}

// Rows sub-sliced from wider backing rows (the Go analogue of upstream's
// non-contiguous NumPy view) should be handled transparently.
func TestNonContiguous(t *testing.T) {
	sub := make([][]float64, 3)
	for i := range sub {
		sub[i] = dense8x8[i][:3]
	}
	got, err := Lapjv(sub)
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, got, 8.0, []int{1, 2, 0}, []int{2, 0, 1})
}

// Rectangular matrices are padded automatically when WithExtendCost is enabled.
func TestExtension(t *testing.T) {
	cost := make([][]float64, 2)
	for i := range cost {
		cost[i] = dense8x8[i][:4]
	}
	got, err := Lapjv(cost, WithExtendCost(true))
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, got, 3.0, []int{1, 2}, []int{-1, 0, 1, -1})
}

// A 4×4 matrix embedding the 2×4 case with manually added high-cost padding,
// verifying parity with automatic cost extension.
func TestNoExtension(t *testing.T) {
	c := [][]float64{
		dense8x8[0][:4],
		dense8x8[1][:4],
		{1001, 1001, 1001, 2001},
		{2001, 1001, 1001, 1001},
	}
	got, err := Lapjv(c)
	if err != nil {
		t.Fatal(err)
	}
	if got.Cost-2002 != 3.0 {
		t.Fatalf("cost = %v, want 2005", got.Cost)
	}
	assertIntSlice(t, "x", got.X, []int{1, 2, 0, 3})
	assertIntSlice(t, "y", got.Y, []int{2, 0, 1, 3})
}

// Assignments exceeding the cost limit should remain unassigned (-1).
func TestCostLimit(t *testing.T) {
	cost := make([][]float64, 3)
	for i := range cost {
		cost[i] = dense8x8[i][:3]
	}
	got, err := Lapjv(cost, WithCostLimit(4.99))
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, got, 3.0, []int{1, 2, -1}, []int{-1, 0, 1})
}

// Standard dense square matrices of varying sizes.
func TestSquare(t *testing.T) {
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
		got, err := Lapjv(tc.cost)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		assertResult(t, got, tc.cost_, tc.x, tc.y)
	}
}

// Solvable sparse matrix with disallowed edges represented as +Inf.
func TestSparseSquare(t *testing.T) {
	inf := math.Inf(1)
	cost := [][]float64{
		{11., 20., inf, inf, inf},
		{12., inf, 12., inf, inf},
		{inf, 11., 10., 15., 9.},
		{15., inf, inf, 22., inf},
		{13., inf, inf, inf, 15.},
	}
	got, err := Lapjv(cost)
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, got, 71., []int{0, 2, 1, 3, 4}, []int{0, 2, 1, 3, 4})
}

// Matrices with no valid complete assignment should yield +Inf total cost.
func TestInfsUnsolvable(t *testing.T) {
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
		got, err := Lapjv(cost)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if !math.IsInf(got.Cost, 1) {
			t.Fatalf("case %d: cost = %v, want +Inf", i, got.Cost)
		}
	}
}

// Matrix where +Inf constraints force a single unique assignment.
func TestInfUnique(t *testing.T) {
	inf := math.Inf(1)
	cost := [][]float64{
		{1000, 4, 1, inf},
		{1, 1000, 3, inf},
		{5, 1, 1000, inf},
		{inf, inf, inf, 0},
	}
	got, err := Lapjv(cost)
	if err != nil {
		t.Fatal(err)
	}
	if got.Cost != 3. {
		t.Fatalf("cost = %v, want 3", got.Cost)
	}
	assertIntSlice(t, "x", got.X, []int{2, 0, 1, 3})
}

// Entire column of +Inf makes the problem unsolvable.
func TestInfCol(t *testing.T) {
	inf := math.Inf(1)
	cost := [][]float64{
		{0., inf, 0., 0., inf},
		{inf, inf, 0., 0., 0.},
		{inf, inf, inf, 0., inf},
		{inf, inf, inf, 0., 0.},
		{0., inf, 0., inf, inf},
	}
	got, err := Lapjv(cost)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got.Cost, 1) {
		t.Fatalf("cost = %v, want +Inf", got.Cost)
	}
}

// Entire row of +Inf makes the problem unsolvable.
func TestInfRow(t *testing.T) {
	inf := math.Inf(1)
	cost := [][]float64{
		{0., 0., 0., 0., inf},
		{inf, inf, 0., 0., 0.},
		{inf, inf, inf, inf, inf},
		{inf, inf, inf, 0., 0.},
		{0., 0., 0., inf, inf},
	}
	got, err := Lapjv(cost)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got.Cost, 1) {
		t.Fatalf("cost = %v, want +Inf", got.Cost)
	}
}

// All-Inf matrix edge case.
func TestAllInf(t *testing.T) {
	cost := make([][]float64, 5)
	for i := range cost {
		cost[i] = []float64{math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1), math.Inf(1)}
	}
	got, err := Lapjv(cost)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got.Cost, 1) {
		t.Fatalf("cost = %v, want +Inf", got.Cost)
	}
}

// Regression test for the augmenting row reduction (ARR) infinite loop on near-tied floats.
func TestArrLoop(t *testing.T) {
	cc := []float64{
		2.593883482138951146e-01, 3.080381437461217620e-01,
		1.976243020727339317e-01, 2.462740976049606068e-01,
		4.203993396282833528e-01, 4.286184525458427985e-01,
		1.706431415909629434e-01, 2.192929371231896185e-01,
		2.117769622802734286e-01, 2.604267578125001315e-01,
	}
	ii := []int{0, 0, 1, 1, 2, 2, 5, 5, 6, 6}
	jj := []int{0, 1, 0, 1, 1, 2, 0, 1, 0, 1}
	cost := make([][]float64, 7)
	for i := range cost {
		cost[i] = []float64{1000, 1000, 1000}
	}
	for k := range cc {
		cost[ii[k]][jj[k]] = cc[k]
	}
	got, err := Lapjv(cost, WithExtendCost(true))
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.Cost-0.8455356917416) > 1e-10 {
		t.Fatalf("cost = %v, want ~0.8455356917416", got.Cost)
	}
	yOK := equalInts(got.Y, []int{5, 1, 2}) || equalInts(got.Y, []int{1, 5, 2})
	if !yOK {
		t.Fatalf("y = %v, want [5 1 2] or [1 5 2]", got.Y)
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
