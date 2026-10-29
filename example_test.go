package lap_test

import (
	"fmt"
	"math"

	"github.com/PloyBox/go-lap"
)

// ExampleLapjv solves a square assignment problem.
func ExampleLapjv() {
	cost := [][]float64{
		{1000, 2, 11, 10, 8, 7, 6, 5},
		{6, 1000, 1, 8, 8, 4, 6, 7},
		{5, 12, 1000, 11, 8, 12, 3, 11},
		{11, 9, 10, 1000, 1, 9, 8, 10},
		{11, 11, 9, 4, 1000, 2, 10, 9},
		{12, 8, 5, 2, 11, 1000, 11, 9},
		{10, 11, 12, 10, 9, 12, 1000, 3},
		{10, 10, 10, 10, 6, 3, 1, 1000},
	}
	res, err := lap.Lapjv(cost)
	fmt.Println(res.Cost, res.X, res.Y, err)
	// Output: 17 [1 2 0 4 5 3 7 6] [2 0 1 5 3 4 7 6] <nil>
}

// ExampleLapjv_extendCost pads a non-square matrix; unmatched columns
// are reported as -1.
func ExampleLapjv_extendCost() {
	rect := [][]float64{{1, 2, 3}, {4, 5, 6}}
	res, err := lap.Lapjv(rect, lap.WithExtendCost(true))
	fmt.Println(res.Cost, res.X, res.Y, err)
	// Output: 6 [0 1] [0 1 -1] <nil>
}

// ExampleLapjv_costLimit drops assignments at or above the limit.
func ExampleLapjv_costLimit() {
	sq3 := [][]float64{{1000, 2, 11}, {6, 1000, 1}, {5, 12, 1000}}
	res, err := lap.Lapjv(sq3, lap.WithCostLimit(4.99))
	fmt.Println(res.Cost, res.X, res.Y, err)
	// Output: 3 [1 2 -1] [-1 0 1] <nil>
}

// ExampleLapjv_returnCost mirrors return_cost=False: Cost stays NaN.
func ExampleLapjv_returnCost() {
	res, _ := lap.Lapjv([][]float64{{1, 2}, {2, 1}}, lap.WithReturnCost(false))
	fmt.Println(math.IsNaN(res.Cost), res.X, res.Y)
	// Output: true [0 1] [0 1]
}

// ExampleLapmod solves a sparse assignment problem given in CSR form:
// cost[i][kk[ii[i]+k]] == cc[ii[i]+k], with kk sorted within each row.
func ExampleLapmod() {
	cc := []float64{1000, 2, 11, 6, 1000, 1, 5, 12, 1000}
	ii := []int{0, 3, 6, 9}
	kk := []int{0, 1, 2, 0, 1, 2, 0, 1, 2}
	res, err := lap.Lapmod(len(ii)-1, cc, ii, kk)
	fmt.Println(res.Cost, res.X, res.Y, err)
	// Output: 8 [1 2 0] [2 0 1] <nil>
}
