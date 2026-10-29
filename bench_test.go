package lap

import (
	"sort"
	"testing"
)

// lcgMatrix builds the same deterministic matrix the C reference harness
// uses, so Go and C timings are directly comparable.
func lcgMatrix(n int) [][]float64 {
	state := uint64(1)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(state>>33) / float64(uint64(1)<<31)
	}
	flat := make([]float64, n*n)
	k := 0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			flat[k] = next()
			k++
		}
	}
	return rowViews(flat, n)
}

func BenchmarkLapjv1k(b *testing.B) {
	cost := lcgMatrix(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Lapjv(cost); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLapjv2k(b *testing.B) {
	cost := lcgMatrix(2000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Lapjv(cost); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLapjv4k(b *testing.B) {
	cost := lcgMatrix(4000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Lapjv(cost); err != nil {
			b.Fatal(err)
		}
	}
}

// lcgSparseCSR builds a deterministic sparse CSR matrix: row i stores
// perRow consecutive columns (wrapping around), so the diagonal is always
// present and a feasible assignment exists.
func lcgSparseCSR(n, perRow int) ([]float64, []int, []int) {
	state := uint64(1)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(state>>33) / float64(uint64(1)<<31)
	}
	cc := make([]float64, 0, n*perRow)
	kk := make([]int, 0, n*perRow)
	ii := make([]int, n+1)
	cols := make([]int, perRow)
	for i := 0; i < n; i++ {
		for k := 0; k < perRow; k++ {
			cols[k] = (i + k) % n
		}
		sort.Ints(cols)
		for _, j := range cols {
			cc = append(cc, next())
			kk = append(kk, j)
		}
		ii[i+1] = len(cc)
	}
	return cc, ii, kk
}

// lcgSparseCSRRandom builds a deterministic random sparse CSR matrix: row i
// stores perRow-1 LCG-sampled columns plus the diagonal, so a feasible
// assignment always exists. Costs are integers in [0, span); small spans
// create many cost ties, which push the solve into the find_path
// augmentation phase the way upstream's seeded integer-cost fixtures do.
// span must stay below large (1e6).
func lcgSparseCSRRandom(n, perRow, span int) ([]float64, []int, []int) {
	state := uint64(1)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(state>>33) / float64(uint64(1)<<31)
	}
	cc := make([]float64, 0, n*perRow)
	kk := make([]int, 0, n*perRow)
	ii := make([]int, n+1)
	chosen := make([]bool, n)
	cols := make([]int, perRow)
	for i := 0; i < n; i++ {
		for j := range chosen {
			chosen[j] = false
		}
		m := 0
		for m < perRow-1 {
			j := int(next() * float64(n))
			if j != i && !chosen[j] {
				chosen[j] = true
				cols[m] = j
				m++
			}
		}
		cols[m] = i
		sort.Ints(cols)
		for _, j := range cols {
			cc = append(cc, float64(int(next()*float64(span))))
			kk = append(kk, j)
		}
		ii[i+1] = len(cc)
	}
	return cc, ii, kk
}

func BenchmarkLapmodSparse1k(b *testing.B) {
	cc, ii, kk := lcgSparseCSR(1000, 50)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Lapmod(len(ii)-1, cc, ii, kk); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLapmodSparse4k(b *testing.B) {
	cc, ii, kk := lcgSparseCSR(4000, 40)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Lapmod(len(ii)-1, cc, ii, kk); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkLapmodSparsePath1k uses a random sparse layout with small-span
// integer costs, whose ties leave enough rows for the find_path
// augmentation phase after the two augmenting-row-reduction rounds.
func BenchmarkLapmodSparsePath1k(b *testing.B) {
	cc, ii, kk := lcgSparseCSRRandom(1000, 10, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Lapmod(len(ii)-1, cc, ii, kk); err != nil {
			b.Fatal(err)
		}
	}
}
