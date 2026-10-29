#define _POSIX_C_SOURCE 199309L
// Minimal harness to time the upstream C lapmod against this Go port.
// Uses the same deterministic LCG CSR matrices as bench_test.go: the banded
// layout by default, or, when a span argument is given, the random
// integer-cost layout whose ties reach the find_path augmentation phase.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lapjv.h"

static unsigned long long lcg = 1;
static double urand(void) {
	lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
	return (double)(lcg >> 33) / (double)(1ULL << 31);
}

static int cmp_int(const void *a, const void *b) {
	int x = *(const int *)a, y = *(const int *)b;
	return (x > y) - (x < y);
}

int main(int argc, char **argv) {
	int n = argc > 1 ? atoi(argv[1]) : 1000;
	int per_row = argc > 2 ? atoi(argv[2]) : 50;
	int span = argc > 3 ? atoi(argv[3]) : 0;
	int random_layout = span > 0;
	size_t nnz = (size_t)n * per_row;
	cost_t *cc = malloc(sizeof(cost_t) * nnz);
	uint_t *ii = malloc(sizeof(uint_t) * (n + 1));
	uint_t *kk = malloc(sizeof(uint_t) * nnz);
	int *cols = malloc(sizeof(int) * per_row);
	ii[0] = 0;
	if (random_layout) {
		char *chosen = malloc(n);
		for (int i = 0; i < n; i++) {
			memset(chosen, 0, n);
			int m = 0;
			while (m < per_row - 1) {
				int j = (int)(urand() * n);
				if (j != i && !chosen[j]) {
					chosen[j] = 1;
					cols[m++] = j;
				}
			}
			cols[m] = i;
			qsort(cols, per_row, sizeof(int), cmp_int);
			for (int k = 0; k < per_row; k++) {
				cc[ii[i] + k] = (cost_t)(int)(urand() * span);
				kk[ii[i] + k] = (uint_t)cols[k];
			}
			ii[i + 1] = ii[i] + per_row;
		}
		free(chosen);
	} else {
		for (int i = 0; i < n; i++) {
			for (int k = 0; k < per_row; k++) cols[k] = (i + k) % n;
			qsort(cols, per_row, sizeof(int), cmp_int);
			for (int k = 0; k < per_row; k++) {
				cc[ii[i] + k] = urand();
				kk[ii[i] + k] = (uint_t)cols[k];
			}
			ii[i + 1] = ii[i] + per_row;
		}
	}
	int_t *x = malloc(sizeof(int_t) * n);
	int_t *y = malloc(sizeof(int_t) * n);
	for (int rep = 0; rep < 5; rep++) {
		struct timespec t0, t1;
		clock_gettime(CLOCK_MONOTONIC, &t0);
		lapmod_internal(n, cc, ii, kk, x, y, FP_DYNAMIC);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		printf("%.1f ", (t1.tv_sec - t0.tv_sec) * 1e3 +
		                    (t1.tv_nsec - t0.tv_nsec) / 1e6);
	}
	double total = 0.0;
	for (int i = 0; i < n; i++) {
		int lo = ii[i], hi = ii[i + 1];
		while (lo < hi) {
			int mid = lo + (hi - lo) / 2;
			if (kk[mid] < (uint_t)x[i]) lo = mid + 1; else hi = mid;
		}
		if (lo < ii[i + 1] && kk[lo] == (uint_t)x[i]) {
			total += cc[lo];
		} else {
			total = INFINITY;
			break;
		}
	}
	printf("(n=%d per_row=%d%s ms) cost=%g\n", n, per_row,
	       random_layout ? " random" : "", total);
	return 0;
}
