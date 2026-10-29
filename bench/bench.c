#define _POSIX_C_SOURCE 199309L
// Minimal harness to time the upstream C lapjv against this Go port.
// Uses the same deterministic LCG matrix as bench_test.go.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lapjv.h"

static unsigned long long lcg = 1;
static double urand(void) {
	lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
	return (double)(lcg >> 33) / (double)(1ULL << 31);
}

int main(int argc, char **argv) {
	int n = argc > 1 ? atoi(argv[1]) : 1000;
	cost_t *flat = malloc(sizeof(cost_t) * n * n);
	cost_t **cost = malloc(sizeof(cost_t *) * n);
	for (int i = 0; i < n; i++) {
		cost[i] = flat + (size_t)i * n;
		for (int j = 0; j < n; j++) cost[i][j] = urand();
	}
	int_t *x = malloc(sizeof(int_t) * n);
	int_t *y = malloc(sizeof(int_t) * n);
	for (int rep = 0; rep < 5; rep++) {
		struct timespec t0, t1;
		clock_gettime(CLOCK_MONOTONIC, &t0);
		lapjv_internal(n, cost, x, y);
		clock_gettime(CLOCK_MONOTONIC, &t1);
		printf("%.1f ", (t1.tv_sec - t0.tv_sec) * 1e3 +
		                    (t1.tv_nsec - t0.tv_nsec) / 1e6);
	}
	printf("(n=%d ms)\n", n);
	return 0;
}
