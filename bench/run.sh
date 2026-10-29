#!/bin/sh
# Time the upstream C lapjv and lapmod (lap 0.5.13) on the same deterministic
# matrices as the Go benchmarks in bench_test.go. Requires a C compiler and
# network access on first run. Each config runs 20 times x 5 reps = 100
# samples, matching the methodology of the numbers in the README; pass a
# number of invocations to shorten (e.g. `sh run.sh 1`).
set -e
cd "$(dirname "$0")"
reps=${1:-20}

if [ ! -f lapjv.c ] || [ ! -f lapmod.c ]; then
	curl -sL -o lap.tar.gz \
		"https://files.pythonhosted.org/packages/f1/ae/5cc637c2e5158b7dcf1a9744d33b11dfc21d9309931169402f573e4d1ee3/lap-0.5.13.tar.gz"
	tar xzf lap.tar.gz
	cp lap-0.5.13/_lapjv_cpp/lapjv.h .
	cp lap-0.5.13/_lapjv_cpp/lapjv.cpp lapjv.c
	cp lap-0.5.13/_lapjv_cpp/lapmod.cpp lapmod.c
fi

gcc -O2 -std=c99 -o bench_c bench.c lapjv.c
for n in 1000 2000 4000; do
	for i in $(seq $reps); do ./bench_c "$n"; done
done

gcc -O2 -std=c99 -o bench_mod_c bench_mod.c lapmod.c
for i in $(seq $reps); do ./bench_mod_c 1000 50; done
for i in $(seq $reps); do ./bench_mod_c 4000 40; done
for i in $(seq $reps); do ./bench_mod_c 1000 10 100; done
