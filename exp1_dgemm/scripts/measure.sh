#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dgemm"
SIZES=(512 1024 2048 4096)
IMPLS=(cublas tiled naive)

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_sweep.csv"
echo "impl,m,n,k,time_ms,gflops" > "$LOG"

for n in "${SIZES[@]}"; do
  for impl in "${IMPLS[@]}"; do
    echo "Running $impl N=$n"
    # TODO(student): parse binary output and append to CSV (e.g., using grep/awk)
    "$BIN" --m "$n" --n "$n" --k "$n" --impl "$impl" --no-verify
  done
done

echo "Results stored in $LOG"

