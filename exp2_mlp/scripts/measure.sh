#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dmlp"
LAYERS=("512,512,512" "1024,2048,1024" "2048,2048,2048")
BATCHES=(64 128 256 512)
IMPLS=(baseline activation_fused)
ACTIVATION="relu"

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_mlp_sweep.csv"
echo "impl,layers,batch,activation,time_ms,gflops" > "$LOG"

for layers in "${LAYERS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "Running $impl layers=$layers batch=$batch"
      # TODO(student): parse stdout from the binary and append to the CSV.
      "$BIN" --layers "$layers" --batch "$batch" --activation "$ACTIVATION" --impl "$impl" --no-verify
    done
  done
done

echo "Results stored in $LOG"
