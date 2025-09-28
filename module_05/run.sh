#!/usr/bin/env bash
# Run multiple grids/blocks for rubric screenshots.
set -e
BIN=build/memory_demo
[ -x "$BIN" ] || { echo "Build first: make"; exit 1; }

N=${1:-1048576}
REPS=${2:-5}

# At least 64 threads per rubric; vary threads/blocks:
for B in 64 128 256; do
  echo "== BLOCK SIZE ${B} =="
  "$BIN" -n "$N" -b "$B" -k all -r "$REPS"
done

# Also vary N (threads) implicitly via larger arrays (optional)
for NN in 262144 1048576 4194304; do
  echo "== N ${NN}, block 256 =="
  "$BIN" -n "$NN" -b 256 -k all -r "$REPS"
done