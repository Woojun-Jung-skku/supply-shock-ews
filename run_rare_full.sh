#!/bin/bash
# Rare arm: one process per regime (3 processes), 5 replications each. Add --no-nets to the python line for the configuration reported in the paper.
# Run from the repository root (the folder that contains code/). Re-run the same command to resume.
#   bash run_rare_full.sh            # 3 processes
#   REPS=5 OUT=results_rare bash run_rare_full.sh
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results_rare}; REPS=${REPS:-5}; HOURS=${HOURS:-48}
[ -d .venv ] && source .venv/bin/activate
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
mkdir -p "$OUT/logs"
for reg in stable gradual abrupt; do
  nohup python3 run_rare_full.py --out "$OUT" --hours "$HOURS" --reps "$REPS" --regimes "$reg" > "$OUT/logs/run_$reg.out" 2>&1 &
  echo "started $reg (pid $!)"
done
echo "progress: tail -f $OUT/logs/pilot.log   |   results: $OUT/B/*.jsonl (one line per finished replication)"
