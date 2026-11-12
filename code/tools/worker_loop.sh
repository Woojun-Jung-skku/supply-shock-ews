#!/bin/bash
# v1.4: run one plan slot as a chain of fresh worker processes.
#   tools/worker_loop.sh SLOT OUT PLAN [extra run_mac.py args]
# The worker leaves with code 75 when its memory passes --max-rss-mb (or its JIT broke) and is restarted
# here; a killed worker (e.g. out of memory, 137) is restarted as well. Finished replications, Study B
# origins and ML origins are on disk, so a restart only repeats the replication that was running.
# Stops when the worker exits 0 (its plan list is done) or after MAXRESTART restarts.
SLOT=$1; OUT=$2; PLAN=$3; shift 3
DIR="$(cd "$(dirname "$0")/.." && pwd)"
# a fixed hash seed makes the traced programs identical across processes, so the persistent compilation
# cache hits after a restart (~15 s instead of ~65 s per program; random seeds gave 3 hits in 69 compiles)
export PYTHONHASHSEED=0
mkdir -p "$OUT/logs"
n=0
while true; do
  python3 "$DIR/run_mac.py" --out "$OUT" --workers 2 --nslots 2 --plan "$PLAN" --worker "$SLOT" "$@" >> "$OUT/logs/loop_$SLOT.out" 2>&1
  rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] slot $SLOT worker exited with $rc (restart $n)" >> "$OUT/logs/loop_$SLOT.log"
  [ $rc -eq 0 ] && break
  n=$((n+1))
  [ $n -ge "${MAXRESTART:-300}" ] && break
  sleep 3
done
