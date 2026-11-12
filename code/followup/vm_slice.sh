#!/bin/bash
# One slice of a follow-up arm on the Mac VM, whose commands end after ~3 minutes: N worker processes run
# segmented fits (checkpoint + TimeUp before the deadline), then the progress is printed. Re-run to continue.
#   followup/vm_slice.sh ARM OUT NWORKERS [SECONDS] [extra mac_arms.py args, e.g. --tiny]
ARM=$1; OUT=$2; N=$3; SEC=${4:-165}; shift 4 2>/dev/null || shift $#
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
export PYTHONHASHSEED=0 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
mkdir -p "$OUT/logs"
pids=()
for k in $(seq 0 $((N - 1))); do
  timeout $((SEC + 10)) python3 followup/mac_arms.py "$ARM" --out "$OUT" --worker "$k" --nslots "$N" --segment-seconds "$SEC" \
      --boot-seconds "${BOOT:-60}" --xla-cache "${XLA_CACHE:-$HOME/jax_cache}" "$@" > "$OUT/logs/slice_${ARM}_w$k.out" 2>&1 &
  pids+=($!)
done
( peak=0; while sleep 10; do u=$(free -m | awk 'NR==2{print $3}'); [ "$u" -gt "$peak" ] && peak=$u; echo "$peak" > "$OUT/logs/mem_peak.txt"; done ) &
mon=$!
wait "${pids[@]}"
kill "$mon" 2>/dev/null
echo "$(date +%T) $ARM: $N workers x ${SEC}s; memory peak $(cat "$OUT/logs/mem_peak.txt" 2>/dev/null) MB"
for k in $(seq 0 $((N - 1))); do
  tail -n 1 "$OUT/logs/${ARM}_w$k.log" 2>/dev/null | cut -c1-150
done
python3 followup/mac_arms.py progress --out "$OUT" 2>/dev/null | head -3
