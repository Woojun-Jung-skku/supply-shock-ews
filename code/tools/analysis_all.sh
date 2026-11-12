#!/bin/bash
# v1.4: everything behind §5.2-5.5 from the result folders in one go.
#   tools/analysis_all.sh ANALYSIS_DIR RESULTS_DIR [RESULTS_DIR ...]
set -e
AN=$1; shift
DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$AN"
python3 "$DIR/tools/analyze.py" "$AN" "$@"
python3 "$DIR/tools/fitstats.py" "$@" > "$AN/fitstats.txt"
python3 "$DIR/tools/extra_b.py" "$AN" "$1" > "$AN/extra_b.txt"
for L in 1 2 3; do python3 "$DIR/tools/table5b.py" "$AN" $L > /dev/null; done
for R in stable gradual abrupt; do python3 "$DIR/tools/table5b.py" "$AN" 2 $R > /dev/null; done
python3 "$DIR/tools/fig6_final.py" "$AN" "$AN/fig6" > "$AN/fig6_numbers.txt" || echo "fig6 failed"
python3 "$DIR/tools/fig7_final.py" "$AN" "$AN/fig7" > "$AN/fig7_numbers.txt" || echo "fig7 failed"
ls "$AN"
