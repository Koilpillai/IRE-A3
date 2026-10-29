#!/usr/bin/env bash
# Regenerates every number, figure and table behind the report.
#
#   bash run_all.sh            full datasets  -> results/
#   bash run_all.sh --quick    first 100,000 vectors, fewer queries -> results_quick/
#
# Expects the two data files in data/ (see README.md) and the packages of requirements.txt to be installed.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
QUICK=""
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK="--quick" ;;
        *) echo "usage: bash run_all.sh [--quick]" >&2; exit 2 ;;
    esac
done

# hnswlib needs a recent libstdc++. Under an Anaconda-based Python the conda copy is older and shadows the system
# one ("GLIBCXX_3.4.xx not found"); if so, preload the system copy.
if ! "$PYTHON" -c "import hnswlib" 2>/dev/null; then
    for lib in /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib64/libstdc++.so.6 /usr/lib/libstdc++.so.6; do
        if [ -e "$lib" ] && LD_PRELOAD="$lib" "$PYTHON" -c "import hnswlib" 2>/dev/null; then
            echo "note: importing hnswlib needs LD_PRELOAD=$lib (conda's libstdc++ is too old)"
            export LD_PRELOAD="$lib"
            break
        fi
    done
fi

echo "== unit tests"
"$PYTHON" -m pytest -q

echo "== Part 1: exact baseline"
"$PYTHON" part1_exact.py $QUICK

echo "== Part 2: benchmark HNSW, IVF-Flat and IVF-PQ"
"$PYTHON" part2_benchmark.py $QUICK
"$PYTHON" part2_report.py $QUICK

# Parts 3 and 4 and the report figures are added here as they are implemented.
