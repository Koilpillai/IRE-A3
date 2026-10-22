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

echo "== unit tests"
"$PYTHON" -m pytest -q

echo "== Part 1: exact baseline"
"$PYTHON" part1_exact.py $QUICK

# Parts 2-4 and the figures/tables are added here as they are implemented.
