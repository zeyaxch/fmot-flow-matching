#!/bin/sh
# Every 2D experiment in order: the exact field (no training), the main comparison and the two
# sweeps of FM-OT. Usage: PY=python3 sh scripts/run_all_2d.sh
set -e
PY=${PY:-python3}
cd "$(dirname "$0")/.."
$PY scripts/run_exact_2d.py
$PY scripts/run_2d.py --experiment main
$PY scripts/run_2d.py --experiment sigma0
$PY scripts/run_2d.py --experiment batch
echo "all 2D runs finished"
