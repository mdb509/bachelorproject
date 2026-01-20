#!/usr/bin/env bash

# Script to run the benchmark and export results
set -euo pipefail

# Export directory inside container (bind-mount host ./export here)
EXPORT_DIR="${EXPORT_DIR:-/export}"

# Create export dir (if it's a bind mount, this maps to the host folder)
mkdir -p "$EXPORT_DIR"

cd /workspace/bachelorproject

# run benchmark
python3 main.py
python3 plot/plot.py || true

# copy results to export directory
cp -f benchmark_all.json "$EXPORT_DIR/benchmarks_all.json" 2>/dev/null || true
cp -r results "$EXPORT_DIR/" 2>/dev/null || true

# optional: relax permissions so the host user can read the files
chmod -R a+rwX "$EXPORT_DIR" 2>/dev/null || true

echo "Export fertig: $EXPORT_DIR"
