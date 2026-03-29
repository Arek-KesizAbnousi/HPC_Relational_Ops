#!/usr/bin/env bash
# run_profiling.sh — Reproduce the Nsight Systems + Nsight Compute traces
#
# Prerequisites:
#   - NVIDIA CUDA Toolkit with nsys and ncu on PATH
#   - Project built: make
#
# Usage:
#   chmod +x profiling/run_profiling.sh
#   ./profiling/run_profiling.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

BINARY="./HPC_RelOps"
ROWS_A=100000
ROWS_B=100000
KEY_RANGE=50000
REPEATS=5
OUTDIR="profiling"

# Build optimised binary
echo "=== Building ==="
make clean && make
echo ""

# ---- Nsight Systems: timeline + CUDA API trace ----
echo "=== Nsight Systems: timeline trace ==="
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output="${OUTDIR}/nsys_report" \
  --force-overwrite=true \
  ${BINARY} ${ROWS_A} ${ROWS_B} ${KEY_RANGE} ${REPEATS}

echo ""
echo "Timeline report saved: ${OUTDIR}/nsys_report.nsys-rep"
echo "Open with:  nsys-ui ${OUTDIR}/nsys_report.nsys-rep"
echo ""

# ---- Nsight Systems: summary stats ----
echo "=== Nsight Systems: CUDA API summary ==="
nsys stats "${OUTDIR}/nsys_report.nsys-rep" --report cuda_api_sum 2>/dev/null || true
echo ""
echo "=== Nsight Systems: kernel summary ==="
nsys stats "${OUTDIR}/nsys_report.nsys-rep" --report cuda_gpu_kern_sum 2>/dev/null || true
echo ""

# ---- Nsight Compute: kernel-level metrics (single iteration) ----
echo "=== Nsight Compute: kernel analysis ==="
ncu --set full \
  --target-processes all \
  -o "${OUTDIR}/ncu_report" \
  ${BINARY} ${ROWS_A} ${ROWS_B} ${KEY_RANGE} 1

echo ""
echo "Kernel report saved: ${OUTDIR}/ncu_report.ncu-rep"
echo "Open with:  ncu-ui ${OUTDIR}/ncu_report.ncu-rep"
echo ""
echo "=== Profiling complete ==="
