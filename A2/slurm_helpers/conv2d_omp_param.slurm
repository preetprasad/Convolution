#!/bin/bash
#SBATCH --job-name=conv2d_omp_param
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --partition=cits3402
#SBATCH --output=logs/conv2d_omp_%j.out
#SBATCH --error=logs/conv2d_omp_%j.err

# Load compiler (tolerant & quiet). Some clusters don't have gcc/14.2.
if type module &>/dev/null; then
  module -q load gcc/14.2 2>/dev/null || \
  module -q load gcc       2>/dev/null || \
  echo "Note: gcc module not loaded; using system compiler (cc)" >&2
fi

# Usage:
#   sbatch conv2d_omp_param.slurm H W kH kW [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]
#
# Positional:
#   H, W           : input matrix dimensions
#   kH, kW         : kernel dimensions
#
# Optional:
#   THREADS        : OpenMP threads (default: 8, must match --cpus-per-task)
#   SCHED          : OpenMP schedule type (default: static)
#   CHUNK          : OpenMP chunk size (optional, leave empty for default)
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   STRIDE_H       : vertical stride (default: 1)
#   STRIDE_W       : horizontal stride (default: 1)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv2d_omp ] || [ conv2d_omp.c -nt conv2d_omp ]; then
  cc -std=c11 -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra -o conv2d_omp conv2d_omp.c -lm || exit 1
fi

if [ $# -lt 4 ]; then
  echo "Usage: sbatch conv2d_omp_param.slurm H W kH kW [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]" >&2
  exit 2
fi

H="$1"
W="$2"
KH="$3"
KW="$4"
THREADS="${5:-8}"      # OpenMP threads
SCHED="${6:-static}"   # OpenMP schedule
CHUNK="${7:-}"         # OpenMP chunk size
MODE="${8:-same}"      # convolution mode
PADDING="${9:-zero}"   # padding mode
STRIDE_H="${10:-1}"    # vertical stride
STRIDE_W="${11:-1}"    # horizontal stride
SEED="${12:-}"         # RNG seed

OUT="/dev/null"

# OMP environment
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
if [ "$SCHED" = "auto" ]; then
  export OMP_SCHEDULE=auto
else
  if [ -n "$CHUNK" ] && [ "$CHUNK" -gt 0 ] 2>/dev/null; then
    export OMP_SCHEDULE="$SCHED:$CHUNK"
  else
    export OMP_SCHEDULE="$SCHED"
  fi
fi

# Build command
CMD=( ./conv2d_omp -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -m "$MODE" -p "$PADDING" -sH "$STRIDE_H" -sW "$STRIDE_W" )
CMD+=( --threads "$THREADS" --schedule "$SCHED" )
[ -n "$CHUNK" ] && [ "$CHUNK" -gt 0 ] 2>/dev/null && CMD+=( --chunk "$CHUNK" )
[ -n "$SEED" ] && CMD+=( --seed "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
