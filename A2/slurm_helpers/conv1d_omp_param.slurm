#!/bin/bash
#SBATCH --job-name=conv1d_omp_param
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --partition=cits3402
#SBATCH --output=logs/conv1d_omp_%j.out
#SBATCH --error=logs/conv1d_omp_%j.err

# Load compiler (tolerant & quiet). Some clusters don't have gcc/14.2.
if type module &>/dev/null; then
  module -q load gcc/14.2 2>/dev/null || \
  module -q load gcc       2>/dev/null || \
  echo "Note: gcc module not loaded; using system compiler (cc)" >&2
fi

# Usage:
#   sbatch conv1d_omp_param.slurm N K [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE] [SEED]
#
# Positional:
#   N, K           : input and kernel lengths
#
# Optional:
#   THREADS        : OpenMP threads (default: 8, must match --cpus-per-task)
#   SCHED          : OpenMP schedule type (default: static)
#   CHUNK          : OpenMP chunk size (optional, leave empty for default)
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   STRIDE         : output stride (default: 1)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv1d_omp ] || [ conv1d_omp.c -nt conv1d_omp ]; then
  cc -std=c11 -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra -o conv1d_omp conv1d_omp.c -lm || exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: sbatch conv1d_omp_param.slurm N K [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE] [SEED]" >&2
  exit 2
fi

N="$1"
K="$2"
THREADS="${3:-8}"      # OpenMP threads
SCHED="${4:-static}"   # OpenMP schedule
CHUNK="${5:-}"         # OpenMP chunk size
MODE="${6:-same}"      # convolution mode
PADDING="${7:-zero}"   # padding mode
STRIDE="${8:-1}"       # output stride
SEED="${9:-}"          # RNG seed

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
CMD=( ./conv1d_omp -L "$N" -kL "$K" -o "$OUT" -m "$MODE" -p "$PADDING" -st "$STRIDE" )
CMD+=( --threads "$THREADS" --schedule "$SCHED" )
[ -n "$CHUNK" ] && [ "$CHUNK" -gt 0 ] 2>/dev/null && CMD+=( --chunk "$CHUNK" )
[ -n "$SEED" ] && CMD+=( -s "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
