#!/bin/bash
#SBATCH --job-name=conv1d_omp_param
#SBATCH --partition=cits3402
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/conv1d_omp_%j.out
#SBATCH --error=logs/conv1d_omp_%j.err

# Compile the program (if needed)
mkdir -p logs
if [ ! -x ./conv1d_omp ] || [ conv1d_omp.c -nt conv1d_omp ]; then
  cc -std=c11 -O3 -march=native -funroll-loops -fopt-info-vec -fopenmp -o conv1d_omp conv1d_omp.c || exit 1
fi

# Validate args
if [ $# -lt 2 ]; then
  echo "Usage: sbatch conv1d_omp_param.slurm <N> <K> [THREADS] [SCHED] [CHUNK] [SEED]" >&2
  exit 2
fi

N="$1"
K="$2"
THREADS="${3:-8}"
SCHED="${4:-static}"
CHUNK="${5:-}"
SEED="${6:-}"

# Output suppressed
OUT="/dev/null"

# OMP env
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
if [ "$SCHED" = "auto" ]; then
  export OMP_SCHEDULE=auto
else
  if [ "$CHUNK" -gt 0 ] 2>/dev/null; then
    export OMP_SCHEDULE="$SCHED:$CHUNK"
  else
    export OMP_SCHEDULE="$SCHED"
  fi
fi

# Build command
CMD=( ./conv1d_omp -L "$N" -kL "$K" -o "$OUT" -m same -p zero )
CMD+=( --threads "$THREADS" --schedule "$SCHED" )
if [ "$CHUNK" -gt 0 ] 2>/dev/null; then
  CMD+=( --chunk "$CHUNK" )
fi
[ -n "$SEED" ] && CMD+=( -s "$SEED" )

"${CMD[@]}"