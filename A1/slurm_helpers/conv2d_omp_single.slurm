#!/bin/bash
#SBATCH --job-name=conv2d_omp_single
#SBATCH --partition=cits3402
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/conv2d_omp_single_%j.out
#SBATCH --error=logs/conv2d_omp_single_%j.err

# Usage:
#   sbatch conv2d_omp_single.slurm [THREADS] [SCHED] [MODE] [PADDING] [SEED]

mkdir -p logs results
cc -std=c11 -Wall -Wextra -O3 -march=native -fopenmp -o conv2d_omp conv2d_omp.c || exit 1

THREADS="${1:-8}"
SCHED="${2:-static}"
MODE="${3:-same}"
PADDING="${4:-zero}"
SEED="${5:-}"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_SCHEDULE="$SCHED"

H=2048; W=2048; KH=7; KW=7
OUT="/dev/null"

CMD=( ./conv2d_omp -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -m "$MODE" -p "$PADDING" )
CMD+=( --threads "$THREADS" --schedule "$SCHED" )
[ -n "$SEED" ] && CMD+=( --seed "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
