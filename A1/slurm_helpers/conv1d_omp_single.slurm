#!/bin/bash
#SBATCH --job-name=conv1d_omp_single
#SBATCH --partition=cits3402
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/conv1d_omp_single_%j.out
#SBATCH --error=logs/conv1d_omp_single_%j.err

mkdir -p logs results

# Compile
cc -std=c11 -Wall -Wextra -fopenmp -o conv1d_omp conv1d_omp.c

# Args
THREADS="${1:-8}"
SCHED="${2:-static}"
SEED="${3:-}"

# OMP env
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_SCHEDULE="$SCHED"

# Single run
CMD=( ./conv1d_omp --len 1070409999 --klen 8399 --out /dev/null --mode same --padding zero )
CMD+=( --threads "$THREADS" --schedule "$SCHED" )
[ -n "$SEED" ] && CMD+=( --seed "$SEED" )

"${CMD[@]}"