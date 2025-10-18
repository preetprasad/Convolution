#!/bin/bash
#SBATCH --job-name=conv2d_omp_param
#SBATCH --partition=cits3402
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/conv2d_omp_%j.out
#SBATCH --error=logs/conv2d_omp_%j.err

# Usage:
#   sbatch conv2d_omp_param.slurm H W kH kW [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]

mkdir -p logs

# Compile if needed
if [ ! -x ./conv2d_omp ] || [ conv2d_omp.c -nt conv2d_omp ]; then
  cc -O3 -march=native -funroll-loops -fopt-info-vec -fopenmp -Wall -Wextra -o conv2d_omp conv2d_omp.c || exit 1
fi

if [ $# -lt 4 ]; then
  echo "Usage: sbatch conv2d_omp_param.slurm H W kH kW [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]" >&2
  exit 2
fi

H="$1"; W="$2"; KH="$3"; KW="$4"
THREADS="${5:-8}"
SCHED="${6:-static}"
CHUNK="${7:-}"
MODE="${8:-same}"        # same|full
PADDING="${9:-zero}"     # zero|none|const
SEED="${10:-}"

OUT="/dev/null"

# OMP env
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

CMD=( ./conv2d_omp -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -m "$MODE" -p "$PADDING" )
CMD+=( --threads "$THREADS" --schedule "$SCHED" )
[ -n "$CHUNK" ] && [ "$CHUNK" -gt 0 ] 2>/dev/null && CMD+=( --chunk "$CHUNK" )
[ -n "$SEED" ] && CMD+=( --seed "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"