#!/bin/bash
#SBATCH --job-name=conv1d_mpi_omp_param
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --partition=cits3402
#SBATCH --output=logs/conv1d_mpi_omp_%j.out
#SBATCH --error=logs/conv1d_mpi_omp_%j.err

# Load compiler + MPI
module load gcc/14.2
module load openmpi/5.0.5

# Usage:
#   sbatch conv1d_mpi_omp_param.slurm N K [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE] [SEED]
#
# Positional:
#   N, K           : input and kernel lengths
#
# Optional:
#   NP             : number of MPI processes (default: 4, must match --ntasks)
#   THREADS        : OpenMP threads per process (default: 4, must match --cpus-per-task)
#   SCHED          : OpenMP schedule type (default: static)
#   CHUNK          : OpenMP chunk size (optional, leave empty for default)
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   STRIDE         : output stride (default: 1)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv1d_mpi_omp ] || [ conv1d_mpi_omp.c -nt conv1d_mpi_omp ]; then
  mpicc -std=c11 -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra -o conv1d_mpi_omp conv1d_mpi_omp.c -lm || exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: sbatch conv1d_mpi_omp_param.slurm N K [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE] [SEED]" >&2
  exit 2
fi

N="$1"
K="$2"
NP="${3:-4}"           # number of MPI processes
THREADS="${4:-4}"      # OpenMP threads per process
SCHED="${5:-static}"   # OpenMP schedule
CHUNK="${6:-}"         # OpenMP chunk size
MODE="${7:-same}"      # convolution mode
PADDING="${8:-zero}"   # padding mode
STRIDE="${9:-1}"       # output stride
SEED="${10:-}"         # RNG seed

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
CMD=( mpiexec -n "$NP" ./conv1d_mpi_omp -L "$N" -kL "$K" -o "$OUT" -m "$MODE" -p "$PADDING" -st "$STRIDE" )
CMD+=( --threads "$THREADS" --schedule "$SCHED" )
[ -n "$CHUNK" ] && [ "$CHUNK" -gt 0 ] 2>/dev/null && CMD+=( --chunk "$CHUNK" )
[ -n "$SEED" ] && CMD+=( -se "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
