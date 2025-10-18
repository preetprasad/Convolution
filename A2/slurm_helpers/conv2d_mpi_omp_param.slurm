#!/bin/bash
#SBATCH --job-name=conv2d_mpi_omp_param
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --partition=cits3402
#SBATCH --output=logs/conv2d_mpi_omp_%j.out
#SBATCH --error=logs/conv2d_mpi_omp_%j.err

# Load compiler/MPI (tolerant & quiet). Some clusters use different module names.
if type module &>/dev/null; then
  module -q load gcc/14.2    2>/dev/null || module -q load gcc       2>/dev/null || echo "Note: gcc module not loaded; using system compiler" >&2
  module -q load openmpi/5.0.5 2>/dev/null || module -q load openmpi 2>/dev/null || echo "Note: openmpi module not loaded; relying on mpicc in PATH" >&2
fi

# Usage:
#   sbatch conv2d_mpi_omp_param.slurm H W kH kW [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]
#
# Positional:
#   H, W           : input matrix dimensions
#   kH, kW         : kernel dimensions
#
# Optional:
#   NP             : number of MPI processes (default: 4, must match --ntasks)
#   THREADS        : OpenMP threads per process (default: 4, must match --cpus-per-task)
#   SCHED          : OpenMP schedule type (default: static)
#   CHUNK          : OpenMP chunk size (optional, leave empty for default)
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   STRIDE_H       : vertical stride (default: 1)
#   STRIDE_W       : horizontal stride (default: 1)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv2d_mpi_omp ] || [ conv2d_mpi_omp.c -nt conv2d_mpi_omp ]; then
  mpicc -std=c11 -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra -o conv2d_mpi_omp conv2d_mpi_omp.c -lm || exit 1
fi

if [ $# -lt 4 ]; then
  echo "Usage: sbatch conv2d_mpi_omp_param.slurm H W kH kW [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]" >&2
  exit 2
fi

H="$1"
W="$2"
KH="$3"
KW="$4"
NP="${5:-4}"           # number of MPI processes
THREADS="${6:-4}"      # OpenMP threads per process
SCHED="${7:-static}"   # OpenMP schedule
CHUNK="${8:-}"         # OpenMP chunk size
MODE="${9:-same}"      # convolution mode
PADDING="${10:-zero}"  # padding mode
STRIDE_H="${11:-1}"    # vertical stride
STRIDE_W="${12:-1}"    # horizontal stride
SEED="${13:-}"         # RNG seed

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
CMD=( mpiexec -n "$NP" ./conv2d_mpi_omp -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -m "$MODE" -p "$PADDING" -sH "$STRIDE_H" -sW "$STRIDE_W" )
CMD+=( --threads "$THREADS" --schedule "$SCHED" )
[ -n "$CHUNK" ] && [ "$CHUNK" -gt 0 ] 2>/dev/null && CMD+=( --chunk "$CHUNK" )
[ -n "$SEED" ] && CMD+=( -se "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
