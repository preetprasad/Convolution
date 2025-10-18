#!/bin/bash
#SBATCH --job-name=conv1d_mpi_param
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --partition=cits3402
#SBATCH --output=logs/conv1d_mpi_%j.out
#SBATCH --error=logs/conv1d_mpi_%j.err

# Load compiler + MPI
# Load compiler/MPI (tolerant & quiet). Some clusters use different module names.
if type module &>/dev/null; then
  module -q load gcc/14.2    2>/dev/null || module -q load gcc       2>/dev/null || echo "Note: gcc module not loaded; using system compiler" >&2
  module -q load openmpi/5.0.5 2>/dev/null || module -q load openmpi 2>/dev/null || echo "Note: openmpi module not loaded; relying on mpicc in PATH" >&2
fi

# Usage:
#   sbatch conv1d_mpi_param.slurm N K [NP] [MODE] [PADDING] [STRIDE] [SEED]
#
# Positional:
#   N, K           : input and kernel lengths
#
# Optional:
#   NP             : number of MPI processes (default: 4, must match --ntasks)
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   STRIDE         : output stride (default: 1)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv1d_mpi ] || [ conv1d_mpi.c -nt conv1d_mpi ]; then
  mpicc -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra -o conv1d_mpi conv1d_mpi.c -lm || exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: sbatch conv1d_mpi_param.slurm N K [NP] [MODE] [PADDING] [STRIDE] [SEED]" >&2
  exit 2
fi

N="$1"
K="$2"
NP="${3:-4}"          # number of MPI processes
MODE="${4:-same}"     # convolution mode
PADDING="${5:-zero}"  # padding mode
STRIDE="${6:-1}"      # output stride
SEED="${7:-}"         # RNG seed

OUT="/dev/null"

# Build command
CMD=( mpiexec -n "$NP" ./conv1d_mpi -L "$N" -kL "$K" -o "$OUT" -m "$MODE" -p "$PADDING" -st "$STRIDE" )
[ -n "$SEED" ] && CMD+=( -se "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
