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
module load gcc/14.2
module load openmpi/5.0.5

# Usage:
#   sbatch conv1d_mpi_param.slurm N K [NP] [MODE] [PADDING] [SEED]
#
# Positional:
#   N, K           : input and kernel lengths
#
# Optional:
#   NP             : number of MPI processes (default: 4, must match --ntasks)
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv1d_mpi ] || [ conv1d_mpi.c -nt conv1d_mpi ]; then
  mpicc -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra -o conv1d_mpi conv1d_mpi.c -lm || exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: sbatch conv1d_mpi_param.slurm N K [NP] [MODE] [PADDING] [SEED]" >&2
  exit 2
fi

N="$1"
K="$2"
NP="${3:-4}"          # number of MPI processes
MODE="${4:-same}"     # convolution mode
PADDING="${5:-zero}"  # padding mode
SEED="${6:-}"         # RNG seed

OUT="/dev/null"

# Build command
CMD=( mpiexec -n "$NP" ./conv1d_mpi -L "$N" -kL "$K" -o "$OUT" -m "$MODE" -p "$PADDING" )
[ -n "$SEED" ] && CMD+=( -s "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
