#!/bin/bash
#SBATCH --job-name=conv2d_mpi_param
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --partition=cits3402
#SBATCH --output=logs/conv2d_mpi_%j.out
#SBATCH --error=logs/conv2d_mpi_%j.err

# Load compiler/MPI (tolerant & quiet). Some clusters use different module names.
if type module &>/dev/null; then
  module -q load gcc/14.2    2>/dev/null || module -q load gcc       2>/dev/null || echo "Note: gcc module not loaded; using system compiler" >&2
  module -q load openmpi/5.0.5 2>/dev/null || module -q load openmpi 2>/dev/null || echo "Note: openmpi module not loaded; relying on mpicc in PATH" >&2
fi

# Usage:
#   sbatch conv2d_mpi_param.slurm H W kH kW [NP] [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]
#
# Positional:
#   H, W           : input matrix dimensions
#   kH, kW         : kernel dimensions
#
# Optional:
#   NP             : number of MPI processes (default: 4, must match --ntasks)
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   STRIDE_H       : vertical stride (default: 1)
#   STRIDE_W       : horizontal stride (default: 1)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv2d_mpi ] || [ conv2d_mpi.c -nt conv2d_mpi ]; then
  mpicc -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra -o conv2d_mpi conv2d_mpi.c -lm || exit 1
fi

if [ $# -lt 4 ]; then
  echo "Usage: sbatch conv2d_mpi_param.slurm H W kH kW [NP] [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]" >&2
  exit 2
fi

H="$1"
W="$2"
KH="$3"
KW="$4"
NP="${5:-4}"          # number of MPI processes
MODE="${6:-same}"     # convolution mode
PADDING="${7:-zero}"  # padding mode
STRIDE_H="${8:-1}"    # vertical stride
STRIDE_W="${9:-1}"    # horizontal stride
SEED="${10:-}"        # RNG seed

OUT="/dev/null"

# Build command
CMD=( mpiexec -n "$NP" ./conv2d_mpi -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -m "$MODE" -p "$PADDING" -sH "$STRIDE_H" -sW "$STRIDE_W" )
[ -n "$SEED" ] && CMD+=( -se "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
