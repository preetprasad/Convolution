#!/bin/bash
#SBATCH --job-name=conv1d_seq_param
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --partition=cits3402
#SBATCH --output=logs/conv1d_%j.out
#SBATCH --error=logs/conv1d_%j.err

# Load compiler (tolerant & quiet). Some clusters don't have gcc/14.2.
if type module &>/dev/null; then
  module -q load gcc/14.2 2>/dev/null || \
  module -q load gcc       2>/dev/null || \
  echo "Note: gcc module not loaded; using system compiler (cc)" >&2
fi

# Usage:
#   sbatch conv1d_seq_param.slurm N K [MODE] [PADDING] [STRIDE] [SEED]
#
# Positional:
#   N, K           : input and kernel lengths
#
# Optional:
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   STRIDE         : output stride (default: 1)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv1d ] || [ conv1d.c -nt conv1d ]; then
  cc -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra -o conv1d conv1d.c -lm || exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: sbatch conv1d_seq_param.slurm N K [MODE] [PADDING] [STRIDE] [SEED]" >&2
  exit 2
fi

N="$1"
K="$2"
MODE="${3:-same}"     # convolution mode
PADDING="${4:-zero}"  # padding mode
STRIDE="${5:-1}"      # output stride
SEED="${6:-}"         # RNG seed

OUT="/dev/null"

# Build command
CMD=( ./conv1d -L "$N" -kL "$K" -o "$OUT" -m "$MODE" -p "$PADDING" -st "$STRIDE" )
[ -n "$SEED" ] && CMD+=( -s "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
