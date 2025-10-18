#!/bin/bash
#SBATCH --job-name=conv2d_seq_param
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --partition=cits3402
#SBATCH --output=logs/conv2d_%j.out
#SBATCH --error=logs/conv2d_%j.err

# Load compiler (tolerant & quiet). Some clusters don't have gcc/14.2.
if type module &>/dev/null; then
  module -q load gcc/14.2 2>/dev/null || \
  module -q load gcc       2>/dev/null || \
  echo "Note: gcc module not loaded; using system compiler (cc)" >&2
fi

# Usage:
#   sbatch conv2d_seq_param.slurm H W kH kW [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]
#
# Positional:
#   H, W           : input matrix dimensions
#   kH, kW         : kernel dimensions
#
# Optional:
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   STRIDE_H       : vertical stride (default: 1)
#   STRIDE_W       : horizontal stride (default: 1)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv2d ] || [ conv2d.c -nt conv2d ]; then
  cc -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra -o conv2d conv2d.c -lm || exit 1
fi

if [ $# -lt 4 ]; then
  echo "Usage: sbatch conv2d_seq_param.slurm H W kH kW [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]" >&2
  exit 2
fi

H="$1"
W="$2"
KH="$3"
KW="$4"
MODE="${5:-same}"     # convolution mode
PADDING="${6:-zero}"  # padding mode
STRIDE_H="${7:-1}"    # vertical stride
STRIDE_W="${8:-1}"    # horizontal stride
SEED="${9:-}"         # RNG seed

OUT="/dev/null"

# Build command
CMD=( ./conv2d -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -m "$MODE" -p "$PADDING" -sH "$STRIDE_H" -sW "$STRIDE_W" )
[ -n "$SEED" ] && CMD+=( -s "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
