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

# Load compiler
module load gcc/14.2

# Usage:
#   sbatch conv2d_seq_param.slurm H W kH kW [MODE] [PADDING] [SEED]
#
# Positional:
#   H, W           : input matrix dimensions
#   kH, kW         : kernel dimensions
#
# Optional:
#   MODE           : convolution mode (same|full, default: same)
#   PADDING        : padding mode (zero|none|const, default: zero)
#   SEED           : RNG seed (optional)

mkdir -p logs metrics

# Compile if needed
if [ ! -x ./conv2d ] || [ conv2d.c -nt conv2d ]; then
  cc -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra -o conv2d conv2d.c -lm || exit 1
fi

if [ $# -lt 4 ]; then
  echo "Usage: sbatch conv2d_seq_param.slurm H W kH kW [MODE] [PADDING] [SEED]" >&2
  exit 2
fi

H="$1"
W="$2"
KH="$3"
KW="$4"
MODE="${5:-same}"     # convolution mode
PADDING="${6:-zero}"  # padding mode
SEED="${7:-}"         # RNG seed

OUT="/dev/null"

# Build command
CMD=( ./conv2d -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -m "$MODE" -p "$PADDING" )
[ -n "$SEED" ] && CMD+=( -s "$SEED" )

echo "Running: ${CMD[*]}"
"${CMD[@]}"
