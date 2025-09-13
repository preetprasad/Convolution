#!/bin/bash
#SBATCH --job-name=conv1d_param
#SBATCH --partition=cits3402
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/conv1d_%j.out
#SBATCH --error=logs/conv1d_%j.err

# Compile the program (if needed)
if [ ! -x ./conv1d ] || [ conv1d.c -nt conv1d ]; then
  cc -std=c11 -Wall -Wextra -Werror -o conv1d conv1d.c || exit 1
fi

# Validate args
if [ $# -lt 2 ]; then
  echo "Usage: sbatch conv1d_param.slurm <N> <K> [SEED]" >&2
  exit 2
fi
N="$1"
K="$2"
SEED="${3:-}"

# Ensure dirs exist
# mkdir -p metrics

# Separate dirs for different optimisation levels
# mkdir -p metrics/o0

# Write output array to /dev/null
OUT="/dev/null"

# Run with defaults except L, kL, out and optional seed
if [ -n "$SEED" ]; then
  ./conv1d -L "$N" -kL "$K" -o "$OUT" -s "$SEED"
else
  ./conv1d -L "$N" -kL "$K" -o "$OUT"
fi