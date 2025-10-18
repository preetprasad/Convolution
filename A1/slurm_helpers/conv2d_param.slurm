#!/bin/bash
#SBATCH --job-name=conv2d_param
#SBATCH --partition=cits3402
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/conv2d_%j.out
#SBATCH --error=logs/conv2d_%j.err

# Compile the program (if needed)
if [ ! -x ./conv2d ] || [ conv2d.c -nt conv2d ]; then
  cc -std=c11 -O3 -march=native -funroll-loops -fopt-info-vec -Wall -Wextra -o conv2d conv2d.c || exit 1
fi

# Args: H W kH kW [SEED]
if [ $# -lt 4 ]; then
  echo "Usage: sbatch conv2d_param.slurm <H> <W> <kH> <kW> [SEED]" >&2
  exit 2
fi
H="$1"
W="$2"
KH="$3"
KW="$4"
SEED="${5:-}"

mkdir -p logs metrics/o0

OUT="/dev/null"

# Run with defaults (mode=same, padding=zero); pass seed if provided
if [ -n "$SEED" ]; then
  ./conv2d -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -s "$SEED"
else
  ./conv2d -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT"
fi