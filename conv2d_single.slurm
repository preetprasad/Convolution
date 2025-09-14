#!/bin/bash
#SBATCH --job-name=conv2d_single
#SBATCH --partition=cits3402
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/conv2d_single_%j.out
#SBATCH --error=logs/conv2d_single_%j.err

# Compile
cc -std=c11 -Wall -Wextra -o conv2d conv2d.c || exit 1
mkdir -p results

# Args: [SEED]
SEED="${1:-}"

# Example dims (edit as you like)
H=11300; W=11300; KH=1130; KW=1130
OUT="results/Y.txt"   # write a real file here if you want an artifact

if [ -n "$SEED" ]; then
  ./conv2d -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -m same -p zero -s "$SEED"
else
  ./conv2d -H "$H" -W "$W" -kH "$KH" -kW "$KW" -o "$OUT" -m same -p zero
fi