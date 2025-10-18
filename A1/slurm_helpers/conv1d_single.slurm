#!/bin/bash
#SBATCH --job-name=conv1d_single
#SBATCH --partition=cits3402
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/conv1d_single_%j.out
#SBATCH --error=logs/conv1d_single_%j.err

# Compile the program
cc -std=c11 -Wall -Wextra -o conv1d conv1d.c

# Create output directory
mkdir -p results

# Run a single test with chosen lengths
./conv1d --len 1000000 --klen 10001 --out results/y.txt --seed 42 --mode same --padding zero