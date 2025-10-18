#!/bin/bash
# Usage:
#   ./stress_sweep.sh LMIN LMAX KMIN KMAX [SEED] [MAX_IN_FLIGHT]
#
# Example (no seed, default concurrency=20):
#   ./stress_sweep.sh 10 100 5 25
#
# Example (with seed=42, max 30 jobs in flight):
#   ./stress_sweep.sh 10 100 5 25 42 30

if [ $# -lt 4 ]; then
  echo "Usage: $0 LMIN LMAX KMIN KMAX [SEED] [MAX_IN_FLIGHT]" >&2
  exit 1
fi

LMIN="$1"
LMAX="$2"
KMIN="$3"
KMAX="$4"
SEED="${5:-}"           # optional
MAX_IN_FLIGHT="${6:-20}" # optional, default concurrency limit

for ((L=LMIN; L<=LMAX; L++)); do
  for ((K=KMIN; K<=KMAX; K++)); do

    # throttle submissions: wait if too many jobs in flight
    while [ "$(squeue -u "$USER" | grep -c conv1d_param)" -ge "$MAX_IN_FLIGHT" ]; do
      sleep 5
    done

    if [ -n "$SEED" ]; then
      sbatch conv1d_param.slurm "$L" "$K" "$SEED"
    else
      sbatch conv1d_param.slurm "$L" "$K"
    fi

  done
done