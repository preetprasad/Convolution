#!/bin/bash
# Usage:
#   ./stress_sweep_omp.sh LMIN LMAX KMIN KMAX [MAX_IN_FLIGHT] [THREADS] [SCHED] [CHUNK] [SEED]

if [ $# -lt 4 ]; then
  echo "Usage: $0 LMIN LMAX KMIN KMAX [MAX_IN_FLIGHT] [THREADS] [SCHED] [CHUNK] [SEED]" >&2
  exit 1
fi

LMIN="$1"; LMAX="$2"; KMIN="$3"; KMAX="$4"
MAX_IN_FLIGHT="${5:-20}"
THREADS="${6:-8}"
SCHED="${7:-static}"
CHUNK="${8:-}"
SEED="${9:-}"

for ((L=LMIN; L<=LMAX; L++)); do
  for ((K=KMIN; K<=KMAX; K++)); do

    while [ "$(squeue -u "$USER" | grep -c conv1d_omp_param)" -ge "$MAX_IN_FLIGHT" ]; do
      sleep 5
    done

    sbatch conv1d_omp_param.slurm "$L" "$K" "$THREADS" "$SCHED" "$CHUNK" "$SEED"
  done
done