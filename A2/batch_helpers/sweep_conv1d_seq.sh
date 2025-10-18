#!/bin/bash
# Parallel SLURM sweep for conv1d (sequential baseline)
# Submits multiple jobs concurrently with throttling
#
# Usage:
#   ./sweep_conv1d_seq.sh NMIN NMAX N_STEP KMIN KMAX K_STEP [MAX_IN_FLIGHT] [SEED]
#
# Example (default 20 concurrent):
#   ./sweep_conv1d_seq.sh 100000 1000000 100000 101 1001 100
#
# Example (with custom concurrency and seed):
#   ./sweep_conv1d_seq.sh 100000 1000000 100000 101 1001 100 30 42

if [ $# -lt 6 ]; then
  echo "Usage: $0 NMIN NMAX N_STEP KMIN KMAX K_STEP [MAX_IN_FLIGHT] [SEED]" >&2
  exit 1
fi

NMIN="$1"; NMAX="$2"; NSTEP="$3"
KMIN="$4"; KMAX="$5"; KSTEP="$6"
MAX_IN_FLIGHT="${7:-20}"
SEED="${8:-}"

for ((N=NMIN; N<=NMAX; N+=NSTEP)); do
  for ((K=KMIN; K<=KMAX; K+=KSTEP)); do

    # Throttle submissions
    while [ "$(squeue -u "$USER" | grep -c conv1d_seq_param)" -ge "$MAX_IN_FLIGHT" ]; do
      sleep 5
    done

    if [ -n "$SEED" ]; then
      sbatch slurm_helpers/conv1d_seq_param.slurm "$N" "$K" same zero "$SEED"
    else
      sbatch slurm_helpers/conv1d_seq_param.slurm "$N" "$K" same zero
    fi
  done
done

echo "Submitted all jobs for N=$NMIN..$NMAX (step $NSTEP), K=$KMIN..$KMAX (step $KSTEP)"
