#!/bin/bash
# Parallel SLURM sweep for conv1d_omp (OpenMP)
# Submits multiple jobs concurrently with throttling
#
# Usage:
#   ./sweep_conv1d_omp.sh NMIN NMAX N_STEP KMIN KMAX K_STEP [MAX_IN_FLIGHT] [THREADS] [SCHED] [CHUNK] [SEED]
#
# Example (default 20 concurrent, 8 threads, static schedule):
#   ./sweep_conv1d_omp.sh 100000 1000000 100000 101 1001 100
#
# Example (with custom parallelism):
#   ./sweep_conv1d_omp.sh 100000 1000000 100000 101 1001 100 30 16 dynamic 10 42

if [ $# -lt 6 ]; then
  echo "Usage: $0 NMIN NMAX N_STEP KMIN KMAX K_STEP [MAX_IN_FLIGHT] [THREADS] [SCHED] [CHUNK] [SEED]" >&2
  exit 1
fi

NMIN="$1"; NMAX="$2"; NSTEP="$3"
KMIN="$4"; KMAX="$5"; KSTEP="$6"
MAX_IN_FLIGHT="${7:-20}"
THREADS="${8:-8}"
SCHED="${9:-static}"
CHUNK="${10:-}"
SEED="${11:-}"

for ((N=NMIN; N<=NMAX; N+=NSTEP)); do
  for ((K=KMIN; K<=KMAX; K+=KSTEP)); do

    # Throttle submissions
    while [ "$(squeue -u "$USER" | grep -c conv1d_omp_param)" -ge "$MAX_IN_FLIGHT" ]; do
      sleep 5
    done

    sbatch slurm_helpers/conv1d_omp_param.slurm "$N" "$K" "$THREADS" "$SCHED" "$CHUNK" same zero "$SEED"
  done
done

echo "Submitted all jobs for N=$NMIN..$NMAX (step $NSTEP), K=$KMIN..$KMAX (step $KSTEP)"
echo "OpenMP config: THREADS=$THREADS, SCHED=$SCHED, CHUNK=${CHUNK:-default}"
