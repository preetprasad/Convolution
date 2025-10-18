#!/bin/bash
# Parallel SLURM sweep for conv1d_mpi_omp
# Submits multiple jobs concurrently with throttling
#
# Usage:
#   ./sweep_conv1d_mpi_omp.sh LMIN LMAX L_STEP KMIN KMAX K_STEP [MAX_IN_FLIGHT] [NP] [THREADS] [SCHED] [CHUNK] [SEED]
#
# Example (default 20 concurrent, np=4, threads=4):
#   ./sweep_conv1d_mpi_omp.sh 1000000 10000000 1000000 101 1001 100
#
# Example (with custom parallelism):
#   ./sweep_conv1d_mpi_omp.sh 1000000 10000000 1000000 101 1001 100 30 8 2 dynamic 16 42

if [ $# -lt 6 ]; then
  echo "Usage: $0 LMIN LMAX L_STEP KMIN KMAX K_STEP [MAX_IN_FLIGHT] [NP] [THREADS] [SCHED] [CHUNK] [SEED]" >&2
  exit 1
fi

LMIN="$1"; LMAX="$2"; LSTEP="$3"
KMIN="$4"; KMAX="$5"; KSTEP="$6"
MAX_IN_FLIGHT="${7:-20}"
NP="${8:-4}"
THREADS="${9:-4}"
SCHED="${10:-static}"
CHUNK="${11:-}"
SEED="${12:-}"

for ((L=LMIN; L<=LMAX; L+=LSTEP)); do
  for ((K=KMIN; K<=KMAX; K+=KSTEP)); do

    # Throttle submissions
    while [ "$(squeue -u "$USER" | grep -c conv1d_mpi_omp_param)" -ge "$MAX_IN_FLIGHT" ]; do
      sleep 5
    done

    sbatch slurm_helpers/conv1d_mpi_omp_param.slurm "$L" "$K" "$NP" "$THREADS" "$SCHED" "$CHUNK" same zero "$SEED"
  done
done

echo "Submitted all jobs for L=$LMIN..$LMAX (step $LSTEP), K=$KMIN..$KMAX (step $KSTEP)"
