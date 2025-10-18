#!/bin/bash
# Parallel SLURM sweep for conv1d_mpi (pure MPI, no OpenMP)
# Submits multiple jobs concurrently with throttling
#
# Usage:
#   ./sweep_conv1d_mpi.sh NMIN NMAX N_STEP KMIN KMAX K_STEP [MAX_IN_FLIGHT] [NP] [SEED]
#
# Example (default 20 concurrent, np=4):
#   ./sweep_conv1d_mpi.sh 100000 1000000 100000 101 1001 100
#
# Example (with custom parallelism):
#   ./sweep_conv1d_mpi.sh 100000 1000000 100000 101 1001 100 30 8 42

if [ $# -lt 6 ]; then
  echo "Usage: $0 NMIN NMAX N_STEP KMIN KMAX K_STEP [MAX_IN_FLIGHT] [NP] [SEED]" >&2
  exit 1
fi

NMIN="$1"; NMAX="$2"; NSTEP="$3"
KMIN="$4"; KMAX="$5"; KSTEP="$6"
MAX_IN_FLIGHT="${7:-20}"
NP="${8:-4}"
SEED="${9:-}"

for ((N=NMIN; N<=NMAX; N+=NSTEP)); do
  for ((K=KMIN; K<=KMAX; K+=KSTEP)); do

    # Throttle submissions
    while [ "$(squeue -u "$USER" | grep -c conv1d_mpi_param)" -ge "$MAX_IN_FLIGHT" ]; do
      sleep 5
    done

    sbatch slurm_helpers/conv1d_mpi_param.slurm "$N" "$K" "$NP" same zero "$SEED"
  done
done

echo "Submitted all jobs for N=$NMIN..$NMAX (step $NSTEP), K=$KMIN..$KMAX (step $KSTEP)"
echo "MPI config: NP=$NP"
