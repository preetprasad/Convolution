#!/bin/bash
# Parallel SLURM sweep for conv2d_mpi
# Submits multiple jobs concurrently with throttling
#
# Usage:
#   ./sweep_conv2d_mpi.sh HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [MAX_IN_FLIGHT] [NP] [SEED]
#
# Example (default 20 concurrent, np=4):
#   ./sweep_conv2d_mpi.sh 1024 4096 1024 1024 4096 1024 3 11 2 3 11 2
#
# Example (with custom parallelism):
#   ./sweep_conv2d_mpi.sh 1024 4096 1024 1024 4096 1024 3 11 2 3 11 2 30 9 42

if [ $# -lt 12 ]; then
  echo "Usage: $0 HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [MAX_IN_FLIGHT] [NP] [SEED]" >&2
  exit 1
fi

HMIN="$1"; HMAX="$2"; HSTEP="$3"
WMIN="$4"; WMAX="$5"; WSTEP="$6"
KHMIN="$7"; KHMAX="$8"; KHSTEP="$9"
KWMIN="${10}"; KWMAX="${11}"; KWSTEP="${12}"
MAX_IN_FLIGHT="${13:-20}"
NP="${14:-4}"
SEED="${15:-}"

for ((H=HMIN; H<=HMAX; H+=HSTEP)); do
  for ((W=WMIN; W<=WMAX; W+=WSTEP)); do
    for ((KH=KHMIN; KH<=KHMAX; KH+=KHSTEP)); do
      for ((KW=KWMIN; KW<=KWMAX; KW+=KWSTEP)); do

        # Throttle submissions
        while [ "$(squeue -u "$USER" | grep -c conv2d_mpi_param)" -ge "$MAX_IN_FLIGHT" ]; do
          sleep 5
        done

        sbatch slurm_helpers/conv2d_mpi_param.slurm "$H" "$W" "$KH" "$KW" "$NP" same zero "$SEED"
      done
    done
  done
done

echo "Submitted all jobs for H=$HMIN..$HMAX, W=$WMIN..$WMAX, kH=$KHMIN..$KHMAX, kW=$KWMIN..$KWMAX"
