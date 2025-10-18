#!/bin/bash
# Parallel SLURM sweep for conv2d (sequential baseline)
# Submits multiple jobs concurrently with throttling
#
# Usage:
#   ./sweep_conv2d_seq.sh HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [MAX_IN_FLIGHT] [SEED]
#
# Example (default 20 concurrent):
#   ./sweep_conv2d_seq.sh 1024 4096 1024 1024 4096 1024 3 11 2 3 11 2
#
# Example (with custom concurrency):
#   ./sweep_conv2d_seq.sh 1024 4096 1024 1024 4096 1024 3 11 2 3 11 2 30 42

if [ $# -lt 12 ]; then
  echo "Usage: $0 HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [MAX_IN_FLIGHT] [SEED]" >&2
  exit 1
fi

HMIN="$1"; HMAX="$2"; HSTEP="$3"
WMIN="$4"; WMAX="$5"; WSTEP="$6"
KHMIN="$7"; KHMAX="$8"; KHSTEP="$9"
KWMIN="${10}"; KWMAX="${11}"; KWSTEP="${12}"
MAX_IN_FLIGHT="${13:-20}"
SEED="${14:-}"

for ((H=HMIN; H<=HMAX; H+=HSTEP)); do
  for ((W=WMIN; W<=WMAX; W+=WSTEP)); do
    for ((KH=KHMIN; KH<=KHMAX; KH+=KHSTEP)); do
      for ((KW=KWMIN; KW<=KWMAX; KW+=KWSTEP)); do

        # Throttle submissions
        while [ "$(squeue -u "$USER" | grep -c conv2d_seq_param)" -ge "$MAX_IN_FLIGHT" ]; do
          sleep 5
        done

        if [ -n "$SEED" ]; then
          sbatch slurm_helpers/conv2d_seq_param.slurm "$H" "$W" "$KH" "$KW" same zero "$SEED"
        else
          sbatch slurm_helpers/conv2d_seq_param.slurm "$H" "$W" "$KH" "$KW" same zero
        fi
      done
    done
  done
done

echo "Submitted all jobs for H=$HMIN..$HMAX, W=$WMIN..$WMAX, kH=$KHMIN..$KHMAX, kW=$KWMIN..$KWMAX"
