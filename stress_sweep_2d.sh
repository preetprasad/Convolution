#!/bin/bash
# Usage:
#   ./stress_sweep_2d.sh HMIN HMAX WMIN WMAX KHMIN KHMAX KWMIN KWMAX [SEED] [MAX_IN_FLIGHT]
#
# Example (no seed, default 20 concurrent):
#   ./stress_sweep_2d.sh 256 1024 256 1024 3 11 3 11
#
# Example (with seed=42, max 30 jobs):
#   ./stress_sweep_2d.sh 256 1024 256 1024 3 11 3 11 42 30

if [ $# -lt 8 ]; then
  echo "Usage: $0 HMIN HMAX WMIN WMAX KHMIN KHMAX KWMIN KWMAX [SEED] [MAX_IN_FLIGHT]" >&2
  exit 1
fi

HMIN="$1"; HMAX="$2"
WMIN="$3"; WMAX="$4"
KHMIN="$5"; KHMAX="$6"
KWMIN="$7"; KWMAX="$8"
SEED="${9:-}"
MAX_IN_FLIGHT="${10:-20}"

for ((H=HMIN; H<=HMAX; H++)); do
  for ((W=WMIN; W<=WMAX; W++)); do
    for ((KH=KHMIN; KH<=KHMAX; KH++)); do
      for ((KW=KWMIN; KW<=KWMAX; KW++)); do

        # throttle submissions: wait if too many jobs in flight
        while [ "$(squeue -u "$USER" | grep -c conv2d_param)" -ge "$MAX_IN_FLIGHT" ]; do
          sleep 5
        done

        if [ -n "$SEED" ]; then
          sbatch conv2d_param.slurm "$H" "$W" "$KH" "$KW" "$SEED"
        else
          sbatch conv2d_param.slurm "$H" "$W" "$KH" "$KW"
        fi

      done
    done
  done
done