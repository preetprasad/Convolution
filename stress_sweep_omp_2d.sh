#!/bin/bash
# Usage:
#   ./stress_sweep_omp_2d.sh HMIN HMAX WMIN WMAX KHMIN KHMAX KWMIN KWMAX [MAX_IN_FLIGHT] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]

if [ $# -lt 8 ]; then
  echo "Usage: $0 HMIN HMAX WMIN WMAX KHMIN KHMAX KWMIN KWMAX [MAX_IN_FLIGHT] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]" >&2
  exit 1
fi

HMIN="$1"; HMAX="$2"; WMIN="$3"; WMAX="$4"
KHMIN="$5"; KHMAX="$6"; KWMIN="$7"; KWMAX="$8"
MAX_IN_FLIGHT="${9:-20}"
THREADS="${10:-8}"
SCHED="${11:-static}"
CHUNK="${12:-}"
MODE="${13:-same}"
PADDING="${14:-zero}"
SEED="${15:-}"

submit() {
  local H="$1" W="$2" KH="$3" KW="$4"
  sbatch conv2d_omp_param.slurm "$H" "$W" "$KH" "$KW" "$THREADS" "$SCHED" "$CHUNK" "$MODE" "$PADDING" "$SEED"
}

for ((H=HMIN; H<=HMAX; H*=2)); do
  for ((W=WMIN; W<=WMAX; W*=2)); do
    for ((KH=KHMIN; KH<=KHMAX; KH+=2)); do
      for ((KW=KWMIN; KW<=KWMAX; KW+=2)); do
        while [ "$(squeue -u "$USER" | grep -c conv2d_omp_param)" -ge "$MAX_IN_FLIGHT" ]; do
          sleep 5
        done
        submit "$H" "$W" "$KH" "$KW"
      done
    done
  done
done