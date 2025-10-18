#!/usr/bin/env bash
# Usage:
#   ./stress_sweep_serial_omp_2d.sh HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP \
#       [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]

set -euo pipefail

if [[ $# -lt 12 ]]; then
  echo "Usage: $0 HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]" >&2
  exit 1
fi

HMIN="$1"; HMAX="$2"; HSTEP="$3"
WMIN="$4"; WMAX="$5"; WSTEP="$6"
KHMIN="$7"; KHMAX="$8"; KHSTEP="$9"
KWMIN="${10}"; KWMAX="${11}"; KWSTEP="${12}"

POST_COPY_WAIT="${13:-10}"
FILE_WAIT_RETRIES="${14:-60}"
THREADS="${15:-8}"
SCHED="${16:-static}"
CHUNK="${17:-}"
MODE="${18:-same}"
PADDING="${19:-zero}"
SEED="${20:-}"

mkdir -p logs metrics

range_step() { local s="$1" e="$2" st="$3"; for ((x=s; x<=e; x+=st)); do echo "$x"; done; }

submit_and_block() {
  local H="$1" W="$2" KH="$3" KW="$4"
  local submit_out jobid
  submit_out=$(sbatch conv2d_omp_param.slurm "$H" "$W" "$KH" "$KW" "$THREADS" "$SCHED" "$CHUNK" "$MODE" "$PADDING" "$SEED")
  jobid=$(awk '{print $4}' <<<"$submit_out")

  echo "Submitted JOBID=$jobid (H=$H, W=$W, KH=$KH, KW=$KW)"
  local err="logs/conv2d_omp_${jobid}.err"
  local csv="metrics/o0/metrics_SLURM_${jobid}.csv"

  while squeue -j "$jobid" -h | grep -q . ; do sleep 10; done
  [ ! -s "$err" ] || [ ! -s "$csv" ] && sleep "$POST_COPY_WAIT"

  local tries=0
  until [[ -s "$err" && -s "$csv" ]]; do
    ((tries++))
    if (( tries > FILE_WAIT_RETRIES )); then
      echo "ERROR: Files not found for JOBID=$jobid" >&2; exit 4
    fi
    sleep 2
  done
  echo "OK: JOBID=$jobid completed (stderr+CSV found)"
}

for H in $(range_step "$HMIN" "$HMAX" "$HSTEP"); do
  for W in $(range_step "$WMIN" "$WMAX" "$WSTEP"); do
    for KH in $(range_step "$KHMIN" "$KHMAX" "$KHSTEP"); do
      for KW in $(range_step "$KWMIN" "$KWMAX" "$KWSTEP"); do
        submit_and_block "$H" "$W" "$KH" "$KW"
      done
    done
  done
done