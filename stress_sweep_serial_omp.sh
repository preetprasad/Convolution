#!/usr/bin/env bash
# Usage:
#   ./stress_sweep_serial_omp.sh LMIN LMAX L_STEP KMIN KMAX K_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [THREADS] [SCHED] [CHUNK] [SEED]

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 LMIN LMAX L_STEP KMIN KMAX K_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [THREADS] [SCHED] [CHUNK] [SEED]" >&2
  exit 1
fi

LMIN="$1"; LMAX="$2"; LSTEP="$3"
KMIN="$4"; KMAX="$5"; KSTEP="$6"
POST_COPY_WAIT="${7:-10}"
FILE_WAIT_RETRIES="${8:-60}"
THREADS="${9:-8}"
SCHED="${10:-static}"
CHUNK="${11:-}"
SEED="${12:-}"

#mkdir -p logs metrics/omp/o3 results/omp/o3

range_step() { local s="$1" e="$2" st="$3"; for ((x=s; x<=e; x+=st)); do echo "$x"; done; }

submit_and_block() {
  local L="$1" K="$2"
  local submit_out jobid
  submit_out=$(sbatch conv1d_omp_param.slurm "$L" "$K" "$THREADS" "$SCHED" "$CHUNK" "$SEED")
  jobid=$(awk '{print $4}' <<<"$submit_out")

  echo "Submitted JOBID=$jobid (L=$L, K=$K)"
  local err="logs/conv1d_omp_${jobid}.err"
  local csv="metrics/omp/o0/metrics_SLURM_${jobid}.csv"

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

for L in $(range_step "$LMIN" "$LMAX" "$LSTEP"); do
  for K in $(range_step "$KMIN" "$KMAX" "$KSTEP"); do
    submit_and_block "$L" "$K"
  done
done