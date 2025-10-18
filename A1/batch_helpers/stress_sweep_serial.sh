#!/usr/bin/env bash
# Serial SLURM sweep for conv1d: stepped loops over L and K.
# Submits one job at a time and blocks until BOTH stderr and metrics CSV appear.
#
# Usage:
#   ./stress_sweep_serial.sh LMIN LMAX L_STEP KMIN KMAX K_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]
#
# Positional:
#   LMIN LMAX L_STEP   : inclusive stepped range for -L
#   KMIN KMAX K_STEP   : inclusive stepped range for -kL
#
# Optional:
#   POST_COPY_WAIT     : seconds to wait after job leaves queue unless files exist (default: 10)
#   FILE_WAIT_RETRIES  : how many 2s retries for files (default: 60 ≈ 120s)
#   SEED               : RNG seed forwarded to conv1d (omit to use program default)

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 LMIN LMAX L_STEP KMIN KMAX K_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]" >&2
  exit 1
fi

LMIN="$1"; LMAX="$2"; LSTEP="$3"
KMIN="$4"; KMAX="$5"; KSTEP="$6"
POST_COPY_WAIT="${7:-10}"     # seconds
FILE_WAIT_RETRIES="${8:-60}"  # each retry waits 2s
SEED="${9:-}"                 # optional

# Validation
for v in "$LMIN" "$LMAX" "$LSTEP" "$KMIN" "$KMAX" "$KSTEP" "$POST_COPY_WAIT" "$FILE_WAIT_RETRIES"; do
  [[ "$v" =~ ^-?[0-9]+$ ]] || { echo "Error: non-integer argument: $v" >&2; exit 2; }
done
(( LSTEP > 0 )) || { echo "Error: L_STEP must be > 0" >&2; exit 2; }
(( KSTEP > 0 )) || { echo "Error: K_STEP must be > 0" >&2; exit 2; }

echo "Sweep config:"
echo "  L: $LMIN..$LMAX step $LSTEP"
echo "  K: $KMIN..$KMAX step $KSTEP"
echo "  POST_COPY_WAIT=${POST_COPY_WAIT}s  FILE_WAIT_RETRIES=$FILE_WAIT_RETRIES  SEED=${SEED:-<default>}"

mkdir -p logs metrics/o0 results/o0

range_step() {
  local start="$1" end="$2" step="$3" x
  for ((x=start; x<=end; x+=step)); do
    echo "$x"
  done
}

submit_and_block() {
  local L="$1" K="$2"

  local submit_out jobid
  if [[ -n "$SEED" ]]; then
    submit_out=$(sbatch conv1d_param.slurm "$L" "$K" "$SEED")
  else
    submit_out=$(sbatch conv1d_param.slurm "$L" "$K")
  fi

  jobid=$(awk '{print $4}' <<<"$submit_out")
  [[ -n "${jobid:-}" ]] || { echo "Failed to parse job id from: $submit_out" >&2; exit 3; }
  echo "Submitted JOBID=$jobid  (L=$L, K=$K)"

  local err="logs/conv1d_${jobid}.err"
  local csv="metrics/o0/metrics_SLURM_${jobid}.csv"

  # Wait while job is still in queue/running
  while squeue -j "$jobid" -h | grep -q . ; do
    sleep 10
  done

  # If files already present, skip POST_COPY_WAIT; else wait a bit
  if [[ ! (-s "$err" && -s "$csv") ]]; then
    sleep "$POST_COPY_WAIT"
  fi

  # Retry for files to appear
  local tries=0
  until [[ -s "$err" && -s "$csv" ]]; do
    (( tries++ ))
    if (( tries > FILE_WAIT_RETRIES )); then
      echo "ERROR: Files not found for JOBID=$jobid after waiting." >&2
      [[ -s "$err" ]] || echo "  Missing: $err" >&2
      [[ -s "$csv" ]] || echo "  Missing: $csv" >&2
      exit 4
    fi
    sleep 2
  done

  echo "OK: Found stderr ($err) and metrics CSV ($csv) for JOBID=$jobid"
}

for L in $(range_step "$LMIN" "$LMAX" "$LSTEP"); do
  echo "=== L=$L ==="
  for K in $(range_step "$KMIN" "$KMAX" "$KSTEP"); do
    echo " -> K=$K"
    submit_and_block "$L" "$K"
  done
done

echo "All jobs completed and verified."