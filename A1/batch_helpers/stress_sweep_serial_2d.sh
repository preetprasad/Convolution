#!/usr/bin/env bash
# Serial SLURM sweep for conv2d: stepped loops over H,W and kH,kW.
# Submits one job at a time and blocks until BOTH stderr and metrics CSV appear.
#
# Usage:
#   ./stress_sweep_serial_2d.sh HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]
#
# Positional:
#   HMIN HMAX H_STEP   : inclusive stepped range for -H
#   WMIN WMAX W_STEP   : inclusive stepped range for -W
#   KHMIN KHMAX KH_STEP: inclusive stepped range for -kH
#   KWMIN KWMAX KW_STEP: inclusive stepped range for -kW
#
# Optional:
#   POST_COPY_WAIT     : seconds to wait after job leaves queue unless files exist (default: 10)
#   FILE_WAIT_RETRIES  : how many 2s retries for files (default: 60 ≈ 120s)
#   SEED               : RNG seed forwarded to conv2d (omit to use program default)

set -euo pipefail

if [[ $# -lt 12 ]]; then
  echo "Usage: $0 HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]" >&2
  exit 1
fi

HMIN="$1"; HMAX="$2"; HSTEP="$3"
WMIN="$4"; WMAX="$5"; WSTEP="$6"
KHMIN="$7"; KHMAX="$8"; KHSTEP="$9"
KWMIN="${10}"; KWMAX="${11}"; KWSTEP="${12}"
POST_COPY_WAIT="${13:-10}"     # seconds
FILE_WAIT_RETRIES="${14:-60}"  # each retry waits 2s
SEED="${15:-}"                 # optional

# Basic validation
for v in "$HMIN" "$HMAX" "$HSTEP" "$WMIN" "$WMAX" "$WSTEP" "$KHMIN" "$KHMAX" "$KHSTEP" "$KWMIN" "$KWMAX" "$KWSTEP" "$POST_COPY_WAIT" "$FILE_WAIT_RETRIES"; do
  [[ "$v" =~ ^-?[0-9]+$ ]] || { echo "Error: non-integer argument: $v" >&2; exit 2; }
done
(( HSTEP > 0 )) || { echo "Error: H_STEP must be > 0" >&2; exit 2; }
(( WSTEP > 0 )) || { echo "Error: W_STEP must be > 0" >&2; exit 2; }
(( KHSTEP > 0 )) || { echo "Error: KH_STEP must be > 0" >&2; exit 2; }
(( KWSTEP > 0 )) || { echo "Error: KW_STEP must be > 0" >&2; exit 2; }

echo "Sweep config:"
echo "  H: $HMIN..$HMAX step $HSTEP"
echo "  W: $WMIN..$WMAX step $WSTEP"
echo "  kH: $KHMIN..$KHMAX step $KHSTEP"
echo "  kW: $KWMIN..$KWMAX step $KWSTEP"
echo "  POST_COPY_WAIT=${POST_COPY_WAIT}s  FILE_WAIT_RETRIES=$FILE_WAIT_RETRIES  SEED=${SEED:-<default>}"

mkdir -p logs metrics/o0 results/o0

range_step() { local s="$1" e="$2" st="$3"; local x; for ((x=s; x<=e; x+=st)); do echo "$x"; done; }

submit_and_block() {
  local H="$1" W="$2" KH="$3" KW="$4"

  local submit_out jobid
  if [[ -n "$SEED" ]]; then
    submit_out=$(sbatch conv2d_param.slurm "$H" "$W" "$KH" "$KW" "$SEED")
  else
    submit_out=$(sbatch conv2d_param.slurm "$H" "$W" "$KH" "$KW")
  fi

  jobid=$(awk '{print $4}' <<<"$submit_out")
  [[ -n "${jobid:-}" ]] || { echo "Failed to parse job id from: $submit_out" >&2; exit 3; }
  echo "Submitted JOBID=$jobid  (H=$H, W=$W, kH=$KH, kW=$KW)"

  local err="logs/conv2d_${jobid}.err"
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

for H in $(range_step "$HMIN" "$HMAX" "$HSTEP"); do
  echo "=== H=$H ==="
  for W in $(range_step "$WMIN" "$WMAX" "$WSTEP"); do
    echo " -> W=$W"
    for KH in $(range_step "$KHMIN" "$KHMAX" "$KHSTEP"); do
      for KW in $(range_step "$KWMIN" "$KWMAX" "$KWSTEP"); do
        submit_and_block "$H" "$W" "$KH" "$KW"
      done
    done
  done
done

echo "All jobs completed and verified."