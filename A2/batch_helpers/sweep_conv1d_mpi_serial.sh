#!/usr/bin/env bash
# Serial SLURM sweep for conv1d_mpi (pure MPI, no OpenMP)
# Submits one job at a time and blocks until BOTH stderr and metrics CSV appear.
#
# Usage:
#   ./sweep_conv1d_mpi_serial.sh NMIN NMAX N_STEP KMIN KMAX K_STEP [NP] [STRIDE] [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]
#
# Positional:
#   NMIN NMAX N_STEP   : inclusive stepped range for -L (input length)
#   KMIN KMAX K_STEP   : inclusive stepped range for -kL (kernel length)
#
# Optional:
#   NP                 : Number of MPI processes (default: 4)
#   STRIDE             : output stride (default: 1)
#   POST_COPY_WAIT     : seconds to wait after job leaves queue unless files exist (default: 10)
#   FILE_WAIT_RETRIES  : how many 2s retries for files (default: 60 ≈ 120s)
#   SEED               : RNG seed forwarded to conv1d_mpi (omit to use program default)

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 NMIN NMAX N_STEP KMIN KMAX K_STEP [NP] [STRIDE] [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]" >&2
  exit 1
fi

NMIN="$1"; NMAX="$2"; NSTEP="$3"
KMIN="$4"; KMAX="$5"; KSTEP="$6"
NP="${7:-4}"
STRIDE="${8:-1}"              # output stride
POST_COPY_WAIT="${9:-10}"     # seconds
FILE_WAIT_RETRIES="${10:-60}"  # each retry waits 2s
SEED="${11:-}"                # optional

# Validation
for v in "$NMIN" "$NMAX" "$NSTEP" "$KMIN" "$KMAX" "$KSTEP" "$NP" "$POST_COPY_WAIT" "$FILE_WAIT_RETRIES"; do
  [[ "$v" =~ ^-?[0-9]+$ ]] || { echo "Error: non-integer argument: $v" >&2; exit 2; }
done
(( NSTEP > 0 )) || { echo "Error: N_STEP must be > 0" >&2; exit 2; }
(( KSTEP > 0 )) || { echo "Error: K_STEP must be > 0" >&2; exit 2; }

echo "Sweep config:"
echo "  N: $NMIN..$NMAX step $NSTEP"
echo "  K: $KMIN..$KMAX step $KSTEP"
echo "  NP=$NP  STRIDE=$STRIDE"
echo "  POST_COPY_WAIT=${POST_COPY_WAIT}s  FILE_WAIT_RETRIES=$FILE_WAIT_RETRIES  SEED=${SEED:-<default>}"

mkdir -p logs metrics

range_step() {
  local start="$1" end="$2" step="$3" x
  for ((x=start; x<=end; x+=step)); do
    echo "$x"
  done
}

submit_and_block() {
  local N="$1" K="$2"

  local submit_out jobid
  if [[ -n "$SEED" ]]; then
    submit_out=$(sbatch slurm_helpers/conv1d_mpi_param.slurm "$N" "$K" "$NP" same zero "$STRIDE" "$SEED")
  else
    submit_out=$(sbatch slurm_helpers/conv1d_mpi_param.slurm "$N" "$K" "$NP" same zero "$STRIDE")
  fi

  jobid=$(awk '{print $4}' <<<"$submit_out")
  [[ -n "${jobid:-}" ]] || { echo "Failed to parse job id from: $submit_out" >&2; exit 3; }
  echo "Submitted JOBID=$jobid  (N=$N, K=$K, np=$NP)"

  local err="logs/conv1d_mpi_${jobid}.err"
  local csv="metrics/metrics_SLURM_${jobid}.csv"

  # Wait while job is still in queue/running
  while squeue -j "$jobid" -h 2>/dev/null | grep -q . ; do
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

for N in $(range_step "$NMIN" "$NMAX" "$NSTEP"); do
  echo "=== N=$N ==="
  for K in $(range_step "$KMIN" "$KMAX" "$KSTEP"); do
    echo " -> K=$K"
    submit_and_block "$N" "$K"
  done
done

echo "All jobs completed and verified."
