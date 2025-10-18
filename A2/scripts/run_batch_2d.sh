#!/usr/bin/env bash
# scripts/run_batch_2d.sh
# Comprehensive 2D convolution/correlation validation batch runner

set -euo pipefail

CC=${CC:-cc}
CFLAGS="-std=c11 -O2 -Wall -Wextra -Werror"
EXE=./conv2d

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR="$ROOT_DIR/tests_2d"
PY_HELP="$ROOT_DIR/python_helpers/verify_conv2d_scipy.py"
PY_CMP="$ROOT_DIR/python_helpers/compare_matrices.py"

# --- Staff test definitions ---
F=(f0.txt f1.txt f2.txt f3.txt f4.txt f5.txt)
G=(g0.txt g1.txt g2.txt g3.txt g4.txt g5.txt)
SH=(1 3 1 2 1 3)
SW=(1 2 1 3 1 7)
MODE=same
PAD=zero

# --- Tolerance settings ---
RTOL=1e-3  # 0.1% relative tolerance
ATOL=1e-3  # 0.001 absolute tolerance

# --- Output directories ---
OUT_MY_CONV="$TEST_DIR/out_my_conv"
OUT_MY_CORR="$TEST_DIR/out_my_corr"
OUT_SP_CONV="$TEST_DIR/out_scipy_conv"
OUT_SP_CORR="$TEST_DIR/out_scipy_corr"
mkdir -p "$OUT_MY_CONV" "$OUT_MY_CORR" "$OUT_SP_CONV" "$OUT_SP_CORR"

echo "[build] $CC $CFLAGS -o $EXE $ROOT_DIR/conv2d.c"
$CC $CFLAGS -o "$EXE" "$ROOT_DIR/conv2d.c"

# --- Counters ---
my_conv_pass=0;  my_conv_total=0
my_corr_pass=0;  my_corr_total=0
sp_conv_pass=0;  sp_conv_total=0
sp_corr_pass=0;  sp_corr_total=0
my_vs_sp_pass=0; my_vs_sp_total=0
mycorr_vs_spcorr_pass=0; mycorr_vs_spcorr_total=0

# Helper function: compare two files with numerical tolerance
compare_with_tolerance() {
  local file1=$1
  local file2=$2
  
  if [[ ! -f "$file1" || ! -f "$file2" ]]; then
    return 1
  fi
  
  python3 "$PY_CMP" "$file1" "$file2" "$RTOL" "$ATOL" 2>/dev/null
  return $?
}

report() {
  local ok=$1 label=$2
  if [[ $ok -eq 1 ]]; then
    printf "OK   : %s\n" "$label"
  else
    printf "FAIL : %s\n" "$label"
  fi
}

echo
echo "=== Running Batch Validation (mode=$MODE, pad=$PAD, tol=±${ATOL}) ==="
echo

for i in $(seq 0 5); do
  f="$TEST_DIR/${F[$i]}"
  g="$TEST_DIR/${G[$i]}"
  sH="${SH[$i]}"
  sW="${SW[$i]}"
  staff="$TEST_DIR/o${i}_sH_${sH}_sW_${sW}.txt"

  [[ -f "$f" && -f "$g" && -f "$staff" ]] || { echo "Skip case $i (missing files)"; continue; }

  echo "--- Case $i: f=$(basename "$f"), g=$(basename "$g"), sH=$sH, sW=$sW ---"

  # 1) My CONV
  my_conv_out="$OUT_MY_CONV/o${i}_mine.txt"
  "$EXE" --conv -f "$f" -g "$g" -m "$MODE" -sH "$sH" -sW "$sW" -o "$my_conv_out" >/dev/null 2>&1
  ((my_conv_total++))
  if compare_with_tolerance "$my_conv_out" "$staff"; then ok=1; else ok=0; fi
  report $ok "my CONV vs staff (case $i)"
  ((my_conv_pass+=ok))

  # 2) My CORR
  my_corr_out="$OUT_MY_CORR/o${i}_mine_corr.txt"
  "$EXE" --corr -f "$f" -g "$g" -m "$MODE" -sH "$sH" -sW "$sW" -o "$my_corr_out" >/dev/null 2>&1
  ((my_corr_total++))
  if compare_with_tolerance "$my_corr_out" "$staff"; then ok=1; else ok=0; fi
  report $ok "my CORR vs staff (case $i)"
  ((my_corr_pass+=ok))

  # 3) SciPy CONV
  sp_conv_out="$OUT_SP_CONV/o${i}_scipy.txt"
  python3 "$PY_HELP" --op conv --f "$f" --g "$g" \
    --mode "$MODE" --pad "$PAD" --sH "$sH" --sW "$sW" --out "$sp_conv_out" >/dev/null 2>&1
  ((sp_conv_total++))
  if compare_with_tolerance "$sp_conv_out" "$staff"; then ok=1; else ok=0; fi
  report $ok "SciPy CONV vs staff (case $i)"
  ((sp_conv_pass+=ok))

  # 4) SciPy CORR
  sp_corr_out="$OUT_SP_CORR/o${i}_scipy_corr.txt"
  python3 "$PY_HELP" --op corr --f "$f" --g "$g" \
    --mode "$MODE" --pad "$PAD" --sH "$sH" --sW "$sW" --out "$sp_corr_out" >/dev/null 2>&1
  ((sp_corr_total++))
  if compare_with_tolerance "$sp_corr_out" "$staff"; then ok=1; else ok=0; fi
  report $ok "SciPy CORR vs staff (case $i)"
  ((sp_corr_pass+=ok))

  # 5) My CONV vs SciPy CONV
  ((my_vs_sp_total++))
  if compare_with_tolerance "$my_conv_out" "$sp_conv_out"; then ok=1; else ok=0; fi
  report $ok "my CONV vs SciPy CONV (case $i)"
  ((my_vs_sp_pass+=ok))

  # 6) My CORR vs SciPy CORR
  ((mycorr_vs_spcorr_total++))
  if compare_with_tolerance "$my_corr_out" "$sp_corr_out"; then ok=1; else ok=0; fi
  report $ok "my CORR vs SciPy CORR (case $i)"
  ((mycorr_vs_spcorr_pass+=ok))
done

echo
echo "=== SUMMARY (tolerance: rtol=$RTOL, atol=$ATOL) ==="
printf "my CONV  vs staff         : %d/%d passed\n" "$my_conv_pass" "$my_conv_total"
printf "my CORR  vs staff         : %d/%d passed\n" "$my_corr_pass" "$my_corr_total"
printf "SciPy CONV vs staff       : %d/%d passed\n" "$sp_conv_pass" "$sp_conv_total"
printf "SciPy CORR vs staff       : %d/%d passed\n" "$sp_corr_pass" "$sp_corr_total"
printf "my CONV  vs SciPy CONV    : %d/%d passed\n" "$my_vs_sp_pass" "$my_vs_sp_total"
printf "my CORR  vs SciPy CORR    : %d/%d passed\n" "$mycorr_vs_spcorr_pass" "$mycorr_vs_spcorr_total"
echo
echo "Outputs written to:"
echo "  $OUT_MY_CONV"
echo "  $OUT_MY_CORR"
echo "  $OUT_SP_CONV"
echo "  $OUT_SP_CORR"