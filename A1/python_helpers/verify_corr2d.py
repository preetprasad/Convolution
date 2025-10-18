#!/usr/bin/env python3
# Minimal SciPy-based 2-D correlation checker.
# Usage:
#   python verify_corr2d_min.py -F tests_2d/f0.txt -G tests_2d/g0.txt -O ref_corr_o0.txt
# This writes ref_corr_o0.txt in the same "H W + values" format (3 decimals) for manual diff.

import argparse
from pathlib import Path
import numpy as np
from scipy.signal import correlate

def read_hw_matrix(path: str) -> np.ndarray:
    """
    Reads matrix in assignment format:
      first line (or first two tokens): H W
      followed by H*W floats (any whitespace, any line breaks).
    Returns float64 ndarray of shape (H, W).
    """
    tokens = Path(path).read_text().split()
    if len(tokens) < 2:
        raise ValueError(f"{path}: expected 'H W' header.")
    H = int(tokens[0]); W = int(tokens[1])
    vals = list(map(float, tokens[2:]))
    if len(vals) != H * W:
        raise ValueError(f"{path}: expected {H*W} values, found {len(vals)}.")
    return np.array(vals, dtype=np.float64).reshape(H, W)

def write_hw_matrix(path: str, M: np.ndarray):
    """
    Writes matrix in assignment format:
      line 1: 'H W'
      next lines: H*W floats (3 decimals), row-major; one row per line.
    """
    H, W = M.shape
    with open(path, "w") as f:
        f.write(f"{H} {W}\n")
        for r in range(H):
            row = " ".join(f"{float(M[r, c]):.3f}" for c in range(W))
            f.write(row + ("\n" if r + 1 < H else ""))

def main():
    ap = argparse.ArgumentParser(description="Minimal 2D correlation checker (SciPy correlate).")
    ap.add_argument("-F", required=True, help="Path to F (input image) in H W + values format")
    ap.add_argument("-G", required=True, help="Path to G (kernel) in H W + values format")
    ap.add_argument("-O", required=True, help="Path to write output (same format)")
    args = ap.parse_args()

    F = read_hw_matrix(args.F)
    G = read_hw_matrix(args.G)

    # Match your C runs (SAME size output). Change to 'full' if needed.
    MODE = "same"
    Y = correlate(F, G, mode=MODE)

    write_hw_matrix(args.O, Y)

    # Also print a quick summary so you can eyeball without opening the file.
    print(f"mode={MODE} -> wrote {args.O} with shape {Y.shape} (H W)")

if __name__ == "__main__":
    main()