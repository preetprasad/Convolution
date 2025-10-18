#!/usr/bin/env python3
"""
Compare two matrix files with numerical tolerance.
Usage: compare_matrices.py file1.txt file2.txt [rtol] [atol]
Exit code: 0 if close, 1 if different, 2 if error
"""
import sys
import numpy as np

def load_matrix(path):
    """Load matrix from assignment format: 'H W' header + values"""
    with open(path) as f:
        h, w = map(int, f.readline().split())
        data = []
        for line in f:
            data.extend(map(float, line.split()))
        if len(data) != h * w:
            raise ValueError(f"Expected {h}×{w}={h*w} values, got {len(data)}")
        return np.array(data).reshape(h, w)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} file1.txt file2.txt [rtol] [atol]", file=sys.stderr)
        sys.exit(2)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3  # Default: 0.1% relative
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3  # Default: 0.001 absolute
    
    try:
        a = load_matrix(file1)
        b = load_matrix(file2)
        
        if a.shape != b.shape:
            print(f"Shape mismatch: {a.shape} vs {b.shape}", file=sys.stderr)
            sys.exit(1)
        
        if np.allclose(a, b, rtol=rtol, atol=atol):
            # Passed - print nothing for cleaner output
            sys.exit(0)
        else:
            diff = np.abs(a - b)
            max_diff = diff.max()
            max_idx = np.unravel_index(diff.argmax(), diff.shape)
            print(f"FAIL: max_diff={max_diff:.6f} at {max_idx}", file=sys.stderr)
            print(f"  {file1}[{max_idx}] = {a[max_idx]:.6f}", file=sys.stderr)
            print(f"  {file2}[{max_idx}] = {b[max_idx]:.6f}", file=sys.stderr)
            sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
