# python_helpers/verify_conv2d_scipy.py
import argparse, numpy as np
from scipy.signal import convolve2d, correlate2d

def read_2d_txt(path):
    with open(path, "r") as f:
        h, w = map(int, f.readline().split())
        vals = np.fromstring(f.read(), sep=' ', dtype=np.float64)
        if vals.size != h*w:
            raise ValueError(f"{path}: expected {h*w} values, got {vals.size}")
        return vals.reshape(h, w)

def write_2d_txt(path, A):
    H, W = A.shape
    with open(path, "w") as f:
        f.write(f"{H} {W}\n")
        for r in range(H):
            f.write(" ".join(f"{x:.3f}" for x in A[r]))
            if r+1 < H: f.write("\n")
        f.write("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True)
    ap.add_argument("--g", required=True)
    ap.add_argument("--mode", choices=["same","full"], default="same")
    ap.add_argument("--pad", choices=["zero","const"], default="zero")
    ap.add_argument("--cval", type=float, default=0.0)
    ap.add_argument("--sH", type=int, default=1)
    ap.add_argument("--sW", type=int, default=1)
    ap.add_argument("--op", choices=["conv","corr"], default="conv")  # matches your --conv/--corr
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    F = read_2d_txt(args.f).astype(np.float64)
    G = read_2d_txt(args.g).astype(np.float64)

    boundary = "fill"
    fillv = args.cval if args.pad == "const" else 0.0

    if args.op == "conv":
        Y = convolve2d(F, G, mode=args.mode, boundary=boundary, fillvalue=fillv)
    else:  # corr
        Y = correlate2d(F, G, mode=args.mode, boundary=boundary, fillvalue=fillv)

    # stride sampling to match your program’s output grid
    Y = Y[::args.sH, ::args.sW]

    # match program’s 3dp text rounding
    Y = np.round(Y, 3)
    write_2d_txt(args.out, Y)

if __name__ == "__main__":
    main()