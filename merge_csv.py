#!/usr/bin/env python3
"""
merge_csv.py
------------
Utility script to merge multiple CSV files into a single CSV file.
Designed for combining performance metrics (e.g., SLURM job outputs).

Features:
- Reads multiple CSVs provided as CLI arguments (supports globbing like metrics/*.csv).
- Appends to metrics/merged.csv if it already exists (instead of overwriting).
- Concatenates all rows into a single DataFrame.
- Automatically sorts rows by N and K (ascending).
- Optionally deletes source CSVs after merging (if --delete flag is passed).
- Skips unreadable/invalid CSV files with a warning.

Usage:
    python3 merge_csv.py metrics/*.csv
    python3 merge_csv.py metrics/*.csv --delete
"""

import sys
import pandas as pd
import os

def merge_csv(files, out="metrics/merged.csv", delete=False):
    """
    Merge multiple CSV files into one.

    Args:
        files (list of str): List of CSV file paths to merge.
        out (str): Output file path for merged CSV.
        delete (bool): Whether to delete source files after merging.

    Returns:
        None. Writes merged CSV to disk.
    """
    merged = []

    # Load existing merged.csv if present (append mode)
    if os.path.exists(out):
        try:
            existing = pd.read_csv(out)
            merged.append(existing)
            print(f"Appending to existing {out}")
        except Exception as e:
            print(f"Warning: could not read existing {out}: {e}", file=sys.stderr)

    # Load new CSV files
    for f in files:
        try:
            df = pd.read_csv(f)
            merged.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}", file=sys.stderr)

    if not merged:
        print("No valid CSV files found.")
        return

    # Concatenate into one DataFrame
    combined = pd.concat(merged, ignore_index=True)

    # Sort rows by N and K if present
    if "N" in combined.columns and "K" in combined.columns:
        combined = combined.sort_values(by=["N", "K"], ascending=[True, True])

    # Ensure target directory exists
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Save merged output (always overwrites, but includes old+new rows)
    combined.to_csv(out, index=False)
    print(f"Merged {len(files)} new files into {out}")

    # Optionally delete source files
    if delete:
        for f in files:
            try:
                os.remove(f)
                print(f"Deleted {f}")
            except OSError as e:
                print(f"Error deleting {f}: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_csv.py <file1.csv> <file2.csv> ... [--delete]")
        sys.exit(1)

    files = [f for f in sys.argv[1:] if not f.startswith("--")]
    delete_flag = "--delete" in sys.argv

    merge_csv(files, out="metrics/merged.csv", delete=delete_flag)