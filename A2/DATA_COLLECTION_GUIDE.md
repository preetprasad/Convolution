# A2 Data Collection Guide

**Date**: October 19, 2025  
**Purpose**: Step-by-step guide for collecting performance data on Kaya HPC cluster

---

## Table of Contents

1. [Pre-Collection Setup](#pre-collection-setup)
2. [Phase 1: Correctness Verification](#phase-1-correctness-verification)
3. [Phase 2: Sequential Baselines](#phase-2-sequential-baselines)
4. [Phase 3: OpenMP Scaling](#phase-3-openmp-scaling)
5. [Phase 4: MPI Scaling](#phase-4-mpi-scaling)
6. [Phase 5: Hybrid MPI+OpenMP](#phase-5-hybrid-mpiopenmp)
7. [Data Organization](#data-organization)
8. [Troubleshooting](#troubleshooting)

---

## Pre-Collection Setup

### 1. Connect to Kaya

```bash
ssh <your-username>@kaya.hpc.uwa.edu.au
```

### 2. Navigate to Project Directory

```bash
cd /path/to/Convolution-1/A2
```

### 3. Load Required Modules

```bash
module load gcc/14.2
module load openmpi/5.0.5
```

### 4. Build All Programs

```bash
# Build everything
make

# Or build specific targets
make seq      # Sequential programs
make omp      # OpenMP programs
make mpi      # MPI programs
make hybrid   # Hybrid MPI+OpenMP programs
```

### 5. Verify Compilation

```bash
# Check all binaries exist
ls -lh conv1d conv1d_omp conv1d_mpi conv1d_mpi_omp
ls -lh conv2d conv2d_omp conv2d_mpi conv2d_mpi_omp
```

### 6. Create Directory Structure

```bash
mkdir -p logs metrics results
```

---

## Phase 1: Correctness Verification

**Goal**: Ensure all implementations produce correct results before performance testing.

### Step 1.1: Test Sequential Baselines (Small Problems)

```bash
# 1D Sequential
sbatch slurm_helpers/conv1d_seq_param.slurm 10000 101 same zero 42

# 2D Sequential  
sbatch slurm_helpers/conv2d_seq_param.slurm 128 128 5 5 same zero 42

# Wait for completion
squeue -u $USER

# Check outputs
cat logs/conv1d_*.err
cat logs/conv2d_*.err
```

**Expected**: Clean execution with GFLOPS metric in stderr.

### Step 1.2: Test OpenMP (Small Problems)

```bash
# 1D OpenMP (4 threads)
sbatch slurm_helpers/conv1d_omp_param.slurm 10000 101 4 static "" same zero 42

# 2D OpenMP (4 threads)
sbatch slurm_helpers/conv2d_omp_param.slurm 128 128 5 5 4 static "" same zero 42
```

### Step 1.3: Test MPI (Small Problems)

```bash
# 1D MPI (2 processes)
sbatch slurm_helpers/conv1d_mpi_param.slurm 10000 101 2 same zero 42

# 2D MPI (4 processes → 2×2 grid)
sbatch slurm_helpers/conv2d_mpi_param.slurm 128 128 5 5 4 same zero 42
```

**Expected**: Stderr shows MPI decomposition (e.g., "ranks=4 (2D: 2x2)").

### Step 1.4: Test Hybrid (Small Problems)

```bash
# 1D Hybrid (2 MPI × 2 OMP = 4 cores)
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 10000 101 2 2 static "" same zero 42

# 2D Hybrid (4 MPI × 2 OMP = 8 cores, 2×2 MPI grid)
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 128 128 5 5 4 2 static "" same zero 42
```

### Step 1.5: Compare Outputs (Optional)

If you saved outputs (not /dev/null), compare for correctness:

```bash
# Compare sequential vs parallel outputs
python3 python_helpers/compare_matrices.py results/seq_output.txt results/mpi_output.txt
```

**✅ Checkpoint**: All implementations run successfully with small problems.

---

## Phase 2: Sequential Baselines

**Goal**: Establish baseline performance for comparison.

### Strategy

Run **medium-sized problems** that take ~30 seconds to several minutes. This gives reliable timing without excessive wall time.

### 2.1: 1D Sequential Baseline

```bash
# Small sweep for baseline data
./batch_helpers/sweep_conv1d_seq.sh \
  1000000 10000000 1000000 \
  101 1001 100 \
  20 42

# Parameter explanation:
# N: 1M to 10M (step 1M) - 10 points
# K: 101 to 1001 (step 100) - 10 points
# Max 20 concurrent jobs
# Seed: 42
```

**Expected Jobs**: 10 × 10 = 100 jobs

### 2.2: 2D Sequential Baseline

```bash
# Smaller sweep for 2D (more expensive)
./batch_helpers/sweep_conv2d_seq.sh \
  1024 4096 512 \
  1024 4096 512 \
  3 7 2 \
  3 7 2 \
  20 42

# Parameter explanation:
# H: 1024 to 4096 (step 512) - 7 points
# W: 1024 to 4096 (step 512) - 7 points
# kH: 3, 5, 7 - 3 points
# kW: 3, 5, 7 - 3 points
```

**Expected Jobs**: 7 × 7 × 3 × 3 = 441 jobs

### Monitor Progress

```bash
# Check queue
squeue -u $USER

# Count completed jobs
ls logs/conv1d_*.err | wc -l
ls logs/conv2d_*.err | wc -l

# Check metrics files
ls metrics/*.csv | wc -l
```

**✅ Checkpoint**: Sequential baseline data collected in `metrics/` directory.

---

## Phase 3: OpenMP Scaling

**Goal**: Measure OpenMP thread scaling and schedule performance.

### 3.1: OpenMP Thread Scaling (1D)

Test different thread counts with **fixed problem size** (strong scaling):

```bash
# Test 1, 2, 4, 8, 16 threads
for threads in 1 2 4 8 16; do
  sbatch slurm_helpers/conv1d_omp_param.slurm \
    5000000 501 $threads static "" same zero 42
done

# Test different schedules (8 threads)
for sched in static dynamic guided auto; do
  sbatch slurm_helpers/conv1d_omp_param.slurm \
    5000000 501 8 $sched "" same zero 42
done

# Test dynamic with different chunk sizes
for chunk in 10 100 1000 10000; do
  sbatch slurm_helpers/conv1d_omp_param.slurm \
    5000000 501 8 dynamic $chunk same zero 42
done
```

### 3.2: OpenMP Scaling Sweep (2D)

```bash
# Systematic sweep across problem sizes and thread counts
# Smaller sweep for time efficiency
./batch_helpers/sweep_conv2d_omp.sh \
  1024 2048 512 \
  1024 2048 512 \
  5 5 2 \
  5 5 2 \
  20 8 static "" 42

# Test different thread counts (manual loop)
for threads in 1 2 4 8 16; do
  ./batch_helpers/sweep_conv2d_omp.sh \
    2048 2048 1024 \
    2048 2048 1024 \
    5 5 2 \
    5 5 2 \
    20 $threads static "" 42
done
```

**Expected**: Data showing speedup vs thread count.

**✅ Checkpoint**: OpenMP scaling data collected.

---

## Phase 4: MPI Scaling

**Goal**: Measure MPI process scaling (strong and weak scaling).

### 4.1: MPI Strong Scaling (1D)

**Fixed problem size, vary process count**:

```bash
# Fixed problem: N=20M, K=1001
# Vary np: 1, 2, 4, 8, 16
for np in 1 2 4 8 16; do
  sbatch slurm_helpers/conv1d_mpi_param.slurm \
    20000000 1001 $np same zero 42
done
```

### 4.2: MPI Weak Scaling (1D)

**Scale problem with processes** (constant work per process):

```bash
# Base: 5M per process
sbatch slurm_helpers/conv1d_mpi_param.slurm 5000000  1001 1  same zero 42  # 1 proc
sbatch slurm_helpers/conv1d_mpi_param.slurm 10000000 1001 2  same zero 42  # 2 proc
sbatch slurm_helpers/conv1d_mpi_param.slurm 20000000 1001 4  same zero 42  # 4 proc
sbatch slurm_helpers/conv1d_mpi_param.slurm 40000000 1001 8  same zero 42  # 8 proc
sbatch slurm_helpers/conv1d_mpi_param.slurm 80000000 1001 16 same zero 42  # 16 proc
```

### 4.3: MPI 2D Grid Scaling (2D)

**Test different grid configurations**:

```bash
# Fixed problem: 4096×4096, kernel 7×7
# Vary np to create different 2D grids
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 1  same zero 42  # 1×1 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 4  same zero 42  # 2×2 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 9  same zero 42  # 3×3 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 16 same zero 42  # 4×4 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 25 same zero 42  # 5×5 grid
```

### 4.4: MPI Scaling Sweep (2D)

```bash
# Sweep with moderate concurrency
./batch_helpers/sweep_conv2d_mpi.sh \
  2048 4096 1024 \
  2048 4096 1024 \
  5 7 2 \
  5 7 2 \
  20 4 42

# Test different process counts (manual)
for np in 1 4 9 16; do
  sbatch slurm_helpers/conv2d_mpi_param.slurm \
    4096 4096 7 7 $np same zero 42
done
```

**✅ Checkpoint**: MPI scaling data collected.

---

## Phase 5: Hybrid MPI+OpenMP

**Goal**: Explore optimal balance between MPI processes and OpenMP threads.

### 5.1: Hybrid Configuration Space (1D)

**Fixed total cores (16), vary MPI×OMP balance**:

```bash
# 16 total cores, different splits
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 16 1  static "" same zero 42  # Pure MPI
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 8  2  static "" same zero 42  # 8×2
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 4  4  static "" same zero 42  # 4×4
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 2  8  static "" same zero 42  # 2×8
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 1  16 static "" same zero 42  # Pure OMP
```

### 5.2: Hybrid Scaling Sweep (1D)

```bash
# Systematic sweep
./batch_helpers/sweep_conv1d_mpi_omp.sh \
  5000000 20000000 5000000 \
  501 1001 500 \
  20 4 4 static "" 42

# Parameter explanation:
# N: 5M to 20M (step 5M) - 4 points
# K: 501, 1001 - 2 points
# np=4, threads=4 (16 total cores)
```

### 5.3: Hybrid 2D Grid Configurations

**Fixed total cores (16), vary MPI grid and OMP threads**:

```bash
# 16 total cores, test different 2D grid × OMP combinations
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 16 1 static "" same zero 42  # 4×4 grid, 1 OMP
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 8  2 static "" same zero 42  # 2√2×2√2 grid, 2 OMP
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 4  4 static "" same zero 42  # 2×2 grid, 4 OMP
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 2  8 static "" same zero 42  # 1×2 grid, 8 OMP
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 1  16 static "" same zero 42 # Pure OMP
```

### 5.4: Hybrid Scaling Sweep (2D)

```bash
# Systematic sweep with hybrid
./batch_helpers/sweep_conv2d_mpi_omp.sh \
  2048 4096 1024 \
  2048 4096 1024 \
  5 7 2 \
  5 7 2 \
  20 4 4 static "" 42
```

**✅ Checkpoint**: Hybrid configuration data collected.

---

## Data Organization

### Directory Structure After Collection

```
A2/
├── logs/                       # SLURM stdout/stderr
│   ├── conv1d_<jobid>.out
│   ├── conv1d_<jobid>.err
│   ├── conv2d_<jobid>.out
│   └── conv2d_<jobid>.err
│
├── metrics/                    # CSV metrics files
│   ├── metrics_SLURM_<jobid>.csv
│   └── ...
│
└── results/                    # Output arrays (if saved)
    └── ...
```

### Metrics CSV Format

Each CSV contains performance data:

```csv
RunID,implementation,N,K,H,W,kH,kW,np,P_rows,P_cols,threads,schedule,chunk,conv_time,GFLOPS
```

**Key Columns**:
- `RunID`: Unique identifier (SLURM job ID)
- `implementation`: seq, omp, mpi, hybrid
- `N,K` or `H,W,kH,kW`: Problem dimensions
- `np,P_rows,P_cols`: MPI configuration
- `threads,schedule,chunk`: OpenMP configuration
- `conv_time`: Convolution time (seconds)
- `GFLOPS`: Performance metric

### Collecting All Metrics

```bash
# Merge all CSV files
cat metrics/*.csv > all_metrics.csv

# Remove duplicate headers (keep only first)
awk 'NR==1 || !/^RunID/' all_metrics.csv > merged_metrics.csv

# Count data points
wc -l merged_metrics.csv
```

---

## Monitoring and Quality Control

### Check Job Status

```bash
# Active jobs
squeue -u $USER

# Completed jobs (last 24 hours)
sacct -u $USER -S $(date -d '24 hours ago' +%Y-%m-%d)

# Failed jobs
sacct -u $USER -S $(date -d '24 hours ago' +%Y-%m-%d) --state=FAILED
```

### Verify Data Quality

```bash
# Count metrics files per implementation
grep -l "conv1d," metrics/*.csv | wc -l    # Sequential 1D
grep -l "conv1d_omp" metrics/*.csv | wc -l # OpenMP 1D
grep -l "conv1d_mpi" metrics/*.csv | wc -l # MPI 1D
grep -l "conv2d," metrics/*.csv | wc -l    # Sequential 2D
grep -l "conv2d_omp" metrics/*.csv | wc -l # OpenMP 2D
grep -l "conv2d_mpi" metrics/*.csv | wc -l # MPI 2D

# Check for errors in logs
grep -i error logs/*.err
grep -i "failed\|abort" logs/*.err

# Check GFLOPS range (should be positive, reasonable values)
grep GFLOPS logs/*.err | sort -t'=' -k2 -n
```

### Data Validation

```bash
# Check CSV integrity
for csv in metrics/*.csv; do
  # Count columns (should be consistent)
  awk -F',' 'NR==1 {print NF; exit}' "$csv"
done

# Look for missing or NaN values
grep -E ",,|NaN|inf" metrics/*.csv

# Check time ranges (should be positive)
awk -F',' 'NR>1 && $NF<0 {print FILENAME":"$0}' metrics/*.csv
```

---

## Suggested Data Collection Schedule

### Day 1: Setup + Correctness

- [ ] Connect to Kaya
- [ ] Build all programs
- [ ] Run Phase 1 (Correctness Verification)
- [ ] **~30 minutes**

### Day 2: Sequential Baselines

- [ ] Run Phase 2.1 (1D Sequential)
- [ ] Run Phase 2.2 (2D Sequential)
- [ ] **~2-4 hours** (depending on problem sizes)

### Day 3: OpenMP Scaling

- [ ] Run Phase 3.1 (1D OpenMP)
- [ ] Run Phase 3.2 (2D OpenMP)
- [ ] **~3-6 hours**

### Day 4: MPI Scaling

- [ ] Run Phase 4.1-4.2 (1D MPI)
- [ ] Run Phase 4.3-4.4 (2D MPI)
- [ ] **~4-8 hours**

### Day 5: Hybrid Exploration

- [ ] Run Phase 5.1-5.2 (1D Hybrid)
- [ ] Run Phase 5.3-5.4 (2D Hybrid)
- [ ] **~4-8 hours**

### Day 6: Verification + Re-runs

- [ ] Check data quality
- [ ] Re-run any failed jobs
- [ ] Fill gaps in parameter space
- [ ] **~2-4 hours**

**Total Estimated Time**: 15-30 hours (spread over 1 week)

---

## Quick Start Commands

### Minimal Data Collection (Fast)

For quick results to start analysis:

```bash
# Sequential baselines
sbatch slurm_helpers/conv1d_seq_param.slurm 5000000 501 same zero 42
sbatch slurm_helpers/conv2d_seq_param.slurm 2048 2048 5 5 same zero 42

# OpenMP scaling (threads: 1,2,4,8)
for t in 1 2 4 8; do
  sbatch slurm_helpers/conv1d_omp_param.slurm 5000000 501 $t static "" same zero 42
  sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 $t static "" same zero 42
done

# MPI scaling (np: 1,4,9)
for np in 1 4 9; do
  sbatch slurm_helpers/conv1d_mpi_param.slurm 5000000 501 $np same zero 42
  sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 $np same zero 42
done

# Hybrid (4×4 config)
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 5000000 501 4 4 static "" same zero 42
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 4 4 static "" same zero 42
```

**Expected**: ~30 jobs, ~1-2 hours total

---

## Troubleshooting

### Issue: Jobs Pending in Queue

**Symptoms**: `squeue` shows jobs in PD (pending) state

**Solutions**:
- Wait for available resources
- Reduce `--ntasks` or `--cpus-per-task` in SLURM scripts
- Check partition limits: `sinfo -p cits3402`

### Issue: Out of Memory (OOM) Errors

**Symptoms**: Jobs fail with "Out of Memory" in logs

**Solutions**:
- Increase `--mem` in SLURM scripts (e.g., `--mem=16G` → `--mem=32G`)
- Reduce problem size
- Use more MPI processes (distributes memory)

### Issue: MPI Jobs Timeout

**Symptoms**: Jobs killed after 15 minutes

**Solutions**:
- Increase `--time` in SLURM scripts (e.g., `--time=00:30:00`)
- Reduce problem size
- Use more processes for strong scaling

### Issue: Missing Metrics Files

**Symptoms**: `metrics/` directory empty or missing CSVs

**Solutions**:
- Check stderr logs for program crashes
- Verify programs compiled correctly (`make clean && make`)
- Check disk quota: `quota -s`

### Issue: Inconsistent Results

**Symptoms**: Same configuration gives different times

**Solutions**:
- Use fixed seed (`-s 42`) for reproducibility
- Check for system load: `squeue -u $USER | wc -l`
- Run multiple replicates and average
- Avoid running during peak hours

---

## Next Steps After Collection

1. **Merge Metrics**: Combine all CSV files
2. **Data Analysis**: Use Python/R to analyze performance
3. **Visualization**: Plot speedup curves, efficiency graphs
4. **Report Writing**: Document findings and insights

**See Also**:
- `TESTING_FRAMEWORK.md` - Detailed script reference
- `SLURM_SCRIPTS_REFERENCE.md` - SLURM parameter guide
- `MAKEFILE_GUIDE.md` - Building programs

---

**Ready to start?** Begin with Phase 1 (Correctness Verification)! 🚀
