# Assignment 2 Testing Framework

This directory contains SLURM scripts and batch helpers for automated testing of MPI and hybrid MPI+OpenMP convolution implementations on the Kaya HPC cluster.

## Directory Structure

```
A2/
├── slurm_helpers/          # Parameterized SLURM job scripts
│   ├── conv1d_mpi_omp_param.slurm
│   ├── conv2d_mpi_param.slurm
│   └── conv2d_mpi_omp_param.slurm
└── batch_helpers/          # Automated sweep scripts
    ├── sweep_conv1d_mpi_omp.sh           # Parallel sweep (concurrent jobs)
    ├── sweep_conv1d_mpi_omp_serial.sh    # Serial sweep (one-at-a-time with verification)
    ├── sweep_conv2d_mpi.sh
    ├── sweep_conv2d_mpi_serial.sh
    ├── sweep_conv2d_mpi_omp.sh
    └── sweep_conv2d_mpi_omp_serial.sh
```

## SLURM Helper Scripts

These parameterized scripts accept command-line arguments to avoid manual editing for each test configuration.

### conv1d_mpi_omp_param.slurm

**Purpose**: Hybrid MPI+OpenMP 1D convolution  
**Resources**: 4 MPI tasks × 4 OpenMP threads (default)

**Usage**:
```bash
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm N K [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]
```

**Parameters**:
- `N`, `K` (required): Input and kernel lengths
- `NP` (optional, default=4): Number of MPI processes
- `THREADS` (optional, default=4): OpenMP threads per process
- `SCHED` (optional, default=static): OpenMP schedule (static|dynamic|guided|auto)
- `CHUNK` (optional): OpenMP chunk size
- `MODE` (optional, default=same): Convolution mode (same|full)
- `PADDING` (optional, default=zero): Padding mode (zero|none|const)
- `SEED` (optional): RNG seed for reproducibility

**Examples**:
```bash
# Basic test with defaults (np=4, threads=4)
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 1000000 101

# Strong scaling: vary np, fixed problem size
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 10000000 1001 8 4

# Weak scaling: increase np and problem size together
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 8 4

# Custom OpenMP schedule with dynamic scheduling and chunk=16
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 5000000 501 4 8 dynamic 16 same zero 42
```

### conv2d_mpi_param.slurm

**Purpose**: Pure MPI 2D convolution with 2D Cartesian topology  
**Resources**: 4 MPI tasks (default)

**Usage**:
```bash
sbatch slurm_helpers/conv2d_mpi_param.slurm H W kH kW [NP] [MODE] [PADDING] [SEED]
```

**Parameters**:
- `H`, `W`, `kH`, `kW` (required): Input and kernel dimensions
- `NP` (optional, default=4): Number of MPI processes (will auto-decompose to P_rows × P_cols grid)
- `MODE` (optional, default=same): Convolution mode
- `PADDING` (optional, default=zero): Padding mode
- `SEED` (optional): RNG seed

**Examples**:
```bash
# 2D grid test: np=4 → 2×2 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5

# Strong scaling: np=9 → 3×3 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 9

# Large problem with np=16 → 4×4 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 16
```

### conv2d_mpi_omp_param.slurm

**Purpose**: Hybrid MPI+OpenMP 2D convolution with 2D Cartesian topology  
**Resources**: 4 MPI tasks × 4 OpenMP threads (default)

**Usage**:
```bash
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm H W kH kW [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]
```

**Parameters**: Same as conv1d_mpi_omp_param.slurm but with 2D dimensions

**Examples**:
```bash
# Hybrid scaling: 4 MPI processes × 4 threads = 16 total cores
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 4 4

# Compare different hybrid configurations (same total cores):
# - 4 MPI × 4 OMP = 16 cores
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 4 4
# - 8 MPI × 2 OMP = 16 cores
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 8 2
# - 16 MPI × 1 OMP = 16 cores (pure MPI)
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 16 1
```

## Batch Helper Scripts

### Parallel Sweep Scripts

Submit multiple jobs concurrently with throttling to avoid overloading the queue.

#### sweep_conv1d_mpi_omp.sh

**Usage**:
```bash
./batch_helpers/sweep_conv1d_mpi_omp.sh LMIN LMAX L_STEP KMIN KMAX K_STEP [MAX_IN_FLIGHT] [NP] [THREADS] [SCHED] [CHUNK] [SEED]
```

**Example**: Test L from 1M to 10M in steps of 1M, K from 101 to 1001 in steps of 100
```bash
./batch_helpers/sweep_conv1d_mpi_omp.sh 1000000 10000000 1000000 101 1001 100
```

**Example**: Strong scaling study with np=1,2,4,8
```bash
# Fixed problem size N=10M, K=1001, vary np
for np in 1 2 4 8; do
  ./batch_helpers/sweep_conv1d_mpi_omp.sh 10000000 10000000 1 1001 1001 1 20 $np 4
done
```

#### sweep_conv2d_mpi.sh

**Usage**:
```bash
./batch_helpers/sweep_conv2d_mpi.sh HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [MAX_IN_FLIGHT] [NP] [SEED]
```

**Example**: Test H,W from 1024 to 4096 in steps of 1024, kernel 3×3 to 11×11
```bash
./batch_helpers/sweep_conv2d_mpi.sh 1024 4096 1024 1024 4096 1024 3 11 2 3 11 2
```

**Example**: Strong scaling with different 2D grid sizes
```bash
# np=4 (2×2 grid)
./batch_helpers/sweep_conv2d_mpi.sh 2048 2048 1 2048 2048 1 5 5 1 5 5 1 20 4
# np=9 (3×3 grid)
./batch_helpers/sweep_conv2d_mpi.sh 2048 2048 1 2048 2048 1 5 5 1 5 5 1 20 9
# np=16 (4×4 grid)
./batch_helpers/sweep_conv2d_mpi.sh 2048 2048 1 2048 2048 1 5 5 1 5 5 1 20 16
```

### Serial Sweep Scripts

Submit one job at a time and **block** until both stderr log and metrics CSV are verified. This ensures:
- No missing data due to filesystem delays
- Sequential execution for controlled experiments
- Immediate error detection

#### sweep_conv1d_mpi_omp_serial.sh

**Usage**:
```bash
./batch_helpers/sweep_conv1d_mpi_omp_serial.sh LMIN LMAX L_STEP KMIN KMAX K_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [NP] [THREADS] [SCHED] [CHUNK] [SEED]
```

**Parameters**:
- `POST_COPY_WAIT` (default=10s): Wait time after job completes before checking for files
- `FILE_WAIT_RETRIES` (default=60): Number of 2s retries to wait for files (total 120s)

**Example**: Careful sweep with verification
```bash
./batch_helpers/sweep_conv1d_mpi_omp_serial.sh 1000000 5000000 1000000 101 501 100
```

**Example**: Strong scaling with verification (critical for performance analysis)
```bash
# Test np=1,2,4,8 on same problem size
for np in 1 2 4 8; do
  ./batch_helpers/sweep_conv1d_mpi_omp_serial.sh 10000000 10000000 1 1001 1001 1 10 60 $np 4 static "" 42
done
```

## Workflow Recommendations

### 1. Initial Correctness Testing

Use **serial sweeps** with small problem sizes to verify correctness:

```bash
# Verify 1D hybrid works across np=1,2,4
for np in 1 2 4; do
  ./batch_helpers/sweep_conv1d_mpi_omp_serial.sh 100000 100000 1 101 101 1 10 60 $np 2
done

# Verify 2D MPI with different grid configurations
for np in 1 4 9; do
  ./batch_helpers/sweep_conv2d_mpi_serial.sh 512 512 1 512 512 1 5 5 1 5 5 1 10 60 $np
done
```

### 2. Strong Scaling Studies

**Definition**: Fix problem size, increase parallelism

```bash
# 1D: Fixed N=20M, K=1001, vary np from 1 to 16
for np in 1 2 4 8 16; do
  sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 $np 4 static "" same zero 42
done

# 2D: Fixed 4096×4096 with 7×7 kernel, vary np
for np in 1 4 9 16 25; do
  sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 $np same zero 42
done
```

### 3. Weak Scaling Studies

**Definition**: Increase problem size proportionally with parallelism (constant work per processor)

```bash
# 1D: Each process handles 5M elements
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 5000000 1001 1 4  # 1 proc × 5M
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 10000000 1001 2 4 # 2 proc × 5M each
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 4 4 # 4 proc × 5M each
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 40000000 1001 8 4 # 8 proc × 5M each

# 2D: Each process handles ~1024×1024 block
sbatch slurm_helpers/conv2d_mpi_param.slurm 1024 1024 5 5 1  # 1×1 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 4  # 2×2 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 3072 3072 5 5 9  # 3×3 grid
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 5 5 16 # 4×4 grid
```

### 4. Hybrid Configuration Exploration

Compare different MPI/OpenMP balances with **same total cores**:

```bash
# Total 16 cores, different MPI×OMP configurations
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 16 1  # Pure MPI
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 8 2   # Balanced
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 4 4   # More OpenMP
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 2 8   # Heavily OpenMP
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 1 16  # Pure OpenMP
```

### 5. OpenMP Schedule Comparison

```bash
# Fixed problem and parallelism, vary schedule
for sched in static dynamic guided auto; do
  sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 10000000 1001 4 4 $sched "" same zero 42
done

# Dynamic with different chunk sizes
for chunk in 1 10 100 1000; do
  sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 10000000 1001 4 4 dynamic $chunk same zero 42
done
```

## Output and Metrics

### CSV Metrics Files

All runs automatically generate CSV files in `metrics/`:
- **On SLURM**: `metrics/metrics_SLURM_<JOBID>.csv`
- **Locally**: `metrics/metrics_LOCAL_<timestamp>_<PID>.csv`

### CSV Schema

**conv1d_mpi_omp**:
```
RunID,N,K,outLen,mode,padding,cval,time,gflops,np,omp_threads,schedule,chunk
```

**conv2d_mpi**:
```
RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,sH,sW,time,gflops,np,P_rows,P_cols
```

**conv2d_mpi_omp** (once updated):
```
RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,sH,sW,time,gflops,np,P_rows,P_cols,omp_threads,schedule,chunk
```

### SLURM Logs

- **stdout**: `logs/<program>_<JOBID>.out`
- **stderr**: `logs/<program>_<JOBID>.err` (includes runtime diagnostics and human-readable metrics)

## Post-Processing

### Collect All Metrics

```bash
# Combine all CSV files into one master file
cd metrics
head -1 $(ls metrics_SLURM_*.csv | head -1) > all_metrics.csv  # Header
tail -q -n +2 metrics_SLURM_*.csv >> all_metrics.csv             # Data rows
```

### Analyze with Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load all metrics
df = pd.read_csv('metrics/all_metrics.csv')

# Strong scaling analysis
problem = df[(df['H'] == 2048) & (df['W'] == 2048) & (df['KH'] == 5)]
speedup = problem.groupby('np')['time'].min().iloc[0] / problem.groupby('np')['time'].min()
efficiency = speedup / problem.groupby('np').first().index

plt.plot(efficiency.index, efficiency.values)
plt.xlabel('Number of Processes')
plt.ylabel('Parallel Efficiency')
plt.title('Strong Scaling Efficiency')
plt.show()

# GFLOPS comparison
df.groupby('np')['gflops'].mean().plot(kind='bar')
plt.ylabel('GFLOPS')
plt.title('Performance vs MPI Processes')
plt.show()
```

## Troubleshooting

### Jobs Fail Silently

**Symptom**: No CSV or stderr files appear after job completes  
**Solution**: Check SLURM output logs in `logs/` directory for compilation errors or invalid arguments

### CSV Files Missing Columns

**Symptom**: Python analysis fails with KeyError  
**Solution**: Ensure all programs have updated `log_metrics` functions with complete parameter lists

### Filesystem Delays

**Symptom**: Serial sweeps report "Missing: metrics CSV" even though job completed  
**Solution**: Increase `POST_COPY_WAIT` and `FILE_WAIT_RETRIES` parameters:
```bash
./batch_helpers/sweep_conv2d_mpi_serial.sh ... 20 90  # 20s wait, 90×2s retries
```

### Queue Overload

**Symptom**: Jobs stuck in pending state  
**Solution**: Reduce `MAX_IN_FLIGHT` parameter:
```bash
./batch_helpers/sweep_conv1d_mpi_omp.sh ... 10  # Limit to 10 concurrent jobs
```

## Best Practices

1. **Always use serial sweeps for critical data**: Ensures no missing data points
2. **Use consistent SEED**: Add `-s 42` for reproducible results
3. **Start small**: Test with small problem sizes first to verify correctness
4. **Monitor queue**: Use `squeue -u $USER` to check job status
5. **Check first job**: Manually verify first CSV and logs before launching large sweeps
6. **Backup metrics**: Copy `metrics/` directory periodically during long experiments
7. **Document runs**: Keep notes on what parameters were tested and why

## Quick Reference

### Check Queue Status
```bash
squeue -u $USER
```

### Cancel All Jobs
```bash
scancel -u $USER
```

### View Recent Metrics
```bash
ls -lt metrics/ | head -10
```

### Check Latest CSV
```bash
cat $(ls -t metrics/metrics_SLURM_*.csv | head -1)
```

### Monitor Job Progress
```bash
tail -f logs/conv2d_mpi_<JOBID>.err
```
