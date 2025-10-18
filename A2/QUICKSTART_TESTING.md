# A2 Testing Quick Start Guide

## 🚀 Ready-to-Use Testing Framework for Assignment 2

This guide gets you running tests on Kaya in **under 5 minutes**.

## Prerequisites

1. SSH into Kaya
2. Navigate to A2 directory: `cd ~/Convolution-1/A2`
3. Ensure your programs compile: `make` or manual `mpicc` commands

## Quick Test Examples

### Example 1: Single Job Test (Verify Compilation)

Test that everything works before launching sweeps:

```bash
# Test conv2d_mpi with np=4 (creates 2×2 grid)
sbatch slurm_helpers/conv2d_mpi_param.slurm 1024 1024 5 5 4

# Check queue
squeue -u $USER

# After job completes, check output
cat logs/conv2d_mpi_*.err | tail -20
cat metrics/metrics_SLURM_*.csv
```

Expected CSV output:
```
RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,sH,sW,time,gflops,np,P_rows,P_cols
SLURM_12345,1024,1024,5,5,1024,1024,conv,same,zero,0,1,1,0.123456,123.456,4,2,2
```

### Example 2: Strong Scaling Study (Fixed Problem Size)

Test scalability by varying np on same problem:

```bash
# Test np = 1, 4, 9, 16 on 2048×2048 problem with 5×5 kernel
for np in 1 4 9 16; do
  sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 $np same zero 42
done

# Monitor jobs
watch -n 2 'squeue -u $USER'

# After completion, analyze
cd metrics
head -1 $(ls metrics_SLURM_*.csv | head -1) > strong_scaling.csv
tail -q -n +2 metrics_SLURM_*.csv >> strong_scaling.csv
```

### Example 3: Hybrid MPI+OpenMP Comparison

Compare pure MPI vs hybrid approaches (same total cores):

```bash
# 16 total cores, different configurations:

# Pure MPI: 16 processes × 1 thread
sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 16

# Balanced: 4 processes × 4 threads
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 4 4

# Pure OpenMP: 1 process × 16 threads (use A1 OMP script)
sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 16
```

### Example 4: Automated Parameter Sweep

Let the scripts do the heavy lifting:

```bash
# Parallel sweep: Tests H,W from 1024 to 4096 (step 1024), kernel 3×3 to 7×7 (step 2)
# With np=4, max 15 concurrent jobs
./batch_helpers/sweep_conv2d_mpi.sh 1024 4096 1024 1024 4096 1024 3 7 2 3 7 2 15 4 42

# Serial sweep (with verification): Safer for critical data
./batch_helpers/sweep_conv2d_mpi_serial.sh 2048 2048 1 2048 2048 1 5 5 1 5 5 1 10 60 4 42
```

### Example 5: 1D Hybrid Testing

```bash
# Test hybrid 1D with np=4, threads=4
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 10000000 1001 4 4 static "" same zero 42

# Sweep different problem sizes with verification
./batch_helpers/sweep_conv1d_mpi_omp_serial.sh 1000000 10000000 1000000 101 1001 100 10 60 4 4 static "" 42
```

## Common Workflows

### Workflow 1: Initial Correctness Check

Before performance testing, verify correctness with small problems:

```bash
# Make sure programs work with different np values
for np in 1 2 4; do
  sbatch slurm_helpers/conv2d_mpi_param.slurm 512 512 3 3 $np same zero 42
done

# Wait for jobs to complete
squeue -u $USER

# Check all jobs succeeded (no errors in .err files)
grep -i "error\|segmentation\|abort" logs/conv2d_mpi_*.err || echo "All jobs OK!"
```

### Workflow 2: Performance Profiling (5min, 10min, 15min targets)

Find problem sizes that hit target execution times:

```bash
# Start with baseline: 1024×1024
sbatch slurm_helpers/conv2d_mpi_param.slurm 1024 1024 5 5 4

# Check time in CSV
cat $(ls -t metrics/metrics_SLURM_*.csv | head -1)
# If time ≈ 1s, scale up: time scales as O(H²) for fixed kernel

# For 5min target (300s): need 300x larger, so H² * 300 ≈ 1024² * 300
# H ≈ 1024 * √300 ≈ 17,733
sbatch slurm_helpers/conv2d_mpi_param.slurm 18000 18000 5 5 4

# Adjust based on actual timing
```

### Workflow 3: Weak Scaling Study

Increase problem size proportionally with np:

```bash
# Each process handles constant work (e.g., 1024×1024 block per process)

# np=1 (1×1 grid): 1024×1024 total
sbatch slurm_helpers/conv2d_mpi_param.slurm 1024 1024 5 5 1 same zero 42

# np=4 (2×2 grid): 2048×2048 total (each process still ~1024×1024)
sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 4 same zero 42

# np=9 (3×3 grid): 3072×3072 total
sbatch slurm_helpers/conv2d_mpi_param.slurm 3072 3072 5 5 9 same zero 42

# np=16 (4×4 grid): 4096×4096 total
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 5 5 16 same zero 42

# Ideal weak scaling: time should stay constant
```

## Monitoring and Debugging

### Check Job Status

```bash
# View queue
squeue -u $USER

# View queue with time remaining
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.10l %.6D %R"

# Count running jobs
squeue -u $USER | grep -c RUNNING
```

### View Live Output

```bash
# Find latest job ID
JOBID=$(squeue -u $USER -h -o "%i" | head -1)

# Tail stderr (shows progress)
tail -f logs/conv2d_mpi_${JOBID}.err

# Or use watch
watch -n 1 "tail -20 logs/conv2d_mpi_${JOBID}.err"
```

### Check Latest Results

```bash
# View most recent CSV
cat $(ls -t metrics/metrics_SLURM_*.csv | head -1)

# View latest 5 CSVs with time and GFLOPS
for f in $(ls -t metrics/metrics_SLURM_*.csv | head -5); do
  echo "=== $f ==="
  tail -1 $f | awk -F',' '{print "Time: "$14"s, GFLOPS: "$15", np: "$16}'
done
```

### Cancel Jobs

```bash
# Cancel all your jobs
scancel -u $USER

# Cancel specific job
scancel <JOBID>

# Cancel all conv2d_mpi jobs
scancel -n conv2d_mpi_param
```

## Collecting Results

### Merge All CSV Files

```bash
cd metrics

# Create master CSV with header
head -1 $(ls metrics_SLURM_*.csv | head -1) > all_conv2d_mpi_results.csv

# Append all data rows (skip headers)
tail -q -n +2 metrics_SLURM_*.csv >> all_conv2d_mpi_results.csv

# Count data points
echo "Total runs: $(tail -n +2 all_conv2d_mpi_results.csv | wc -l)"
```

### Download to Local Machine

```bash
# From your local machine:
scp kaya:~/Convolution-1/A2/metrics/all_conv2d_mpi_results.csv .

# Or entire metrics directory:
scp -r kaya:~/Convolution-1/A2/metrics/ ./
```

## Python Analysis (Run Locally After Download)

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('all_conv2d_mpi_results.csv')

# Strong scaling analysis
problem = df[(df['H'] == 2048) & (df['W'] == 2048) & (df['KH'] == 5) & (df['KW'] == 5)]
grouped = problem.groupby('np').agg({'time': 'min', 'gflops': 'max'})

# Calculate speedup and efficiency
t1 = grouped.loc[1, 'time']
grouped['speedup'] = t1 / grouped['time']
grouped['efficiency'] = grouped['speedup'] / grouped.index

print("Strong Scaling Results:")
print(grouped)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Speedup
axes[0].plot(grouped.index, grouped['speedup'], 'o-', label='Actual')
axes[0].plot(grouped.index, grouped.index, '--', label='Ideal')
axes[0].set_xlabel('Number of Processes (np)')
axes[0].set_ylabel('Speedup')
axes[0].set_title('Strong Scaling Speedup')
axes[0].legend()
axes[0].grid(True)

# Efficiency
axes[1].plot(grouped.index, grouped['efficiency'] * 100, 'o-')
axes[1].axhline(y=100, color='r', linestyle='--', label='Ideal (100%)')
axes[1].set_xlabel('Number of Processes (np)')
axes[1].set_ylabel('Parallel Efficiency (%)')
axes[1].set_title('Parallel Efficiency')
axes[1].legend()
axes[1].grid(True)

# GFLOPS
axes[2].plot(grouped.index, grouped['gflops'], 'o-')
axes[2].set_xlabel('Number of Processes (np)')
axes[2].set_ylabel('GFLOPS')
axes[2].set_title('Performance vs Processes')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('strong_scaling_analysis.png', dpi=300)
plt.show()
```

## Troubleshooting

### "Permission denied" when running scripts

```bash
chmod +x slurm_helpers/*.slurm batch_helpers/*.sh
```

### "Failed to parse job id from: ..."

Check SLURM submission output manually:
```bash
sbatch slurm_helpers/conv2d_mpi_param.slurm 1024 1024 5 5 4
# Should output: "Submitted batch job XXXXX"
```

### No CSV file appears after job completes

1. Check SLURM error log: `cat logs/conv2d_mpi_*.err`
2. Look for compilation errors or runtime failures
3. Verify metrics directory exists: `mkdir -p metrics`
4. Check program actually ran: `grep "conv_time" logs/conv2d_mpi_*.err`

### CSV missing columns (e.g., "np", "P_rows", "P_cols")

Your program might not have the updated `log_metrics` function. Recompile:
```bash
mpicc -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra -o conv2d_mpi conv2d_mpi.c -lm
```

### Jobs stuck in queue (PD state)

Partition might be full. Check:
```bash
sinfo -p cits3402
```

Reduce concurrent jobs:
```bash
./batch_helpers/sweep_conv2d_mpi.sh ... 10  # Last param = MAX_IN_FLIGHT
```

## Next Steps

1. **Verify correctness**: Test small problems with different np values
2. **Baseline performance**: Run single jobs to understand timing
3. **Strong scaling**: Fix problem size, vary np
4. **Weak scaling**: Scale problem size with np
5. **Hybrid tuning**: Compare MPI-only vs MPI+OpenMP
6. **Analysis**: Download CSVs and generate plots

For detailed documentation, see **TESTING_FRAMEWORK.md**
