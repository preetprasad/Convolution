# Data Collection Quick Reference

## 🚀 Quick Start (On Kaya)

```bash
# 1. Setup
ssh <username>@kaya.hpc.uwa.edu.au
cd /path/to/Convolution-1/A2
module load gcc/14.2 openmpi/5.0.5
make

# 2. Run interactive setup
./start_data_collection.sh

# Or run manually...
```

---

## 📋 Phase Checklist

### Phase 1: Correctness (5 min)
```bash
# Test all 8 implementations with small problems
sbatch slurm_helpers/conv1d_seq_param.slurm 10000 101 same zero 42
sbatch slurm_helpers/conv1d_omp_param.slurm 10000 101 4 static "" same zero 42
sbatch slurm_helpers/conv1d_mpi_param.slurm 10000 101 2 same zero 42
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 10000 101 2 2 static "" same zero 42
sbatch slurm_helpers/conv2d_seq_param.slurm 128 128 5 5 same zero 42
sbatch slurm_helpers/conv2d_omp_param.slurm 128 128 5 5 4 static "" same zero 42
sbatch slurm_helpers/conv2d_mpi_param.slurm 128 128 5 5 4 same zero 42
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 128 128 5 5 4 2 static "" same zero 42
```

### Phase 2: Sequential Baselines (2-4 hours)
```bash
# 1D sweep: 10 × 10 = 100 jobs
./batch_helpers/sweep_conv1d_seq.sh 1000000 10000000 1000000 101 1001 100 20 42

# 2D sweep: 7 × 7 × 3 × 3 = 441 jobs
./batch_helpers/sweep_conv2d_seq.sh 1024 4096 512 1024 4096 512 3 7 2 3 7 2 20 42
```

### Phase 3: OpenMP Scaling (3-6 hours)
```bash
# Thread scaling (1,2,4,8,16 threads)
for t in 1 2 4 8 16; do
  sbatch slurm_helpers/conv1d_omp_param.slurm 5000000 501 $t static "" same zero 42
  sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 $t static "" same zero 42
done

# Schedule comparison
for sched in static dynamic guided auto; do
  sbatch slurm_helpers/conv1d_omp_param.slurm 5000000 501 8 $sched "" same zero 42
done

# 2D sweep
./batch_helpers/sweep_conv2d_omp.sh 1024 2048 512 1024 2048 512 5 5 2 5 5 2 20 8 static "" 42
```

### Phase 4: MPI Scaling (4-8 hours)
```bash
# 1D Strong scaling (fixed problem, vary np)
for np in 1 2 4 8 16; do
  sbatch slurm_helpers/conv1d_mpi_param.slurm 20000000 1001 $np same zero 42
done

# 1D Weak scaling (scale problem with np)
sbatch slurm_helpers/conv1d_mpi_param.slurm 5000000  1001 1  same zero 42
sbatch slurm_helpers/conv1d_mpi_param.slurm 10000000 1001 2  same zero 42
sbatch slurm_helpers/conv1d_mpi_param.slurm 20000000 1001 4  same zero 42
sbatch slurm_helpers/conv1d_mpi_param.slurm 40000000 1001 8  same zero 42

# 2D Grid scaling (1×1, 2×2, 3×3, 4×4, 5×5)
for np in 1 4 9 16 25; do
  sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 $np same zero 42
done

# 2D sweep
./batch_helpers/sweep_conv2d_mpi.sh 2048 4096 1024 2048 4096 1024 5 7 2 5 7 2 20 4 42
```

### Phase 5: Hybrid (4-8 hours)
```bash
# 1D: 16 cores, vary MPI×OMP split
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 16 1  static "" same zero 42  # Pure MPI
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 8  2  static "" same zero 42  # 8×2
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 4  4  static "" same zero 42  # 4×4
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 2  8  static "" same zero 42  # 2×8
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 1  16 static "" same zero 42  # Pure OMP

# 2D: 16 cores, vary grid×OMP
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 16 1  static "" same zero 42  # 4×4 grid
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 4  4  static "" same zero 42  # 2×2 grid
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 1  16 static "" same zero 42  # Pure OMP

# Sweeps
./batch_helpers/sweep_conv1d_mpi_omp.sh 5000000 20000000 5000000 501 1001 500 20 4 4 static "" 42
./batch_helpers/sweep_conv2d_mpi_omp.sh 2048 4096 1024 2048 4096 1024 5 7 2 5 7 2 20 4 4 static "" 42
```

---

## 🔍 Monitoring Commands

```bash
# Check queue
squeue -u $USER

# Count completed jobs
ls logs/*.err | wc -l

# Count metrics files
ls metrics/*.csv | wc -l

# Watch latest logs
tail -f logs/conv1d_*.err
tail -f logs/conv2d_*.err

# Check for errors
grep -i error logs/*.err
grep -i "failed\|abort" logs/*.err

# View recent job history
sacct -u $USER -S $(date -d '1 day ago' +%Y-%m-%d)

# Check failed jobs
sacct -u $USER -S $(date -d '1 day ago' +%Y-%m-%d) --state=FAILED
```

---

## 📊 Data Verification

```bash
# Merge all metrics
cat metrics/*.csv | awk 'NR==1 || !/^RunID/' > merged_metrics.csv

# Count data points
wc -l merged_metrics.csv

# Check GFLOPS distribution
grep GFLOPS logs/*.err | awk -F'=' '{print $2}' | sort -n

# Verify CSV integrity
for csv in metrics/*.csv; do
  awk -F',' 'NR==1 {print FILENAME": "NF" columns"}' "$csv"
done

# Check for missing values
grep -E ",,|NaN|inf" metrics/*.csv
```

---

## 🎯 Minimal Dataset (Fast Start)

For quick results to begin analysis (~30 jobs, 1-2 hours):

```bash
# Baselines
sbatch slurm_helpers/conv1d_seq_param.slurm 5000000 501 same zero 42
sbatch slurm_helpers/conv2d_seq_param.slurm 2048 2048 5 5 same zero 42

# OpenMP (1,2,4,8)
for t in 1 2 4 8; do
  sbatch slurm_helpers/conv1d_omp_param.slurm 5000000 501 $t static "" same zero 42
  sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 $t static "" same zero 42
done

# MPI (1,4,9)
for np in 1 4 9; do
  sbatch slurm_helpers/conv1d_mpi_param.slurm 5000000 501 $np same zero 42
  sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 $np same zero 42
done

# Hybrid (4×4)
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 5000000 501 4 4 static "" same zero 42
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 4 4 static "" same zero 42
```

---

## 🛠️ Troubleshooting

| Issue | Command | Fix |
|-------|---------|-----|
| Jobs pending | `squeue -u $USER` | Wait for resources or reduce tasks |
| Out of memory | Check `logs/*.err` | Increase `--mem` in SLURM scripts |
| Jobs timeout | Check time limits | Increase `--time` in scripts |
| No metrics | `ls metrics/` | Check logs for errors, rebuild programs |
| Disk quota | `quota -s` | Clean up old files |

---

## 📈 Expected Data Points

| Phase | 1D Jobs | 2D Jobs | Time | Total Points |
|-------|---------|---------|------|--------------|
| Correctness | 4 | 4 | 5 min | 8 |
| Sequential | 100 | 441 | 2-4 hrs | 541 |
| OpenMP | 50 | 200 | 3-6 hrs | 250 |
| MPI | 100 | 300 | 4-8 hrs | 400 |
| Hybrid | 100 | 300 | 4-8 hrs | 400 |
| **Total** | **~350** | **~1200** | **15-30 hrs** | **~1600** |

---

## 🎓 Analysis Tips

### Metrics to Calculate

- **Speedup**: `T_seq / T_parallel`
- **Efficiency**: `Speedup / num_cores`
- **Parallel Overhead**: `T_parallel - (T_seq / num_cores)`
- **Strong Scaling**: Fixed problem, vary cores
- **Weak Scaling**: Scale problem with cores

### Visualizations

- Speedup vs. cores (log-log)
- Efficiency vs. cores
- GFLOPS vs. problem size
- MPI×OMP heatmap (hybrid)
- OpenMP schedule comparison

### Key Questions

1. What's the optimal thread count for OpenMP?
2. How does 2D Cartesian topology scale vs. 1D?
3. What's the best MPI×OMP balance for hybrid?
4. When does communication overhead dominate?
5. Which schedule (static/dynamic/guided) is best?

---

**See Also**:
- `DATA_COLLECTION_GUIDE.md` - Comprehensive guide
- `TESTING_FRAMEWORK.md` - Script reference
- `SLURM_SCRIPTS_REFERENCE.md` - SLURM details

**Ready?** Run `./start_data_collection.sh` to begin! 🚀
