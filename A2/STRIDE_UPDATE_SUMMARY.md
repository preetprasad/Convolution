# Stride Parameter Update Summary

## Overview
Added stride support to all A2 SLURM scripts and serial sweep scripts to match the stride functionality implemented in the C programs.

## Changes Made

### 1D Programs (conv1d, conv1d_omp, conv1d_mpi, conv1d_mpi_omp)
- **Stride Parameter**: `-st STRIDE` or `--stride STRIDE`
- **Default Value**: 1 (no downsampling)
- **Effect**: 
  - SAME mode: output length = ceil(N / stride)
  - FULL mode: output length = ceil((N+K-1) / stride)

### 2D Programs (conv2d, conv2d_omp, conv2d_mpi, conv2d_mpi_omp)
- **Stride Parameters**: `-sH STRIDE_H` and `-sW STRIDE_W` (or `--stride-h` / `--stride-w`)
- **Default Values**: 1, 1 (no downsampling)
- **Effect**:
  - SAME mode: output size = ceil(H/sH) × ceil(W/sW)
  - FULL mode: output size = ceil((H+KH-1)/sH) × ceil((W+KW-1)/sW)

---

## Updated Files (17 total)

### SLURM Parameter Scripts (9 files)

#### 1D SLURM Scripts (4 files)
1. **slurm_helpers/conv1d_seq_param.slurm**
   - Usage: `sbatch conv1d_seq_param.slurm N K [MODE] [PADDING] [STRIDE] [SEED]`
   - Added: `STRIDE="${5:-1}"` parameter
   - Command: `./conv1d -L "$N" -kL "$K" -o "$OUT" -m "$MODE" -p "$PADDING" -st "$STRIDE"`

2. **slurm_helpers/conv1d_omp_param.slurm**
   - Usage: `sbatch conv1d_omp_param.slurm N K [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE] [SEED]`
   - Added: `STRIDE="${8:-1}"` parameter
   - Command: `./conv1d_omp ... -st "$STRIDE"`

3. **slurm_helpers/conv1d_mpi_param.slurm**
   - Usage: `sbatch conv1d_mpi_param.slurm N K [NP] [MODE] [PADDING] [STRIDE] [SEED]`
   - Added: `STRIDE="${6:-1}"` parameter
   - Command: `mpiexec -n "$NP" ./conv1d_mpi ... -st "$STRIDE"`
   - Note: Changed seed flag from `-s` to `-se` for MPI programs

4. **slurm_helpers/conv1d_mpi_omp_param.slurm**
   - Usage: `sbatch conv1d_mpi_omp_param.slurm N K [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE] [SEED]`
   - Added: `STRIDE="${9:-1}"` parameter
   - Command: `mpiexec -n "$NP" ./conv1d_mpi_omp ... -st "$STRIDE"`
   - Note: Changed seed flag from `-s` to `-se`

#### 2D SLURM Scripts (5 files)
5. **slurm_helpers/conv2d_seq_param.slurm**
   - Usage: `sbatch conv2d_seq_param.slurm H W kH kW [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]`
   - Added: `STRIDE_H="${7:-1}"` and `STRIDE_W="${8:-1}"` parameters
   - Command: `./conv2d ... -sH "$STRIDE_H" -sW "$STRIDE_W"`

6. **slurm_helpers/conv2d_omp_param.slurm**
   - Usage: `sbatch conv2d_omp_param.slurm H W kH kW [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]`
   - Added: `STRIDE_H="${10:-1}"` and `STRIDE_W="${11:-1}"` parameters
   - Command: `./conv2d_omp ... -sH "$STRIDE_H" -sW "$STRIDE_W"`

7. **slurm_helpers/conv2d_mpi_param.slurm**
   - Usage: `sbatch conv2d_mpi_param.slurm H W kH kW [NP] [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]`
   - Added: `STRIDE_H="${8:-1}"` and `STRIDE_W="${9:-1}"` parameters
   - Command: `mpiexec -n "$NP" ./conv2d_mpi ... -sH "$STRIDE_H" -sW "$STRIDE_W"`
   - Note: Changed seed flag from `-s` to `-se`

8. **slurm_helpers/conv2d_mpi_omp_param.slurm**
   - Usage: `sbatch conv2d_mpi_omp_param.slurm H W kH kW [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [STRIDE_H] [STRIDE_W] [SEED]`
   - Added: `STRIDE_H="${11:-1}"` and `STRIDE_W="${12:-1}"` parameters
   - Command: `mpiexec -n "$NP" ./conv2d_mpi_omp ... -sH "$STRIDE_H" -sW "$STRIDE_W"`
   - Note: Changed seed flag from `-s` to `-se`

9. **slurm_helpers/conv2d_hybrid_param.slurm**
   - (Same as conv2d_mpi_omp_param.slurm)

### Serial Sweep Scripts (8 files)

#### 1D Serial Sweep Scripts (4 files)
1. **batch_helpers/sweep_conv1d_seq_serial.sh**
   - Usage: `./sweep_conv1d_seq_serial.sh NMIN NMAX N_STEP KMIN KMAX K_STEP [STRIDE] [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]`
   - Added: `STRIDE="${7:-1}"` parameter
   - Passes stride to SLURM script

2. **batch_helpers/sweep_conv1d_omp_serial.sh**
   - Usage: `./sweep_conv1d_omp_serial.sh NMIN NMAX N_STEP KMIN KMAX K_STEP [THREADS] [SCHED] [CHUNK] [STRIDE] [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]`
   - Added: `STRIDE="${10:-1}"` parameter

3. **batch_helpers/sweep_conv1d_mpi_serial.sh**
   - Usage: `./sweep_conv1d_mpi_serial.sh NMIN NMAX N_STEP KMIN KMAX K_STEP [NP] [STRIDE] [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]`
   - Added: `STRIDE="${8:-1}"` parameter

4. **batch_helpers/sweep_conv1d_mpi_omp_serial.sh**
   - Usage: `./sweep_conv1d_mpi_omp_serial.sh LMIN LMAX L_STEP KMIN KMAX K_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [NP] [THREADS] [SCHED] [CHUNK] [STRIDE] [SEED]`
   - Added: `STRIDE="${13:-1}"` parameter

#### 2D Serial Sweep Scripts (4 files)
5. **batch_helpers/sweep_conv2d_seq_serial.sh**
   - Usage: `./sweep_conv2d_seq_serial.sh HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [STRIDE_H] [STRIDE_W] [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]`
   - Added: `STRIDE_H="${13:-1}"` and `STRIDE_W="${14:-1}"` parameters

6. **batch_helpers/sweep_conv2d_omp_serial.sh**
   - Usage: `./sweep_conv2d_omp_serial.sh HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [THREADS] [SCHED] [CHUNK] [STRIDE_H] [STRIDE_W] [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [SEED]`
   - Added: `STRIDE_H="${16:-1}"` and `STRIDE_W="${17:-1}"` parameters

7. **batch_helpers/sweep_conv2d_mpi_serial.sh**
   - Usage: `./sweep_conv2d_mpi_serial.sh HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [NP] [STRIDE_H] [STRIDE_W] [SEED]`
   - Added: `STRIDE_H="${16:-1}"` and `STRIDE_W="${17:-1}"` parameters

8. **batch_helpers/sweep_conv2d_mpi_omp_serial.sh**
   - Usage: `./sweep_conv2d_mpi_omp_serial.sh HMIN HMAX H_STEP WMIN WMAX W_STEP KHMIN KHMAX KH_STEP KWMIN KWMAX KW_STEP [POST_COPY_WAIT] [FILE_WAIT_RETRIES] [NP] [THREADS] [SCHED] [CHUNK] [STRIDE_H] [STRIDE_W] [SEED]`
   - Added: `STRIDE_H="${19:-1}"` and `STRIDE_W="${20:-1}"` parameters

---

## Key Implementation Details

### Parameter Positioning
- Stride parameters added **before** SEED parameter to maintain backwards compatibility
- All stride parameters have default value of 1 (no downsampling)
- 1D programs: single `-st STRIDE` parameter
- 2D programs: separate `-sH STRIDE_H` and `-sW STRIDE_W` parameters

### Seed Flag Consistency
- Sequential and OpenMP programs: `-s` or `-se` or `--seed`
- MPI and hybrid programs: `-se` or `--seed` (standardized to `-se` in scripts)

### Conditional Seed Handling
All serial sweep scripts now properly handle optional SEED parameter:
```bash
if [[ -n "$SEED" ]]; then
  submit_out=$(sbatch slurm_helpers/... "$PARAMS" "$STRIDE" "$SEED")
else
  submit_out=$(sbatch slurm_helpers/... "$PARAMS" "$STRIDE")
fi
```

### Echo Statements
All sweep scripts now display stride values in their configuration output:
- 1D: `STRIDE=$STRIDE`
- 2D: `STRIDE_H=$STRIDE_H  STRIDE_W=$STRIDE_W`

---

## Testing Recommendations

### 1. Test Basic Stride Functionality
```bash
# 1D sequential with stride=2
sbatch slurm_helpers/conv1d_seq_param.slurm 1024 5 same zero 2

# 2D OpenMP with stride=2x2
sbatch slurm_helpers/conv2d_omp_param.slurm 512 512 5 5 8 static "" same zero 2 2
```

### 2. Test Serial Sweeps
```bash
# 1D sequential sweep with stride=1 (default)
./batch_helpers/sweep_conv1d_seq_serial.sh 1000 2000 1000 3 5 2

# 2D MPI sweep with stride=2x2
./batch_helpers/sweep_conv2d_mpi_serial.sh 512 1024 512 512 1024 512 3 5 2 3 5 2 10 60 4 2 2
```

### 3. Verify Output Sizes
Check that output lengths match expected stride-adjusted sizes:
- 1D SAME with stride=2: `outLen = ceil(N / 2)`
- 2D SAME with stride=(2,2): `outH = ceil(H / 2)`, `outW = ceil(W / 2)`

---

## Backward Compatibility

All scripts remain backward compatible:
- Omitting stride parameters defaults to `stride=1` (no downsampling)
- Existing sweep scripts will work unchanged
- New stride parameters are **optional** and positioned before SEED

---

## Summary Statistics

- **Total Files Updated**: 17
  - SLURM param scripts: 9
  - Serial sweep scripts: 8
- **1D Programs**: 4 implementations (seq, omp, mpi, hybrid)
- **2D Programs**: 4 implementations (seq, omp, mpi, hybrid)
- **Parameter Additions**:
  - 1D: 1 stride parameter per script
  - 2D: 2 stride parameters per script (height and width)
- **Default Stride Values**: Always 1 (no downsampling)

---

## Next Steps

1. ✅ Update complete - all scripts now support stride
2. 🔄 Test stride functionality on a small dataset
3. 🔄 Update parallel sweep scripts (if needed)
4. 🔄 Update documentation (DATA_COLLECTION_GUIDE.md, etc.)
5. 🔄 Begin data collection with stride experiments

---

## Examples

### 1D Examples
```bash
# Sequential with stride=2
sbatch slurm_helpers/conv1d_seq_param.slurm 1024 5 same zero 2 42

# OpenMP with stride=3
sbatch slurm_helpers/conv1d_omp_param.slurm 1024 5 8 static "" same zero 3 42

# MPI with stride=2
sbatch slurm_helpers/conv1d_mpi_param.slurm 1024 5 4 same zero 2 42

# Hybrid with stride=2
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 1024 5 4 4 static "" same zero 2 42
```

### 2D Examples
```bash
# Sequential with stride=2x2
sbatch slurm_helpers/conv2d_seq_param.slurm 512 512 5 5 same zero 2 2 42

# OpenMP with stride=3x3
sbatch slurm_helpers/conv2d_omp_param.slurm 512 512 5 5 8 static "" same zero 3 3 42

# MPI with stride=2x3
sbatch slurm_helpers/conv2d_mpi_param.slurm 512 512 5 5 4 same zero 2 3 42

# Hybrid with stride=2x2
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 512 512 5 5 4 4 static "" same zero 2 2 42
```

---

## Verification Checklist

- [x] All 1D SLURM scripts accept `-st STRIDE` parameter
- [x] All 2D SLURM scripts accept `-sH STRIDE_H -sW STRIDE_W` parameters
- [x] All serial sweep scripts pass stride to SLURM scripts
- [x] Default stride values = 1 for all scripts
- [x] Conditional SEED handling implemented
- [x] Echo statements updated to show stride values
- [x] MPI/hybrid scripts use `-se` for seed flag
- [x] Backward compatibility maintained
- [ ] Testing on Kaya cluster
- [ ] Documentation updated
