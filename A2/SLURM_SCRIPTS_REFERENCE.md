# SLURM Scripts Reference - Kaya HPC Cluster

## Complete Script Collection (Following Unit Content Standards)

All scripts follow the **CITS3402 unit content format** with proper module loading and resource specification.

### ✅ Script Inventory (8 scripts)

#### 1D Convolution (4 scripts)

1. **conv1d_seq_param.slurm** - Sequential baseline
2. **conv1d_omp_param.slurm** - OpenMP parallelization  
3. **conv1d_mpi_param.slurm** - Pure MPI (1D decomposition)
4. **conv1d_mpi_omp_param.slurm** - Hybrid MPI+OpenMP

#### 2D Convolution (4 scripts)

1. **conv2d_seq_param.slurm** - Sequential baseline
2. **conv2d_omp_param.slurm** - OpenMP parallelization
3. **conv2d_mpi_param.slurm** - Pure MPI (2D Cartesian topology)
4. **conv2d_mpi_omp_param.slurm** - Hybrid MPI+OpenMP (2D topology)

---

## SLURM Header Format (Unit Standard)

All scripts follow this format from the unit content:

```bash
#!/bin/bash
#SBATCH --job-name=<program>_param
#SBATCH --nodes=<N>                    # Number of compute nodes
#SBATCH --ntasks=<T>                   # Total MPI tasks
#SBATCH --ntasks-per-node=<TPD>        # Tasks per node (optional but recommended)
#SBATCH --cpus-per-task=<C>            # CPUs per task (for OpenMP threads)
#SBATCH --time=00:15:00                # Wall time limit
#SBATCH --mem=<M>G                     # Memory per node
#SBATCH --partition=cits3402           # Partition name
#SBATCH --output=logs/<program>_%j.out
#SBATCH --error=logs/<program>_%j.err

# Load modules (following unit example)
module load gcc/14.2          # For cc/mpicc
module load openmpi/5.0.5     # For mpiexec (MPI programs only)
```

### Key Resource Parameters

| Script | Nodes | Tasks | Tasks/Node | CPUs/Task | Total Cores | Notes |
|--------|-------|-------|------------|-----------|-------------|-------|
| **conv1d_seq** | 1 | 1 | 1 | 1 | 1 | Single-threaded |
| **conv1d_omp** | 1 | 1 | 1 | 8 | 8 | 8 OpenMP threads |
| **conv1d_mpi** | 4 | 4 | 1 | 1 | 4 | 1 MPI task/node |
| **conv1d_mpi_omp** | 4 | 4 | 1 | 4 | 16 | Hybrid: 4×4 |
| **conv2d_seq** | 1 | 1 | 1 | 1 | 1 | Single-threaded |
| **conv2d_omp** | 1 | 1 | 1 | 8 | 8 | 8 OpenMP threads |
| **conv2d_mpi** | 4 | 4 | 1 | 1 | 4 | 2×2 MPI grid |
| **conv2d_mpi_omp** | 4 | 4 | 1 | 4 | 16 | Hybrid: 4×4 |

---

## 1. Sequential Scripts

### conv1d_seq_param.slurm

**Purpose**: Sequential 1D convolution baseline

**Resources**:
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
```

**Usage**:
```bash
sbatch slurm_helpers/conv1d_seq_param.slurm N K [MODE] [PADDING] [SEED]
```

**Examples**:
```bash
# Basic test
sbatch slurm_helpers/conv1d_seq_param.slurm 1000000 101

# With seed for reproducibility
sbatch slurm_helpers/conv1d_seq_param.slurm 10000000 1001 same zero 42

# Full mode
sbatch slurm_helpers/conv1d_seq_param.slurm 500000 101 full zero
```

### conv2d_seq_param.slurm

**Purpose**: Sequential 2D convolution baseline

**Resources**:
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
```

**Usage**:
```bash
sbatch slurm_helpers/conv2d_seq_param.slurm H W kH kW [MODE] [PADDING] [SEED]
```

**Examples**:
```bash
# Basic test
sbatch slurm_helpers/conv2d_seq_param.slurm 1024 1024 5 5

# Large problem
sbatch slurm_helpers/conv2d_seq_param.slurm 4096 4096 7 7 same zero 42
```

---

## 2. OpenMP Scripts

### conv1d_omp_param.slurm

**Purpose**: OpenMP-parallelized 1D convolution

**Resources**:
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8    # 8 OpenMP threads
#SBATCH --mem=8G
```

**Usage**:
```bash
sbatch slurm_helpers/conv1d_omp_param.slurm N K [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]
```

**Examples**:
```bash
# Default: 8 threads, static schedule
sbatch slurm_helpers/conv1d_omp_param.slurm 10000000 1001

# 4 threads with dynamic scheduling
sbatch slurm_helpers/conv1d_omp_param.slurm 10000000 1001 4 dynamic

# Custom chunk size
sbatch slurm_helpers/conv1d_omp_param.slurm 10000000 1001 8 dynamic 100 same zero 42
```

### conv2d_omp_param.slurm

**Purpose**: OpenMP-parallelized 2D convolution

**Resources**:
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
```

**Usage**:
```bash
sbatch slurm_helpers/conv2d_omp_param.slurm H W kH kW [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]
```

**Examples**:
```bash
# Default: 8 threads
sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5

# Compare schedules
sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 8 static
sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 8 dynamic
sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 8 guided
```

---

## 3. Pure MPI Scripts

### conv1d_mpi_param.slurm

**Purpose**: Pure MPI 1D convolution (1D decomposition)

**Resources**:
```bash
#SBATCH --nodes=4
#SBATCH --ntasks=4           # Total MPI processes
#SBATCH --ntasks-per-node=1  # 1 process per node
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
```

**Execution**: Uses `mpiexec -n <NP>` (unit standard)

**Usage**:
```bash
sbatch slurm_helpers/conv1d_mpi_param.slurm N K [NP] [MODE] [PADDING] [SEED]
```

**Examples**:
```bash
# Default: np=4
sbatch slurm_helpers/conv1d_mpi_param.slurm 10000000 1001

# Strong scaling: vary np
sbatch slurm_helpers/conv1d_mpi_param.slurm 20000000 1001 1 same zero 42
sbatch slurm_helpers/conv1d_mpi_param.slurm 20000000 1001 2 same zero 42
sbatch slurm_helpers/conv1d_mpi_param.slurm 20000000 1001 4 same zero 42
sbatch slurm_helpers/conv1d_mpi_param.slurm 20000000 1001 8 same zero 42
```

### conv2d_mpi_param.slurm

**Purpose**: Pure MPI 2D convolution with **2D Cartesian topology**

**Resources**:
```bash
#SBATCH --nodes=4
#SBATCH --ntasks=4           # Auto-decomposes to 2×2 grid
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
```

**Grid Decomposition**:
- np=1 → 1×1 grid
- np=4 → 2×2 grid
- np=9 → 3×3 grid
- np=16 → 4×4 grid

**Usage**:
```bash
sbatch slurm_helpers/conv2d_mpi_param.slurm H W kH kW [NP] [MODE] [PADDING] [SEED]
```

**Examples**:
```bash
# 2×2 grid (np=4)
sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 4

# 3×3 grid (np=9) - requires updating --ntasks and --nodes
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 9

# Strong scaling study (fixed problem)
for np in 1 4 9 16; do
  sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 $np same zero 42
done
```

---

## 4. Hybrid MPI+OpenMP Scripts

### conv1d_mpi_omp_param.slurm

**Purpose**: Hybrid MPI+OpenMP 1D convolution

**Resources**:
```bash
#SBATCH --nodes=4
#SBATCH --ntasks=4           # 4 MPI processes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # 4 OpenMP threads per process
#SBATCH --mem=8G
# Total cores: 4 × 4 = 16
```

**Usage**:
```bash
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm N K [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]
```

**Examples**:
```bash
# Default: 4 MPI × 4 OMP = 16 cores
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001

# Compare hybrid configurations (same total cores):
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 8 2  # 8×2=16
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 4 4  # 4×4=16
sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 2 8  # 2×8=16
```

### conv2d_mpi_omp_param.slurm

**Purpose**: Hybrid MPI+OpenMP 2D convolution with **2D Cartesian topology**

**Resources**:
```bash
#SBATCH --nodes=4
#SBATCH --ntasks=4           # 4 MPI processes → 2×2 grid
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # 4 OpenMP threads per process
#SBATCH --mem=16G
# Total cores: 4 × 4 = 16
```

**Usage**:
```bash
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm H W kH kW [NP] [THREADS] [SCHED] [CHUNK] [MODE] [PADDING] [SEED]
```

**Examples**:
```bash
# Default: 4 MPI × 4 OMP
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7

# Hybrid vs pure MPI (same problem, same total cores):
sbatch slurm_helpers/conv2d_mpi_param.slurm 4096 4096 7 7 16         # Pure MPI: 16×1
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 4 4   # Hybrid: 4×4
```

---

## Resource Adjustment Examples

### Scaling to More Nodes

If resources allow, increase parallelism by adjusting `--nodes` and `--ntasks`:

```bash
# 8 MPI processes across 8 nodes
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1

# Or 8 processes across 4 nodes (2 per node)
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
```

### Limited Node Availability

From unit content: "If you cannot request 4 nodes, then try 2 nodes"

```bash
# Original: 4 nodes × 1 task/node = 4 tasks
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1

# Adjusted: 2 nodes × 2 tasks/node = 4 tasks
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
```

---

## Module Loading (Unit Standard)

All MPI scripts load:
```bash
module load gcc/14.2
module load openmpi/5.0.5
```

All non-MPI scripts load:
```bash
module load gcc/14.2
```

### Available Modules on Kaya

Check available modules:
```bash
module avail gcc
module avail openmpi
```

---

## Execution Commands

### Sequential/OpenMP Programs
Executed directly:
```bash
./conv1d -L 1000000 -kL 101 -o /dev/null
./conv2d_omp -H 2048 -W 2048 -kH 5 -kW 5 -o /dev/null --threads 8
```

### MPI Programs
Executed with `mpiexec` (unit standard):
```bash
mpiexec -n 4 ./conv1d_mpi -L 10000000 -kL 1001 -o /dev/null
mpiexec -n 4 ./conv2d_mpi -H 2048 -W 2048 -kH 5 -kW 5 -o /dev/null
```

**Note**: Unit content uses `mpiexec` (not `mpirun`)

---

## Verification Commands

### Check Job Status
```bash
squeue --me
```

### View Output
```bash
# Latest job output
ls -t logs/*.out | head -1 | xargs cat

# Latest error log
ls -t logs/*.err | head -1 | xargs tail -20
```

### Verify MPI Distribution
Check that processes run on different nodes (from unit content):
```bash
# Should show different hostnames for different ranks
grep "rank" logs/conv2d_mpi_*.err
```

### Check Latest Metrics
```bash
cat $(ls -t metrics/metrics_SLURM_*.csv | head -1)
```

---

## Performance Comparison Workflow

### 1. Sequential Baseline
```bash
sbatch slurm_helpers/conv2d_seq_param.slurm 2048 2048 5 5 same zero 42
```

### 2. OpenMP Speedup
```bash
for threads in 1 2 4 8; do
  sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 $threads static "" same zero 42
done
```

### 3. MPI Speedup
```bash
for np in 1 4 9 16; do
  sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 $np same zero 42
done
```

### 4. Hybrid Exploration (16 total cores)
```bash
# Pure MPI
sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 16 same zero 42

# Hybrid configurations
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 8 2 static "" same zero 42
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 4 4 static "" same zero 42
sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 2 8 static "" same zero 42

# Pure OpenMP
sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 16 static "" same zero 42
```

---

## Common Issues and Solutions

### 1. "Module not found"
```bash
# Check available modules
module avail gcc
module avail openmpi

# Load correct versions
module load gcc/14.2
module load openmpi/5.0.5
```

### 2. "Requested more tasks than available"
Reduce `--ntasks` or `--nodes`:
```bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
```

### 3. "Job pending (Resources)"
Queue is full. Check with:
```bash
squeue -p cits3402
```

### 4. Compilation fails
Check module loading in script and ensure source files exist.

---

## Summary Table

| Program | Parallelism | Nodes | Tasks | Threads/Task | Total Cores | Script Name |
|---------|-------------|-------|-------|--------------|-------------|-------------|
| conv1d | None | 1 | 1 | 1 | 1 | conv1d_seq_param.slurm |
| conv1d_omp | OpenMP | 1 | 1 | 8 | 8 | conv1d_omp_param.slurm |
| conv1d_mpi | MPI | 4 | 4 | 1 | 4 | conv1d_mpi_param.slurm |
| conv1d_mpi_omp | Hybrid | 4 | 4 | 4 | 16 | conv1d_mpi_omp_param.slurm |
| conv2d | None | 1 | 1 | 1 | 1 | conv2d_seq_param.slurm |
| conv2d_omp | OpenMP | 1 | 1 | 8 | 8 | conv2d_omp_param.slurm |
| conv2d_mpi | MPI (2D) | 4 | 4 | 1 | 4 | conv2d_mpi_param.slurm |
| conv2d_mpi_omp | Hybrid (2D) | 4 | 4 | 4 | 16 | conv2d_mpi_omp_param.slurm |

---

**All scripts follow CITS3402 unit content standards with:**
- ✅ Proper `--nodes`, `--ntasks`, `--ntasks-per-node` specification
- ✅ Module loading (`gcc/14.2`, `openmpi/5.0.5`)
- ✅ `mpiexec -n` for MPI execution
- ✅ Parameterized arguments (no manual editing)
- ✅ CSV metrics auto-logging
- ✅ Separate stdout/stderr logs
