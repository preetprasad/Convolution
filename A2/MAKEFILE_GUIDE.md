# A2 Makefile Guide

## Overview

The A2 Makefile builds and runs **8 different convolution implementations**:

| Implementation | 1D | 2D | Description |
|----------------|----|----|-------------|
| **Sequential** | ✅ | ✅ | Single-threaded baseline |
| **OpenMP** | ✅ | ✅ | Shared-memory parallel (multi-core) |
| **MPI** | ✅ | ✅ | Distributed-memory parallel (multi-node) |
| **Hybrid** | ✅ | ✅ | MPI + OpenMP combined |

### Key Differences from A1 Makefile

| Feature | A1 | A2 |
|---------|----|----|
| Programs | 4 (seq, omp for 1D/2D) | 8 (seq, omp, mpi, hybrid for 1D/2D) |
| Compilers | `cc` only | `cc` + `mpicc` |
| Execution | Direct (`./program`) | Direct + `mpiexec -n` |
| MPI Parameters | N/A | `NP` (number of processes) |
| 2D Topology | N/A | 2D Cartesian grid (auto-decomposition) |

---

## Quick Start

### Build Everything
```bash
make
# Builds all 8 executables
```

### Build by Category
```bash
make seq     # Build sequential only (conv1d, conv2d)
make omp     # Build OpenMP only (conv1d_omp, conv2d_omp)
make mpi     # Build MPI only (conv1d_mpi, conv2d_mpi)
make hybrid  # Build hybrid only (conv1d_mpi_omp, conv2d_mpi_omp)
```

### Run Quick Tests
```bash
make test
# Runs all 8 programs with small problem sizes for verification
```

### Clean Up
```bash
make clean
# Removes all binaries and output directories
```

---

## Building Programs

### Compilation Details

**Sequential and OpenMP programs** use `cc`:
```bash
cc -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra -o conv1d conv1d.c -lm
cc -std=c11 -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra -o conv1d_omp conv1d_omp.c -lm
```

**MPI programs** use `mpicc`:
```bash
mpicc -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra -o conv1d_mpi conv1d_mpi.c -lm
```

**Hybrid programs** use `mpicc` with OpenMP:
```bash
mpicc -std=c11 -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra -o conv1d_mpi_omp conv1d_mpi_omp.c -lm
```

### On Kaya

Before compiling on Kaya, load required modules:
```bash
module load gcc/14.2
module load openmpi/5.0.5
make
```

---

## Running Programs

### Default Parameters

```bash
# 1D defaults
N=1000000      # Input length
K=10001        # Kernel length

# 2D defaults
H=1024         # Height
W=1024         # Width
KH=5           # Kernel height
KW=5           # Kernel width

# Parallelism defaults
THREADS=8      # OpenMP threads
NP=4           # MPI processes
SCHED=static   # OpenMP schedule
SEED=42        # RNG seed
```

### Sequential Programs

```bash
# 1D Sequential
make run1d
# Equivalent to: ./conv1d -L 1000000 -kL 10001 -o /dev/null -s 42

# 2D Sequential
make run2d
# Equivalent to: ./conv2d -H 1024 -W 1024 -kH 5 -kW 5 -o /dev/null -s 42

# Override parameters
make run1d N=2000000 K=5001
make run2d H=2048 W=2048 KH=7 KW=7
```

### OpenMP Programs

```bash
# 1D OpenMP (default: 8 threads, static schedule)
make run1d_omp

# With custom parameters
make run1d_omp N=10000000 K=1001 THREADS=16 SCHED=dynamic

# 2D OpenMP
make run2d_omp H=2048 W=2048 KH=5 KW=5 THREADS=8

# Try different schedules
make run2d_omp THREADS=8 SCHED=static
make run2d_omp THREADS=8 SCHED=dynamic
make run2d_omp THREADS=8 SCHED=guided
make run2d_omp THREADS=8 SCHED=auto

# With chunk size
make run2d_omp THREADS=8 SCHED=dynamic CHUNK=100
```

### MPI Programs

```bash
# 1D MPI (default: 4 processes)
make run1d_mpi

# With custom process count
make run1d_mpi N=20000000 K=1001 NP=8

# 2D MPI (2D Cartesian topology)
make run2d_mpi
# NP=4 creates 2×2 grid automatically

# Different grid sizes
make run2d_mpi H=2048 W=2048 KH=5 KW=5 NP=1   # 1×1 grid
make run2d_mpi H=2048 W=2048 KH=5 KW=5 NP=4   # 2×2 grid
make run2d_mpi H=4096 W=4096 KH=7 KW=7 NP=9   # 3×3 grid
make run2d_mpi H=4096 W=4096 KH=7 KW=7 NP=16  # 4×4 grid
```

### Hybrid MPI+OpenMP Programs

```bash
# 1D Hybrid (default: 4 MPI × 8 OMP = 32 cores)
make run1d_hybrid

# Custom configuration
make run1d_hybrid N=20000000 K=1001 NP=4 THREADS=4

# 2D Hybrid (2D Cartesian topology + OpenMP)
make run2d_hybrid H=4096 W=4096 KH=7 KW=7 NP=4 THREADS=4
# Total cores: 4 MPI × 4 OMP = 16 cores

# Compare different MPI×OMP balances (same total cores):
make run2d_hybrid NP=16 THREADS=1   # Pure MPI: 16×1
make run2d_hybrid NP=8  THREADS=2   # Balanced: 8×2
make run2d_hybrid NP=4  THREADS=4   # Balanced: 4×4
make run2d_hybrid NP=2  THREADS=8   # More OMP: 2×8
make run2d_hybrid NP=1  THREADS=16  # Pure OMP: 1×16
```

---

## Performance Comparison Workflows

### Workflow 1: Sequential Baseline

```bash
# Get baseline performance
make run1d N=10000000 K=1001 SEED=42
make run2d H=2048 W=2048 KH=5 KW=5 SEED=42
```

### Workflow 2: OpenMP Scaling

```bash
# Test different thread counts
for threads in 1 2 4 8 16; do
  make run2d_omp H=2048 W=2048 KH=5 KW=5 THREADS=$threads SEED=42
done
```

### Workflow 3: MPI Strong Scaling

```bash
# Fixed problem size, vary processes
for np in 1 2 4 8 16; do
  make run2d_mpi H=4096 W=4096 KH=7 KW=7 NP=$np SEED=42
done
```

### Workflow 4: MPI Weak Scaling

```bash
# Scale problem size with processes (constant work per process)
make run2d_mpi H=1024 W=1024 KH=5 KW=5 NP=1  SEED=42  # 1×1 grid
make run2d_mpi H=2048 W=2048 KH=5 KW=5 NP=4  SEED=42  # 2×2 grid
make run2d_mpi H=3072 W=3072 KH=5 KW=5 NP=9  SEED=42  # 3×3 grid
make run2d_mpi H=4096 W=4096 KH=5 KW=5 NP=16 SEED=42  # 4×4 grid
```

### Workflow 5: Hybrid Configuration Exploration

```bash
# Compare different MPI×OMP combinations (16 total cores)
make run2d_mpi    H=2048 W=2048 KH=5 KW=5 NP=16 SEED=42          # Pure MPI
make run2d_hybrid H=2048 W=2048 KH=5 KW=5 NP=8 THREADS=2 SEED=42  # 8×2
make run2d_hybrid H=2048 W=2048 KH=5 KW=5 NP=4 THREADS=4 SEED=42  # 4×4
make run2d_hybrid H=2048 W=2048 KH=5 KW=5 NP=2 THREADS=8 SEED=42  # 2×8
make run2d_omp    H=2048 W=2048 KH=5 KW=5 THREADS=16 SEED=42     # Pure OMP
```

---

## Example Sessions

### Example 1: Quick Correctness Check

```bash
# Build everything
make

# Run quick tests (small problems)
make test

# Output shows:
# ✓ All 8 programs run successfully with small inputs
```

### Example 2: Compare All Implementations (Same Problem)

```bash
# Build all
make

# Run same problem (1024×1024, 5×5 kernel) with all implementations
make run2d              # Sequential baseline
make run2d_omp          # OpenMP (8 threads)
make run2d_mpi          # MPI (4 processes → 2×2 grid)
make run2d_hybrid       # Hybrid (4 MPI × 8 OMP = 32 cores)

# Compare times from output:
# Sequential: ~X.XXX s
# OpenMP:     ~X.XXX s  (speedup vs seq)
# MPI:        ~X.XXX s  (speedup vs seq)
# Hybrid:     ~X.XXX s  (speedup vs seq)
```

### Example 3: Find Optimal Thread Count

```bash
make
for threads in 1 2 4 8 16 32; do
  echo "Testing with $threads threads"
  make run2d_omp H=2048 W=2048 KH=5 KW=5 THREADS=$threads SEED=42
done
# Find the thread count with best GFLOPS
```

### Example 4: MPI 2D Grid Verification

```bash
make

# Test different grid configurations
make run2d_mpi H=1024 W=1024 KH=5 KW=5 NP=1   # 1×1 grid
make run2d_mpi H=1024 W=1024 KH=5 KW=5 NP=4   # 2×2 grid
make run2d_mpi H=1024 W=1024 KH=5 KW=5 NP=9   # 3×3 grid

# Stderr output shows grid decomposition:
# ranks=1 (2D: 1x1) | conv_time=...
# ranks=4 (2D: 2x2) | conv_time=...
# ranks=9 (2D: 3x3) | conv_time=...
```

---

## Advanced Usage

### Custom Compiler

```bash
# Use specific compiler
make CC=gcc-13
make MPICC=mpicc-openmpi
```

### Verbose Compilation

```bash
# Add vectorization info
make COMMON_FLAGS="-std=c11 -O3 -march=native -funroll-loops -fopt-info-vec"
```

### Custom Output Files

```bash
# Save output arrays (instead of /dev/null)
make run2d OUT2D=results/output_seq.txt
make run2d_mpi OUT2D=results/output_mpi.txt NP=4

# Compare outputs
diff results/output_seq.txt results/output_mpi.txt
```

### Environment Variables

For OpenMP programs, the Makefile automatically sets:
```bash
OMP_NUM_THREADS=$THREADS
OMP_PROC_BIND=spread
OMP_PLACES=cores
OMP_SCHEDULE=$SCHED:$CHUNK
```

You can override manually:
```bash
OMP_SCHEDULE=dynamic:10 make run2d_omp
```

---

## Troubleshooting

### Issue: `mpicc: command not found`

**Solution**: Load OpenMPI module (on Kaya):
```bash
module load gcc/14.2
module load openmpi/5.0.5
```

### Issue: `cc: command not found`

**Solution**: Install GCC or use different compiler:
```bash
make CC=gcc
```

### Issue: MPI programs fail with "No slots available"

**Solution**: Reduce `NP` to match available cores:
```bash
make run2d_mpi NP=2  # Instead of NP=4
```

### Issue: OpenMP not working (still using 1 thread)

**Solution**: Ensure OpenMP is compiled in:
```bash
# Rebuild with OpenMP
make clean
make omp

# Verify threads in output
make run2d_omp THREADS=4
# Should show: OMP_NUM_THREADS=4 in environment
```

---

## Makefile Targets Summary

### Build Targets

| Target | Description |
|--------|-------------|
| `make` | Build all 8 programs |
| `make seq` | Build sequential programs only |
| `make omp` | Build OpenMP programs only |
| `make mpi` | Build MPI programs only |
| `make hybrid` | Build hybrid programs only |
| `make clean` | Remove all binaries |

### Run Targets

| Target | Program | Parallelism |
|--------|---------|-------------|
| `make run1d` | conv1d | None |
| `make run1d_omp` | conv1d_omp | OpenMP |
| `make run1d_mpi` | conv1d_mpi | MPI |
| `make run1d_hybrid` | conv1d_mpi_omp | MPI+OpenMP |
| `make run2d` | conv2d | None |
| `make run2d_omp` | conv2d_omp | OpenMP |
| `make run2d_mpi` | conv2d_mpi | MPI (2D Cartesian) |
| `make run2d_hybrid` | conv2d_mpi_omp | MPI+OpenMP (2D Cartesian) |
| `make run` | All programs | Various |
| `make test` | All programs (small) | Quick verification |

### Helper Targets

| Target | Description |
|--------|-------------|
| `make help` | Show detailed help |
| `make dirs` | Create logs/metrics/results directories |

---

## Integration with SLURM Scripts

The Makefile is designed to work **locally** for development and testing. For **Kaya HPC cluster**, use the SLURM scripts in `slurm_helpers/`:

```bash
# Local testing (Makefile)
make run2d_mpi H=1024 W=1024 KH=5 KW=5 NP=4

# Kaya HPC (SLURM scripts)
sbatch slurm_helpers/conv2d_mpi_param.slurm 1024 1024 5 5 4
```

**Key Difference**: SLURM scripts handle:
- Module loading
- Resource allocation (nodes, tasks, memory)
- Queue management
- CSV metrics logging
- Batch job submission

**Use Makefile for**: Quick local tests and development  
**Use SLURM scripts for**: Production runs on Kaya with metrics collection

---

## Comparison: A1 vs A2 Makefile

| Feature | A1 | A2 |
|---------|----|----|
| **Programs** | 4 (seq + omp for 1D/2D) | 8 (seq + omp + mpi + hybrid for 1D/2D) |
| **Compilers** | `cc` only | `cc` + `mpicc` |
| **Parallelism** | OpenMP only | OpenMP + MPI + Hybrid |
| **Execution** | Direct | Direct + `mpiexec -n` |
| **Parameters** | N, K, H, W, THREADS, SCHED | + NP (MPI processes) |
| **Run Targets** | 5 (run, run1d, run1d_omp, run2d, run2d_omp) | 9 (+ run1d_mpi, run1d_hybrid, run2d_mpi, run2d_hybrid) |
| **Test Target** | ❌ | ✅ (tests all 8 programs) |
| **Help Target** | ❌ | ✅ (comprehensive help) |
| **2D Topology** | N/A | 2D Cartesian (MPI_Cart_create) |

---

## Tips and Best Practices

1. **Always test locally first**: Use `make test` before submitting to SLURM
2. **Use consistent seeds**: Add `SEED=42` for reproducible results
3. **Start small**: Test with small problems, then scale up
4. **Profile incrementally**: Test seq → omp → mpi → hybrid in order
5. **Save outputs**: Use `OUT2D=results/output.txt` to verify correctness
6. **Check GFLOPS**: Higher is better, compare across implementations
7. **Monitor resources**: Use `top`/`htop` to verify thread/process counts

---

**Need help?** Run `make help` for detailed usage information!
