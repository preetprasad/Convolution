# MPI Quick Reference: conv1d_mpi vs conv2d_mpi

## Side-by-Side Comparison

| Feature | conv1d_mpi | conv2d_mpi | Notes |
|---------|-----------|-----------|-------|
| **TOPOLOGY** |
| Communicator | 1D Cartesian | 1D Cartesian | Both use MPI_Cart_create |
| Dimensions | `dims[1] = {P}` | `dims[1] = {P}` | Not 2D! |
| Neighbors | left, right | top, bottom | Via MPI_Cart_shift |
| **DECOMPOSITION** |
| Strategy | Linear elements | Row-based | Entire rows per rank |
| Partition Size | N/P elements | H/P rows × W cols | W constant per rank |
| Granularity | Single elements | Full rows | Trade-off: simplicity vs scalability |
| Imbalance | ≤1 element | ≤1 row (W elements) | Minimal in both |
| **HALO EXCHANGE** |
| Width | H = K-1 | HH = KH-1 | Vertical only for 2D |
| Buffer Size | local_count + 2H | (my_rows + 2HH) × W | 2D much larger |
| Messages | 4 (left×2, right×2) | 4 (top×2, bottom×2) | Same pattern |
| Message Size | H floats | HH × W floats | Scales with width |
| Tags | 101, 102 | 101, 102 | Consistent |
| Pattern | Non-blocking | Non-blocking | Irecv → Isend → Waitall |
| **PARALLEL I/O** |
| Binary Header | `[N]` (4 bytes) | `[H][W]` (8 bytes) | int32 values |
| Input Format | `[N][data]` | `[H][W][data]` | Row-major |
| Read Method | MPI_File_read_at_all | MPI_File_read_at_all | Collective |
| Offset Calc | element-based | row-based | `start × W × sizeof(float)` |
| Datatype | Type_contiguous | Type_contiguous | Derived type |
| **COLLECTIVES** |
| Bcast | 8 scalars + kernel | 11 scalars + kernel | Config + data |
| Scatter | Partition + data | Partition + rows | Variable sizes |
| Gather | Output (text only) | Output (text only) | Root assembly |
| Reduce | Metrics (4 values) | Metrics (4 values) | SUM, MAX, MIN |
| **OUTPUT** |
| Decomposition | By index | By row | Matches input |
| Binary Header | `[outLen]` | `[outH][outW]` | Dimensions |
| Write Method | MPI_File_write_at_all | MPI_File_write_at_all | Collective |
| Rounding | 3 decimal places | 3 decimal places | nearbyint(x*1000)*0.001 |
| **PERFORMANCE** |
| Strong Scaling | Good | Better | 2D has larger work/comm |
| Weak Scaling | Near-perfect | Near-perfect | Constant work per rank |
| Memory/Rank | ~120 KB (N=10⁶, P=100) | ~3.2 MB (1024², P=4) | Problem dependent |
| GFLOP/s | 2×K×out / max_time | 2×KH×KW×out / max_time | Kernel only |
| **FEATURES** |
| Modes | SAME, FULL | SAME, FULL | ✅ Both |
| Stride | -st N | -sH M -sW N | ✅ Both |
| Padding | zero/none/const | zero/none/const | ✅ Both |
| Operations | N/A (conv only) | conv/corr | ✅ 2D has both |
| RNG Flags | -se, -s | -se, -s | ✅ Both |
| Parallel-gen | --parallel-gen | --parallel-gen | ✅ Both |
| Text Output | --text | --text | ✅ Both |
| Binary I/O | ✅ Read/Write | ✅ Read/Write | ✅ Both |
| Debug | DEBUG_MPI=1 | DEBUG_MPI=1 | ✅ Both |
| **CODE METRICS** |
| Lines of Code | ~1066 | ~1195 | Similar complexity |
| MPI Functions | 15 unique | 15 unique | Same API subset |
| Communication | Point-to-point + collective | Point-to-point + collective | Identical patterns |

## MPI Function Usage

### Point-to-Point Communication
```c
// Both use identical non-blocking pattern
MPI_Irecv(..., neighbor, tag, comm, &req[0]);
MPI_Isend(..., neighbor, tag, comm, &req[1]);
MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
```

### Collective Communication
```c
MPI_Bcast       // Configuration broadcast
MPI_Scatter     // Metadata distribution
MPI_Scatterv    // Variable-size data distribution
MPI_Gather      // Metadata collection
MPI_Gatherv     // Variable-size data collection
MPI_Reduce      // Performance metrics aggregation
MPI_Barrier     // Synchronization points
```

### File I/O
```c
MPI_File_open          // Parallel file access
MPI_File_read_at       // Positioned read
MPI_File_read_at_all   // Collective read
MPI_File_write_at      // Positioned write
MPI_File_write_at_all  // Collective write
MPI_File_close         // Close file handle
```

### Topology
```c
MPI_Cart_create  // Create Cartesian communicator
MPI_Cart_shift   // Find neighbors
MPI_Comm_rank    // Get rank ID
MPI_Comm_size    // Get total ranks
MPI_Comm_free    // Release communicator
```

### Datatypes
```c
MPI_Type_contiguous  // Create contiguous derived type
MPI_Type_commit      // Register with MPI
MPI_Type_free        // Release type
```

## Communication Patterns

### Halo Exchange (Both)
```
Step 1: Post receives (ready to accept)
   MPI_Irecv(ghost_left,  ..., left_neighbor,  102, ...)
   MPI_Irecv(ghost_right, ..., right_neighbor, 101, ...)

Step 2: Post sends (data available)
   MPI_Isend(boundary_left,  ..., left_neighbor,  101, ...)
   MPI_Isend(boundary_right, ..., right_neighbor, 102, ...)

Step 3: Wait for completion
   MPI_Waitall(4, requests, ...)

Result: Ghost cells populated, safe to compute
```

### Tag Convention
- **101**: Right→Left or Bottom→Top (normal direction)
- **102**: Left→Right or Top→Bottom (reverse direction)
- Symmetric pattern prevents deadlock
- MPI_PROC_NULL safely ignored at boundaries

## Memory Layout

### conv1d_mpi
```
buf: [LEFT_GHOST(H)] [CORE(local_count)] [RIGHT_GHOST(H)]
      ^                ^                   ^
      From left        Original data       From right
```

### conv2d_mpi
```
buf: [TOP_GHOST_ROWS(HH×W)]
     [CORE_ROWS(my_rows×W)]
     [BOTTOM_GHOST_ROWS(HH×W)]
      ^                        ^                          ^
      From top                 Original data              From bottom

Each row: [col0, col1, ..., col(W-1)]  (row-major)
```

## Performance Formulas

### GFLOP/s Calculation
```c
// Per-rank computation
local_flops = 2.0 × kernel_taps × my_output_count
// 2 ops per tap: multiply + add

// Global aggregation
MPI_Reduce(&local_flops, &sum_flops, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
MPI_Reduce(&local_secs, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

// Final metric (uses slowest rank time)
gflops = sum_flops / max_secs / 1e9;
```

### Load Imbalance
```c
MPI_Reduce(&local_secs, &min_secs, ..., MPI_MIN, ...);
imbalance_ratio = max_secs / min_secs;

// Interpretation:
// 1.00 = perfect balance
// 1.05 = 5% imbalance (acceptable)
// 1.10 = 10% imbalance (monitor)
// >1.20 = significant imbalance (investigate)
```

## Scalability Guidelines

### When to Use Each

**conv1d_mpi**:
- Linear signal processing
- Time-series convolution
- 1D filters
- Small kernels (K < 100)
- Any N/P ratio

**conv2d_mpi**:
- Image processing
- 2D filters
- Small-to-moderate width (W < 10000)
- H/P ≥ KH (enough rows per rank)
- Can tolerate fixed width per rank

### Scalability Limits

**conv1d_mpi**:
- Good until: communication time ≈ computation time
- Roughly: P_max ≈ N / (10 × K)
- Example: N=10⁶, K=31 → P_max ≈ 3000 ranks

**conv2d_mpi**:
- Good until: H/P < KH (too few rows per rank)
- Roughly: P_max ≈ H / (2 × KH)
- Example: H=4096, KH=31 → P_max ≈ 65 ranks
- Limited by row decomposition, not width!

## Debugging Checklist

### Before Running
- [ ] Check rank count: P ≤ N (1D) or P ≤ H (2D)
- [ ] Verify input files exist and are readable
- [ ] Ensure output directory writable
- [ ] Confirm kernel size ≤ input size

### If Output Wrong
1. Enable DEBUG_MPI: `-DDEBUG_MPI=1`
2. Check decomposition: verify row/element ranges
3. Test with np=1: should match sequential
4. Verify seed: use -se for determinism
5. Check halo width: H or HH printed in debug

### If Performance Poor
1. Check load imbalance ratio
2. Verify kernel size (too small → comm overhead)
3. Test different rank counts
4. Profile with MPI profiler (mpiP, TAU)
5. Consider stride (reduces work per rank)

## Common Pitfalls

1. **Wrong communicator**: Using MPI_COMM_WORLD instead of cart comm
2. **Missing Barrier**: Before timing or I/O
3. **Wrong offset**: Byte vs element offset in MPI-IO
4. **Unmatched tags**: 101/102 must match between ranks
5. **Deadlock**: Always post receives before sends
6. **Memory**: Allocate halos (not just core data)
7. **2D vs 1D**: conv2d_mpi uses 1D topology!

## Quick Test Commands

```bash
# Compile
mpicc -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c -lm
mpicc -std=c11 -O2 -Wall -Wextra -Werror -o conv2d_mpi conv2d_mpi.c -lm

# Basic test (text)
mpirun -np 4 ./conv1d_mpi -L 1000 -kL 31 -se 42 --text -o out1d.txt
mpirun -np 4 ./conv2d_mpi -H 64 -W 64 -kH 5 -kW 5 -se 42 --text -o out2d.txt

# Binary I/O
mpirun -np 4 ./conv1d_mpi -f input.bin -g kernel.bin -o output.bin
mpirun -np 4 ./conv2d_mpi -f image.bin -g kernel.bin -o output.bin

# Parallel generation
mpirun -np 8 ./conv1d_mpi -L 1000000 -kL 31 --parallel-gen -se 99 -o out.bin
mpirun -np 8 ./conv2d_mpi -H 4096 -W 4096 -kH 11 -kW 11 --parallel-gen -se 99 -o out.bin

# Debug mode
mpicc -DDEBUG_MPI=1 -std=c11 -O2 -o conv1d_mpi_debug conv1d_mpi.c -lm
mpirun -np 4 ./conv1d_mpi_debug -L 100 -kL 5 -se 42 --text -o out.txt 2>debug.log
```

## Summary

Both implementations demonstrate **professional-grade MPI programming**:
- ✅ Clean abstractions (topology, collectives)
- ✅ Robust error handling
- ✅ Scalable I/O patterns
- ✅ Load-balanced decomposition
- ✅ Comprehensive feature set
- ✅ Maintainable code structure

**Key Takeaway**: conv2d_mpi achieves full feature parity with conv1d_mpi while adapting to 2D constraints through row-based decomposition. The 1D topology choice prioritizes implementation simplicity over maximum scalability—an excellent trade-off for educational and moderate-scale production use.
