# 2D Cartesian Topology Test Results

## Test Date
October 18, 2025

## Test Configuration
- Problem: 8×8 matrix, 3×3 kernel
- Seed: 42
- Mode: SAME (stride=1×1)
- Validation: Compare MPI output against sequential version

## Results Summary

### ✅ SAME Mode Tests (All PASS)
| Ranks | Grid Layout | Status | Notes |
|-------|-------------|--------|-------|
| np=1  | 1×1         | ✓ PASS | Single rank (no communication) |
| np=2  | 2×1         | ✓ PASS | Vertical split only |
| np=4  | 2×2         | ✓ PASS | Full 2D grid |
| np=6  | 3×2         | ✓ PASS | Uneven grid dimensions |
| np=9  | 3×3         | ✓ PASS | Symmetric 3×3 grid |

### Additional Tests
- ✅ Larger problem (16×16, np=4): PASS
- ✅ Remainder distribution: PASS (tested with uneven splits)
- ✅ Edge cases: Small blocks (2×2, 3×3) work correctly

### ⚠️ Known Issues
- FULL mode: West halo not correctly populated (columns 0-1 show zeros)
  - SAME mode works perfectly
  - FULL mode needs debugging of halo exchange logic

## Key Improvements from 1D to 2D Topology
1. **True 2D decomposition**: Both H and W partitioned (was H-only before)
2. **Better load balance**: More flexible grid shapes (e.g., 3×2, 3×3)
3. **Scalability**: Supports more rank configurations
4. **Halo exchange**: Fixed per-row send/recv to handle buffer stride

## Critical Bug Fixed
**Issue**: Halo exchange was receiving multiple rows as contiguous block, ignoring buffer stride.  
**Solution**: Send/receive each halo row separately to respect `buf_W` stride in the buffer.

## Files Modified
- `conv2d_mpi.c`: ~450 lines changed
- `conv2d_mpi_1d_topology.c`: Backup of original 1D version
- `CONV2D_MPI_2D_DESIGN.md`: Design document (500+ lines)

