# Design Document: 2D Cartesian Topology for conv2d_mpi.c

## Overview
This document provides detailed pseudocode and design for upgrading `conv2d_mpi.c` from 1D row-based decomposition to 2D block-based decomposition using a 2D Cartesian topology.

## 1. Topology Creation

### Current (1D):
```c
int dims[1] = {world_size};
int periods[1] = {0};
MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &comm);
MPI_Cart_shift(comm, 0, 1, &top, &bottom);
```

### New (2D):
```c
// Let MPI decide optimal 2D grid dimensions
int dims[2] = {0, 0};  // [P_rows, P_cols]
MPI_Dims_create(world_size, 2, dims);

int periods[2] = {0, 0};  // No wraparound
int reorder = 1;  // Allow reordering for better performance
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm);

// Get my coordinates in the grid
int coords[2];  // [my_row, my_col]
MPI_Cart_coords(comm, rank, 2, coords);

// Find 4 neighbors (MPI_PROC_NULL at boundaries)
int north, south, east, west;
MPI_Cart_shift(comm, 0, 1, &north, &south);  // dim 0: vertical
MPI_Cart_shift(comm, 1, 1, &west, &east);     // dim 1: horizontal
```

## 2. Domain Decomposition (2D Block)

### Concept:
```
Global Image (H × W):
┌─────────┬─────────┬─────────┐
│ (0,0)   │ (0,1)   │ (0,2)   │  P_rows = 3
│ Block   │ Block   │ Block   │  P_cols = 3
├─────────┼─────────┼─────────┤  Total ranks = 9
│ (1,0)   │ (1,1)   │ (1,2)   │
│ Block   │ Block   │ Block   │
├─────────┼─────────┼─────────┤
│ (2,0)   │ (2,1)   │ (2,2)   │
│ Block   │ Block   │ Block   │
└─────────┴─────────┴─────────┘

Each block: (my_block_H × my_block_W)
```

### Pseudocode:
```c
int P_rows = dims[0];
int P_cols = dims[1];
int my_row_idx = coords[0];
int my_col_idx = coords[1];

// Vertical decomposition (rows)
int base_rows = H / P_rows;
int rem_rows = H % P_rows;
int my_block_H = base_rows + (my_row_idx < rem_rows ? 1 : 0);

// Calculate my starting row in global image
int my_H_start = 0;
for (int r = 0; r < my_row_idx; r++) {
    int rows_for_r = base_rows + (r < rem_rows ? 1 : 0);
    my_H_start += rows_for_r;
}

// Horizontal decomposition (columns)
int base_cols = W / P_cols;
int rem_cols = W % P_cols;
int my_block_W = base_cols + (my_col_idx < rem_cols ? 1 : 0);

// Calculate my starting column in global image
int my_W_start = 0;
for (int c = 0; c < my_col_idx; c++) {
    int cols_for_c = base_cols + (c < rem_cols ? 1 : 0);
    my_W_start += cols_for_c;
}

// My block: global_image[my_H_start : my_H_start+my_block_H]
//                       [my_W_start : my_W_start+my_block_W]
```

## 3. Halo Exchange (4 Directions)

### Buffer Layout:
```
         West Halo (HW cols)
              ↓
    ┌────┬─────────────┬────┐
    │ NW │    North    │ NE │ ← North Halo (HH rows)
    ├────┼─────────────┼────┤
    │    │             │    │
    │ W  │  MY DATA    │ E  │
    │    │             │    │
    ├────┼─────────────┼────┤
    │ SW │    South    │ SE │ ← South Halo (HH rows)
    └────┴─────────────┴────┘
         ↑
    East Halo (HW cols)

HH = KH - 1  (vertical halo width)
HW = KW - 1  (horizontal halo width)
```

### Strategy 1: 8-Message Pattern (Simplest)
```c
const int HH = KH - 1;  // vertical halo
const int HW = KW - 1;  // horizontal halo

// Allocate buffer with halos
int buf_H = my_block_H + 2*HH;
int buf_W = my_block_W + 2*HW;
float *buf = calloc(buf_H * buf_W, sizeof(float));

// Copy local data to buffer center
for (int i = 0; i < my_block_H; i++) {
    for (int j = 0; j < my_block_W; j++) {
        buf[IDX2(i+HH, j+HW, buf_W)] = F_local[IDX2(i, j, my_block_W)];
    }
}

MPI_Request reqs[8];
int req_idx = 0;

// 1. North/South (contiguous rows)
if (north != MPI_PROC_NULL) {
    // Recv north halo
    MPI_Irecv(&buf[IDX2(0, HW, buf_W)], 
              HH * my_block_W, MPI_FLOAT, north, TAG_SOUTH, comm, &reqs[req_idx++]);
    // Send my top boundary
    MPI_Isend(&F_local[IDX2(0, 0, my_block_W)], 
              HH * my_block_W, MPI_FLOAT, north, TAG_NORTH, comm, &reqs[req_idx++]);
}

if (south != MPI_PROC_NULL) {
    // Recv south halo
    MPI_Irecv(&buf[IDX2(HH + my_block_H, HW, buf_W)], 
              HH * my_block_W, MPI_FLOAT, south, TAG_NORTH, comm, &reqs[req_idx++]);
    // Send my bottom boundary
    int send_start = my_block_H - HH;
    MPI_Isend(&F_local[IDX2(send_start, 0, my_block_W)], 
              HH * my_block_W, MPI_FLOAT, south, TAG_SOUTH, comm, &reqs[req_idx++]);
}

// 2. East/West (non-contiguous columns - pack/unpack)
float *west_send = malloc(HW * my_block_H * sizeof(float));
float *west_recv = malloc(HW * buf_H * sizeof(float));
float *east_send = malloc(HW * my_block_H * sizeof(float));
float *east_recv = malloc(HW * buf_H * sizeof(float));

// Pack west boundary
for (int i = 0; i < my_block_H; i++) {
    for (int j = 0; j < HW; j++) {
        west_send[i*HW + j] = F_local[IDX2(i, j, my_block_W)];
    }
}

// Pack east boundary
for (int i = 0; i < my_block_H; i++) {
    for (int j = 0; j < HW; j++) {
        east_send[i*HW + j] = F_local[IDX2(i, my_block_W - HW + j, my_block_W)];
    }
}

if (west != MPI_PROC_NULL) {
    MPI_Irecv(west_recv, HW * buf_H, MPI_FLOAT, west, TAG_EAST, comm, &reqs[req_idx++]);
    MPI_Isend(west_send, HW * my_block_H, MPI_FLOAT, west, TAG_WEST, comm, &reqs[req_idx++]);
}

if (east != MPI_PROC_NULL) {
    MPI_Irecv(east_recv, HW * buf_H, MPI_FLOAT, east, TAG_WEST, comm, &reqs[req_idx++]);
    MPI_Isend(east_send, HW * my_block_H, MPI_FLOAT, east, TAG_EAST, comm, &reqs[req_idx++]);
}

MPI_Waitall(req_idx, reqs, MPI_STATUSES_IGNORE);

// Unpack west halo
if (west != MPI_PROC_NULL) {
    for (int i = 0; i < buf_H; i++) {
        for (int j = 0; j < HW; j++) {
            buf[IDX2(i, j, buf_W)] = west_recv[i*HW + j];
        }
    }
}

// Unpack east halo
if (east != MPI_PROC_NULL) {
    for (int i = 0; i < buf_H; i++) {
        for (int j = 0; j < HW; j++) {
            buf[IDX2(i, HW + my_block_W + j, buf_W)] = east_recv[i*HW + j];
        }
    }
}

free(west_send); free(west_recv);
free(east_send); free(east_recv);
```

### Strategy 2: MPI_Type_vector (Elegant, No Packing)
```c
// Create datatype for vertical column slice
MPI_Datatype col_type;
MPI_Type_vector(
    my_block_H,     // count: number of blocks
    HW,             // blocklength: elements per block
    my_block_W,     // stride: distance between blocks
    MPI_FLOAT,      // oldtype
    &col_type
);
MPI_Type_commit(&col_type);

if (west != MPI_PROC_NULL) {
    // Recv west halo
    MPI_Irecv(&buf[IDX2(HH, 0, buf_W)], ...);
    // Send west boundary (using col_type for non-contiguous)
    MPI_Isend(&F_local[IDX2(0, 0, my_block_W)], 1, col_type, west, TAG_WEST, comm, ...);
}

// Similar for east
MPI_Type_free(&col_type);
```

## 4. Parallel Generation (2D Indexed)

### Pseudocode:
```c
if (parallel_gen) {
    // Each rank generates its block using global indices
    for (int li = 0; li < my_block_H; li++) {
        for (int lj = 0; lj < my_block_W; lj++) {
            long long gi = my_H_start + li;  // global row
            long long gj = my_W_start + lj;  // global col
            F_local[IDX2(li, lj, my_block_W)] = gen_value_at_index_2d(seed, gi, gj);
        }
    }
}
```

## 5. MPI-IO Binary Read (2D Subarray)

### Concept:
```
Global File (H × W):
┌─────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░ │  Each rank reads its block
│ ░░░░░░░░░░░░░░░░░░░ │  Non-contiguous in file!
│ ░░██████░░░░░░░░░░░ │  
│ ░░██████░░░░░░░░░░░ │  ██ = rank (1,1)'s block
│ ░░██████░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░ │
└─────────────────────┘
```

### Pseudocode:
```c
if (img_path && f_is_bin) {
    MPI_File fh;
    MPI_File_open(comm, img_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    
    // Create subarray datatype for my block in global file
    int global_sizes[2] = {H, W};
    int local_sizes[2] = {my_block_H, my_block_W};
    int starts[2] = {my_H_start, my_W_start};
    
    MPI_Datatype file_type;
    MPI_Type_create_subarray(
        2,                      // ndims
        global_sizes,           // array_of_sizes
        local_sizes,            // array_of_subsizes
        starts,                 // array_of_starts
        MPI_ORDER_C,            // order (row-major)
        MPI_FLOAT,              // oldtype
        &file_type
    );
    MPI_Type_commit(&file_type);
    
    // Set file view (skip 8-byte header)
    MPI_Offset header_offset = 2 * sizeof(int);
    MPI_File_set_view(fh, header_offset, MPI_FLOAT, file_type, "native", MPI_INFO_NULL);
    
    // Collective read
    MPI_File_read_all(fh, F_local, my_block_H * my_block_W, MPI_FLOAT, MPI_STATUS_IGNORE);
    
    MPI_Type_free(&file_type);
    MPI_File_close(&fh);
}
```

## 6. Scatter from Root (2D Manual)

### Pseudocode:
```c
if (!parallel_gen && !f_is_bin) {
    if (rank == 0) {
        // Root scatters blocks to all ranks
        for (int dest_rank = 0; dest_rank < size; dest_rank++) {
            // Calculate dest rank's coordinates
            int dest_coords[2];
            MPI_Cart_coords(comm, dest_rank, 2, dest_coords);
            
            // Calculate dest's block info
            int dest_H_start = ...;  // (same logic as above)
            int dest_W_start = ...;
            int dest_block_H = ...;
            int dest_block_W = ...;
            
            // Pack block from F_root
            float *block = malloc(dest_block_H * dest_block_W * sizeof(float));
            for (int i = 0; i < dest_block_H; i++) {
                for (int j = 0; j < dest_block_W; j++) {
                    block[i * dest_block_W + j] = 
                        F_root[(dest_H_start + i) * W + (dest_W_start + j)];
                }
            }
            
            if (dest_rank == 0) {
                memcpy(F_local, block, dest_block_H * dest_block_W * sizeof(float));
            } else {
                MPI_Send(block, dest_block_H * dest_block_W, MPI_FLOAT, 
                         dest_rank, TAG_SCATTER, comm);
            }
            free(block);
        }
    } else {
        // Non-root receives
        MPI_Recv(F_local, my_block_H * my_block_W, MPI_FLOAT, 
                 0, TAG_SCATTER, comm, MPI_STATUS_IGNORE);
    }
}
```

## 7. Output Decomposition (2D Blocks)

### Concept:
Each rank computes a 2D block of output, then writes to file using subarray.

### Pseudocode:
```c
// Calculate output dimensions for my block
int my_out_H = calculate_output_height(my_block_H, KH, cmode, sH);
int my_out_W = calculate_output_width(my_block_W, KW, cmode, sW);

// Calculate my starting position in global output
int global_out_H = calculate_output_height(H, KH, cmode, sH);
int global_out_W = calculate_output_width(W, KW, cmode, sW);

// Calculate offset based on my grid position
int my_out_H_start = ...;  // Sum of output heights of ranks above me
int my_out_W_start = ...;  // Sum of output widths of ranks to my left

// Allocate local output
float *Y_local = malloc(my_out_H * my_out_W * sizeof(float));

// Compute convolution on my block (with halos)
perform_convolution_2d(buf, G_use, Y_local, 
                       my_block_H, my_block_W, buf_W, 
                       KH, KW, cmode, sH, sW, pmode, cval);

// Write using MPI-IO subarray
MPI_File fh_out;
MPI_File_open(comm, out_path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_out);

int out_global_sizes[2] = {global_out_H, global_out_W};
int out_local_sizes[2] = {my_out_H, my_out_W};
int out_starts[2] = {my_out_H_start, my_out_W_start};

MPI_Datatype out_file_type;
MPI_Type_create_subarray(2, out_global_sizes, out_local_sizes, out_starts,
                         MPI_ORDER_C, MPI_FLOAT, &out_file_type);
MPI_Type_commit(&out_file_type);

// Root writes header
if (rank == 0) {
    int hdr[2] = {global_out_H, global_out_W};
    MPI_File_write_at(fh_out, 0, hdr, 2, MPI_INT, MPI_STATUS_IGNORE);
}
MPI_Barrier(comm);

// All ranks write their blocks
MPI_Offset header_size = 2 * sizeof(int);
MPI_File_set_view(fh_out, header_size, MPI_FLOAT, out_file_type, "native", MPI_INFO_NULL);
MPI_File_write_all(fh_out, Y_local, my_out_H * my_out_W, MPI_FLOAT, MPI_STATUS_IGNORE);

MPI_Type_free(&out_file_type);
MPI_File_close(&fh_out);
```

## 8. Debug Output

### Recommended Debug Prints:
```c
#ifdef DEBUG_MPI
if (rank == 0) {
    printf("2D Cartesian Grid: %d × %d (P_rows × P_cols)\n", P_rows, P_cols);
}
printf("[%d] coords=(%d,%d) block=[%d:%d, %d:%d] size=%dx%d neighbors=(N:%d S:%d E:%d W:%d)\n",
       rank, my_row_idx, my_col_idx,
       my_H_start, my_H_start + my_block_H - 1,
       my_W_start, my_W_start + my_block_W - 1,
       my_block_H, my_block_W,
       north, south, east, west);
#endif
```

## 9. Testing Strategy

### Test Cases:
1. **Small square**: H=8, W=8, KH=3, KW=3, np=4 (2×2 grid)
2. **Small rect**: H=8, W=12, KH=3, KW=3, np=6 (2×3 grid)
3. **Uneven div**: H=10, W=10, KH=3, KW=3, np=4 (2×2 with remainders)
4. **Single row**: H=4, W=16, KH=3, KW=3, np=4 (1×4 grid)
5. **Single col**: H=16, W=4, KH=3, KW=3, np=4 (4×1 grid)

### Validation:
```bash
# Compare against sequential
./conv2d -H 8 -W 8 -kH 3 -kW 3 -se 42 --text -o seq.txt
mpirun -np 4 ./conv2d_mpi -H 8 -W 8 -kH 3 -kW 3 -se 42 --text -o mpi.txt
diff seq.txt mpi.txt
```

## 10. Performance Considerations

### Communication Volume:
- **1D Topology**: 2 × (HH × W) floats per exchange
- **2D Topology**: 2 × (HH × my_block_W + HW × my_block_H) floats per exchange

### When 2D is Better:
- Large W (reduces column halo size: HW × my_block_H << HW × H)
- Square images (balanced decomposition)
- Many ranks (P > H with 1D topology)

### When 1D is Better:
- Small W (less overhead)
- Tall images (H >> W)
- Simple debugging

## 11. Implementation Checklist

- [ ] Update topology creation (2D Cart_create)
- [ ] Add coordinate extraction (Cart_coords)
- [ ] Implement 2D block decomposition
- [ ] Update halo exchange (4 directions + packing)
- [ ] Fix parallel generation (2D indexing)
- [ ] Update MPI-IO binary read (subarray)
- [ ] Update scatter logic (2D manual)
- [ ] Update gather logic (2D manual)
- [ ] Fix output decomposition (2D blocks)
- [ ] Update output MPI-IO (subarray)
- [ ] Add extensive debug prints
- [ ] Test with multiple rank configurations
- [ ] Verify against sequential version

## 12. Common Pitfalls

1. **Off-by-one** in block size calculation with remainders
2. **Wrong stride** in MPI_Type_vector (use my_block_W, not W)
3. **Buffer overflow** in halo allocation (use buf_H, buf_W consistently)
4. **Deadlock** in halo exchange (always post receives before sends)
5. **MPI-IO view** confusion (set_view uses file_type, not element count)
6. **Coordinate confusion** (coords[0] = row, coords[1] = col)
7. **Tag mismatch** in east/west exchange

This design provides the complete blueprint for implementation!
