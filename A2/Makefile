# ============================================================
# Makefile for Assignment 2: MPI + Hybrid MPI/OpenMP Convolution
# ============================================================
#
# Builds all convolution implementations:
#   - Sequential (baseline)
#   - OpenMP (shared memory)
#   - MPI (distributed memory)
#   - MPI+OpenMP Hybrid (distributed + shared memory)
#
# Usage:
#   make              # build all executables
#   make clean        # remove all built binaries
#   make run          # build + run ALL programs with defaults
#
# Individual builds:
#   make seq          # sequential only
#   make omp          # OpenMP only
#   make mpi          # MPI only
#   make hybrid       # MPI+OpenMP only
#
# Individual runs:
#   make run1d        # sequential 1D
#   make run1d_omp    # OpenMP 1D
#   make run1d_mpi    # MPI 1D
#   make run1d_hybrid # MPI+OpenMP 1D
#   make run2d        # sequential 2D
#   make run2d_omp    # OpenMP 2D
#   make run2d_mpi    # MPI 2D (2D Cartesian topology)
#   make run2d_hybrid # MPI+OpenMP 2D (2D Cartesian topology)
#
# Parameter overrides (examples):
#   make run1d N=2000000 K=5001
#   make run1d_omp N=1000000 K=10001 THREADS=16 SCHED=dynamic
#   make run1d_mpi N=10000000 K=1001 NP=8
#   make run1d_hybrid N=20000000 K=1001 NP=4 THREADS=4
#   make run2d H=2048 W=2048 KH=7 KW=7
#   make run2d_mpi H=4096 W=4096 KH=5 KW=5 NP=9
#   make run2d_hybrid H=4096 W=4096 KH=7 KW=7 NP=4 THREADS=4
#
# Notes:
# - Sequential/OMP builds use: cc -O3 -march=native -funroll-loops
# - MPI builds use: mpicc -O3 -march=native -funroll-loops
# - Hybrid builds add: -fopenmp
# - On Kaya, load modules first:
#     module load gcc/14.2
#     module load openmpi/5.0.5
# - MPI execution uses mpiexec (not mpirun)
#
# ============================================================

# Compilers
CC      ?= cc
MPICC   ?= mpicc
LIBS    ?= -lm

# Compiler flags (Kaya O3 style)
COMMON_FLAGS = -std=c11 -O3 -march=native -funroll-loops -Wall -Wextra
OMPFLAG      = -fopenmp
VECFLAG      = -fopt-info-vec  # Optional: shows vectorization info

# Sources
SRC_CONV1D          = conv1d.c
SRC_CONV1D_OMP      = conv1d_omp.c
SRC_CONV1D_MPI      = conv1d_mpi.c
SRC_CONV1D_MPI_OMP  = conv1d_mpi_omp.c

SRC_CONV2D          = conv2d.c
SRC_CONV2D_OMP      = conv2d_omp.c
SRC_CONV2D_MPI      = conv2d_mpi.c
SRC_CONV2D_MPI_OMP  = conv2d_mpi_omp.c

# Binaries
BIN_CONV1D          = conv1d
BIN_CONV1D_OMP      = conv1d_omp
BIN_CONV1D_MPI      = conv1d_mpi
BIN_CONV1D_MPI_OMP  = conv1d_mpi_omp

BIN_CONV2D          = conv2d
BIN_CONV2D_OMP      = conv2d_omp
BIN_CONV2D_MPI      = conv2d_mpi
BIN_CONV2D_MPI_OMP  = conv2d_mpi_omp

# Groups
BINS_SEQ    = $(BIN_CONV1D) $(BIN_CONV2D)
BINS_OMP    = $(BIN_CONV1D_OMP) $(BIN_CONV2D_OMP)
BINS_MPI    = $(BIN_CONV1D_MPI) $(BIN_CONV2D_MPI)
BINS_HYBRID = $(BIN_CONV1D_MPI_OMP) $(BIN_CONV2D_MPI_OMP)
ALL_BINS    = $(BINS_SEQ) $(BINS_OMP) $(BINS_MPI) $(BINS_HYBRID)

.PHONY: all seq omp mpi hybrid clean \
        run run1d run1d_omp run1d_mpi run1d_hybrid \
        run2d run2d_omp run2d_mpi run2d_hybrid

# Default target: build everything
all: $(ALL_BINS)

# Group targets
seq: $(BINS_SEQ)
omp: $(BINS_OMP)
mpi: $(BINS_MPI)
hybrid: $(BINS_HYBRID)

# ============================================================
# Build Rules
# ============================================================

# ----------- Sequential (1D & 2D) -----------
$(BIN_CONV1D): $(SRC_CONV1D)
	$(CC) $(COMMON_FLAGS) -o $@ $< $(LIBS)

$(BIN_CONV2D): $(SRC_CONV2D)
	$(CC) $(COMMON_FLAGS) -o $@ $< $(LIBS)

# ----------- OpenMP (1D & 2D) -----------
$(BIN_CONV1D_OMP): $(SRC_CONV1D_OMP)
	$(CC) $(COMMON_FLAGS) $(OMPFLAG) -o $@ $< $(LIBS)

$(BIN_CONV2D_OMP): $(SRC_CONV2D_OMP)
	$(CC) $(COMMON_FLAGS) $(OMPFLAG) -o $@ $< $(LIBS)

# ----------- MPI (1D & 2D) -----------
$(BIN_CONV1D_MPI): $(SRC_CONV1D_MPI)
	$(MPICC) $(COMMON_FLAGS) -o $@ $< $(LIBS)

$(BIN_CONV2D_MPI): $(SRC_CONV2D_MPI)
	$(MPICC) $(COMMON_FLAGS) -o $@ $< $(LIBS)

# ----------- MPI+OpenMP Hybrid (1D & 2D) -----------
$(BIN_CONV1D_MPI_OMP): $(SRC_CONV1D_MPI_OMP)
	$(MPICC) $(COMMON_FLAGS) $(OMPFLAG) -o $@ $< $(LIBS)

$(BIN_CONV2D_MPI_OMP): $(SRC_CONV2D_MPI_OMP)
	$(MPICC) $(COMMON_FLAGS) $(OMPFLAG) -o $@ $< $(LIBS)

# ----------- Clean -----------
clean:
	$(RM) $(ALL_BINS)
	$(RM) -rf metrics logs results

# ============================================================
# Run Targets
# ============================================================

# Default parameters (override with make VAR=val)
N      ?= 1000000    # 1D input length
K      ?= 10001      # 1D kernel length
SEED   ?= 42         # RNG seed
OUT1D  ?= /dev/null

H      ?= 1024       # 2D height
W      ?= 1024       # 2D width
KH     ?= 5          # 2D kernel height
KW     ?= 5          # 2D kernel width
OUT2D  ?= /dev/null

# OpenMP parameters
THREADS ?= 8         # OpenMP threads
SCHED   ?= static    # OpenMP schedule (static|dynamic|guided|auto)
CHUNK   ?=           # OpenMP chunk size (optional)

# MPI parameters
NP      ?= 4         # Number of MPI processes
MPIEXEC ?= mpiexec   # MPI launcher (mpiexec or mpirun)

# Run all implementations
run: run1d run1d_omp run1d_mpi run1d_hybrid run2d run2d_omp run2d_mpi run2d_hybrid

# ============================================================
# 1D Convolution Runs
# ============================================================

# Sequential 1D
run1d: $(BIN_CONV1D)
	@echo "==> Running $(BIN_CONV1D) [Sequential 1D]"
	@echo "    N=$(N), K=$(K), SEED=$(SEED)"
	./$(BIN_CONV1D) -L $(N) -kL $(K) -o $(OUT1D) -s $(SEED)

# OpenMP 1D
run1d_omp: $(BIN_CONV1D_OMP)
	@echo "==> Running $(BIN_CONV1D_OMP) [OpenMP 1D]"
	@echo "    N=$(N), K=$(K), THREADS=$(THREADS), SCHED=$(SCHED)$(if $(CHUNK),:$(CHUNK),), SEED=$(SEED)"
	@OMP_NUM_THREADS=$(THREADS) OMP_PROC_BIND=spread OMP_PLACES=cores \
	$(if $(filter auto,$(SCHED)),OMP_SCHEDULE=auto,OMP_SCHEDULE=$(SCHED)$(if $(CHUNK),:$(CHUNK),)) \
	./$(BIN_CONV1D_OMP) -L $(N) -kL $(K) -o $(OUT1D) --threads $(THREADS) --schedule $(SCHED) \
	$(if $(CHUNK),--chunk $(CHUNK),) -s $(SEED)

# MPI 1D
run1d_mpi: $(BIN_CONV1D_MPI)
	@echo "==> Running $(BIN_CONV1D_MPI) [Pure MPI 1D]"
	@echo "    N=$(N), K=$(K), NP=$(NP), SEED=$(SEED)"
	$(MPIEXEC) -n $(NP) ./$(BIN_CONV1D_MPI) -L $(N) -kL $(K) -o $(OUT1D) -s $(SEED)

# MPI+OpenMP Hybrid 1D
run1d_hybrid: $(BIN_CONV1D_MPI_OMP)
	@echo "==> Running $(BIN_CONV1D_MPI_OMP) [Hybrid MPI+OpenMP 1D]"
	@echo "    N=$(N), K=$(K), NP=$(NP), THREADS=$(THREADS), SCHED=$(SCHED)$(if $(CHUNK),:$(CHUNK),), SEED=$(SEED)"
	@OMP_NUM_THREADS=$(THREADS) OMP_PROC_BIND=spread OMP_PLACES=cores \
	$(if $(filter auto,$(SCHED)),OMP_SCHEDULE=auto,OMP_SCHEDULE=$(SCHED)$(if $(CHUNK),:$(CHUNK),)) \
	$(MPIEXEC) -n $(NP) ./$(BIN_CONV1D_MPI_OMP) -L $(N) -kL $(K) -o $(OUT1D) \
	--threads $(THREADS) --schedule $(SCHED) $(if $(CHUNK),--chunk $(CHUNK),) -s $(SEED)

# ============================================================
# 2D Convolution Runs
# ============================================================

# Sequential 2D
run2d: $(BIN_CONV2D)
	@echo "==> Running $(BIN_CONV2D) [Sequential 2D]"
	@echo "    H=$(H), W=$(W), KH=$(KH), KW=$(KW), SEED=$(SEED)"
	./$(BIN_CONV2D) -H $(H) -W $(W) -kH $(KH) -kW $(KW) -o $(OUT2D) -s $(SEED)

# OpenMP 2D
run2d_omp: $(BIN_CONV2D_OMP)
	@echo "==> Running $(BIN_CONV2D_OMP) [OpenMP 2D]"
	@echo "    H=$(H), W=$(W), KH=$(KH), KW=$(KW), THREADS=$(THREADS), SCHED=$(SCHED)$(if $(CHUNK),:$(CHUNK),), SEED=$(SEED)"
	@OMP_NUM_THREADS=$(THREADS) OMP_PROC_BIND=spread OMP_PLACES=cores \
	$(if $(filter auto,$(SCHED)),OMP_SCHEDULE=auto,OMP_SCHEDULE=$(SCHED)$(if $(CHUNK),:$(CHUNK),)) \
	./$(BIN_CONV2D_OMP) -H $(H) -W $(W) -kH $(KH) -kW $(KW) -o $(OUT2D) \
	--threads $(THREADS) --schedule $(SCHED) $(if $(CHUNK),--chunk $(CHUNK),) --seed $(SEED)

# MPI 2D (2D Cartesian Topology)
run2d_mpi: $(BIN_CONV2D_MPI)
	@echo "==> Running $(BIN_CONV2D_MPI) [Pure MPI 2D with 2D Cartesian Topology]"
	@echo "    H=$(H), W=$(W), KH=$(KH), KW=$(KW), NP=$(NP), SEED=$(SEED)"
	@echo "    (NP=$(NP) will auto-decompose into P_rows × P_cols grid)"
	$(MPIEXEC) -n $(NP) ./$(BIN_CONV2D_MPI) -H $(H) -W $(W) -kH $(KH) -kW $(KW) -o $(OUT2D) -s $(SEED)

# MPI+OpenMP Hybrid 2D (2D Cartesian Topology)
run2d_hybrid: $(BIN_CONV2D_MPI_OMP)
	@echo "==> Running $(BIN_CONV2D_MPI_OMP) [Hybrid MPI+OpenMP 2D with 2D Cartesian Topology]"
	@echo "    H=$(H), W=$(W), KH=$(KH), KW=$(KW), NP=$(NP), THREADS=$(THREADS), SCHED=$(SCHED)$(if $(CHUNK),:$(CHUNK),), SEED=$(SEED)"
	@echo "    Total cores: $(NP) MPI processes × $(THREADS) OpenMP threads = $(shell echo $$(($(NP) * $(THREADS))))"
	@OMP_NUM_THREADS=$(THREADS) OMP_PROC_BIND=spread OMP_PLACES=cores \
	$(if $(filter auto,$(SCHED)),OMP_SCHEDULE=auto,OMP_SCHEDULE=$(SCHED)$(if $(CHUNK),:$(CHUNK),)) \
	$(MPIEXEC) -n $(NP) ./$(BIN_CONV2D_MPI_OMP) -H $(H) -W $(W) -kH $(KH) -kW $(KW) -o $(OUT2D) \
	--threads $(THREADS) --schedule $(SCHED) $(if $(CHUNK),--chunk $(CHUNK),) --seed $(SEED)

# ============================================================
# Helper Targets
# ============================================================

.PHONY: help test dirs

help:
	@echo "Assignment 2 Makefile - MPI + Hybrid Convolution"
	@echo ""
	@echo "Build targets:"
	@echo "  make          - Build all executables"
	@echo "  make seq      - Build sequential only"
	@echo "  make omp      - Build OpenMP only"
	@echo "  make mpi      - Build MPI only"
	@echo "  make hybrid   - Build MPI+OpenMP only"
	@echo "  make clean    - Remove all binaries"
	@echo ""
	@echo "Run targets:"
	@echo "  make run1d         - Run sequential 1D"
	@echo "  make run1d_omp     - Run OpenMP 1D"
	@echo "  make run1d_mpi     - Run MPI 1D"
	@echo "  make run1d_hybrid  - Run hybrid 1D"
	@echo "  make run2d         - Run sequential 2D"
	@echo "  make run2d_omp     - Run OpenMP 2D"
	@echo "  make run2d_mpi     - Run MPI 2D (2D Cartesian)"
	@echo "  make run2d_hybrid  - Run hybrid 2D (2D Cartesian)"
	@echo "  make run           - Run all above"
	@echo ""
	@echo "Parameter overrides:"
	@echo "  N, K              - 1D problem size (default: N=1000000, K=10001)"
	@echo "  H, W, KH, KW      - 2D problem size (default: H=W=1024, KH=KW=5)"
	@echo "  NP                - MPI processes (default: 4)"
	@echo "  THREADS           - OpenMP threads (default: 8)"
	@echo "  SCHED             - OpenMP schedule (default: static)"
	@echo "  CHUNK             - OpenMP chunk size (optional)"
	@echo "  SEED              - RNG seed (default: 42)"
	@echo ""
	@echo "Examples:"
	@echo "  make run2d_mpi H=2048 W=2048 KH=5 KW=5 NP=9"
	@echo "  make run2d_hybrid H=4096 W=4096 KH=7 KW=7 NP=4 THREADS=4"
	@echo "  make run1d_hybrid N=20000000 K=1001 NP=8 THREADS=2 SCHED=dynamic"

# Test target: quick verification
test: all
	@echo "=== Quick Test Suite ==="
	@echo ""
	@echo "1. Sequential 1D (small problem)"
	@./$(BIN_CONV1D) -L 1000 -kL 11 -o /dev/null -s 42
	@echo ""
	@echo "2. OpenMP 1D (2 threads)"
	@OMP_NUM_THREADS=2 ./$(BIN_CONV1D_OMP) -L 1000 -kL 11 -o /dev/null --threads 2 -s 42
	@echo ""
	@echo "3. MPI 1D (2 processes)"
	@$(MPIEXEC) -n 2 ./$(BIN_CONV1D_MPI) -L 1000 -kL 11 -o /dev/null -s 42
	@echo ""
	@echo "4. Hybrid 1D (2 MPI × 2 OMP)"
	@OMP_NUM_THREADS=2 $(MPIEXEC) -n 2 ./$(BIN_CONV1D_MPI_OMP) -L 1000 -kL 11 -o /dev/null --threads 2 -s 42
	@echo ""
	@echo "5. Sequential 2D (small problem)"
	@./$(BIN_CONV2D) -H 128 -W 128 -kH 3 -kW 3 -o /dev/null -s 42
	@echo ""
	@echo "6. OpenMP 2D (2 threads)"
	@OMP_NUM_THREADS=2 ./$(BIN_CONV2D_OMP) -H 128 -W 128 -kH 3 -kW 3 -o /dev/null --threads 2 --seed 42
	@echo ""
	@echo "7. MPI 2D (4 processes → 2×2 grid)"
	@$(MPIEXEC) -n 4 ./$(BIN_CONV2D_MPI) -H 128 -W 128 -kH 3 -kW 3 -o /dev/null -s 42
	@echo ""
	@echo "8. Hybrid 2D (4 MPI × 2 OMP)"
	@OMP_NUM_THREADS=2 $(MPIEXEC) -n 4 ./$(BIN_CONV2D_MPI_OMP) -H 128 -W 128 -kH 3 -kW 3 -o /dev/null --threads 2 --seed 42
	@echo ""
	@echo "✓ All tests completed!"

# Create necessary directories
dirs:
	mkdir -p logs metrics results
