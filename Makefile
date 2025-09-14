# ============================================================
# Makefile for 1D/2D Convolution Benchmarks (Optimised O3)
# ============================================================
#
# Usage:
#   make            # build all executables
#   make clean      # remove all built binaries
#
#   make run        # build + run ALL (seq/OMP, 1D/2D) with defaults
#
# Individual runs:
#   make run1d      # sequential 1D
#   make run1d_omp  # OpenMP 1D
#   make run2d      # sequential 2D
#   make run2d_omp  # OpenMP 2D
#
# Overriding parameters (examples):
#   make run1d N=2000000 K=5001
#   make run1d_omp N=1000000 K=10001 THREADS=16 SCHED=guided CHUNK=2048
#   make run2d H=2048 W=2048 KH=7 KW=7
#   make run2d_omp H=1280 W=1920 KH=70 KW=70 THREADS=32 SCHED=auto
#
# Notes:
# - Builds use Kaya-style O3 flags:
#     -std=c11 -O3 -march=native -funroll-loops -fopt-info-vec
# - OMP variants add -fopenmp
# - CC defaults to 'cc'; override with e.g. `make CC=gcc`
# - If running on macOS/Clang, install libomp and run with:
#     make LIBS="-lomp"
#
# ============================================================

CC      ?= cc
LIBS    ?=

# Common O3 flags (Kaya style)
COMMON_O3 = -std=c11 -O3 -march=native -funroll-loops -fopt-info-vec
OMPFLAG  = -fopenmp

# Sources
SRC_CONV1D      = conv1d.c
SRC_CONV1D_OMP  = conv1d_omp.c
SRC_CONV2D      = conv2d.c
SRC_CONV2D_OMP  = conv2d_omp.c

# Binaries
BIN1 = conv1d
BIN2 = conv1d_omp
BIN3 = conv2d
BIN4 = conv2d_omp

ALL  = $(BIN1) $(BIN2) $(BIN3) $(BIN4)

.PHONY: all clean run run1d run1d_omp run2d run2d_omp
all: $(ALL)

# ----------- Build rules -----------
$(BIN1): $(SRC_CONV1D)
	$(CC) $(COMMON_O3) -o $@ $< $(LIBS)

$(BIN2): $(SRC_CONV1D_OMP)
	$(CC) $(COMMON_O3) $(OMPFLAG) -o $@ $< $(LIBS)

$(BIN3): $(SRC_CONV2D)
	$(CC) $(COMMON_O3) -o $@ $< $(LIBS)

$(BIN4): $(SRC_CONV2D_OMP)
	$(CC) $(COMMON_O3) $(OMPFLAG) -o $@ $< $(LIBS)

# ----------- Clean -----------
clean:
	$(RM) $(ALL)

# ============================================================
# Run Targets — Minimal CLI (let program defaults handle extras)
# ============================================================

# Default arguments (override with make VAR=val)
N      ?= 1000000   # 1D length
K      ?= 10001     # 1D kernel length
SEED   ?= 42        # RNG seed
OUT1D  ?= /dev/null

H      ?= 1024      # 2D height
W      ?= 1024      # 2D width
KH     ?= 5         # 2D kernel height
KW     ?= 5         # 2D kernel width
OUT2D  ?= /dev/null

THREADS ?= 8        # OMP threads
SCHED   ?= static   # OMP schedule (static|dynamic|guided|auto)
CHUNK   ?=          # OMP chunk size (optional)

# Run all 4 builds
run: run1d run1d_omp run2d run2d_omp

# Sequential 1D
run1d: $(BIN1)
	@echo "==> Running $(BIN1) with N=$(N), K=$(K), SEED=$(SEED)"
	./$(BIN1) -L $(N) -kL $(K) -o $(OUT1D) -s $(SEED)

# OpenMP 1D
run1d_omp: $(BIN2)
	@echo "==> Running $(BIN2) with N=$(N), K=$(K), THREADS=$(THREADS), SCHED=$(SCHED)$(if $(CHUNK),:$(CHUNK),), SEED=$(SEED)"
	@OMP_NUM_THREADS=$(THREADS) OMP_PROC_BIND=spread OMP_PLACES=cores \
	$(if $(filter auto,$(SCHED)),OMP_SCHEDULE=auto,OMP_SCHEDULE=$(SCHED)$(if $(CHUNK),:$(CHUNK),)) \
	./$(BIN2) -L $(N) -kL $(K) -o $(OUT1D) -s $(SEED)

# Sequential 2D
run2d: $(BIN3)
	@echo "==> Running $(BIN3) with H=$(H), W=$(W), KH=$(KH), KW=$(KW), SEED=$(SEED)"
	./$(BIN3) -H $(H) -W $(W) -kH $(KH) -kW $(KW) -o $(OUT2D) -s $(SEED)

# OpenMP 2D
run2d_omp: $(BIN4)
	@echo "==> Running $(BIN4) with H=$(H), W=$(W), KH=$(KH), KW=$(KW), THREADS=$(THREADS), SCHED=$(SCHED)$(if $(CHUNK),:$(CHUNK),), SEED=$(SEED)"
	@OMP_NUM_THREADS=$(THREADS) OMP_PROC_BIND=spread OMP_PLACES=cores \
	$(if $(filter auto,$(SCHED)),OMP_SCHEDULE=auto,OMP_SCHEDULE=$(SCHED)$(if $(CHUNK),:$(CHUNK),)) \
	./$(BIN4) -H $(H) -W $(W) -kH $(KH) -kW $(KW) -o $(OUT2D) -s $(SEED)