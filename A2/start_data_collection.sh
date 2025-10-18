#!/bin/bash
# Quick Start Data Collection Script
# Run this on Kaya to start basic data collection

set -e

echo "=== A2 Data Collection Quick Start ==="
echo ""

# Check if on Kaya
if [[ ! -f /etc/redhat-release ]] || [[ $(hostname) != *"kaya"* ]]; then
    echo "⚠️  Warning: This script should be run on Kaya HPC cluster"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Load modules
echo "Loading modules..."
module load gcc/14.2 2>/dev/null || echo "  gcc/14.2 already loaded"
module load openmpi/5.0.5 2>/dev/null || echo "  openmpi/5.0.5 already loaded"

# Build programs
echo ""
echo "Building programs..."
if ! make -q 2>/dev/null; then
    make
    echo "✅ Build complete"
else
    echo "✅ Programs already built"
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p logs metrics results
echo "✅ Directories ready"

# Phase selection
echo ""
echo "Select data collection phase:"
echo ""
echo "  1) Correctness Verification (quick, ~5 min)"
echo "  2) Minimal Dataset (30 jobs, ~1-2 hours)"
echo "  3) Sequential Baselines (100+ jobs, ~2-4 hours)"
echo "  4) OpenMP Scaling (200+ jobs, ~3-6 hours)"
echo "  5) MPI Scaling (300+ jobs, ~4-8 hours)"
echo "  6) Hybrid Exploration (400+ jobs, ~4-8 hours)"
echo "  7) Full Dataset (all phases, ~15-30 hours)"
echo "  8) Custom (manual configuration)"
echo ""
read -p "Enter phase number [1-8]: " phase

case $phase in
    1)
        echo ""
        echo "=== Phase 1: Correctness Verification ==="
        echo ""
        
        # Sequential
        echo "Testing sequential implementations..."
        sbatch slurm_helpers/conv1d_seq_param.slurm 10000 101 same zero 42
        sbatch slurm_helpers/conv2d_seq_param.slurm 128 128 5 5 same zero 42
        
        # OpenMP
        echo "Testing OpenMP implementations..."
        sbatch slurm_helpers/conv1d_omp_param.slurm 10000 101 4 static "" same zero 42
        sbatch slurm_helpers/conv2d_omp_param.slurm 128 128 5 5 4 static "" same zero 42
        
        # MPI
        echo "Testing MPI implementations..."
        sbatch slurm_helpers/conv1d_mpi_param.slurm 10000 101 2 same zero 42
        sbatch slurm_helpers/conv2d_mpi_param.slurm 128 128 5 5 4 same zero 42
        
        # Hybrid
        echo "Testing hybrid implementations..."
        sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 10000 101 2 2 static "" same zero 42
        sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 128 128 5 5 4 2 static "" same zero 42
        
        echo ""
        echo "✅ Submitted 8 correctness tests"
        ;;
        
    2)
        echo ""
        echo "=== Phase 2: Minimal Dataset ==="
        echo ""
        
        # Sequential baselines
        echo "Sequential baselines..."
        sbatch slurm_helpers/conv1d_seq_param.slurm 5000000 501 same zero 42
        sbatch slurm_helpers/conv2d_seq_param.slurm 2048 2048 5 5 same zero 42
        
        # OpenMP scaling
        echo "OpenMP scaling (1,2,4,8 threads)..."
        for t in 1 2 4 8; do
            sbatch slurm_helpers/conv1d_omp_param.slurm 5000000 501 $t static "" same zero 42
            sbatch slurm_helpers/conv2d_omp_param.slurm 2048 2048 5 5 $t static "" same zero 42
        done
        
        # MPI scaling
        echo "MPI scaling (1,4,9 processes)..."
        for np in 1 4 9; do
            sbatch slurm_helpers/conv1d_mpi_param.slurm 5000000 501 $np same zero 42
            sbatch slurm_helpers/conv2d_mpi_param.slurm 2048 2048 5 5 $np same zero 42
        done
        
        # Hybrid
        echo "Hybrid (4×4 config)..."
        sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 5000000 501 4 4 static "" same zero 42
        sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 2048 2048 5 5 4 4 static "" same zero 42
        
        echo ""
        echo "✅ Submitted ~30 jobs for minimal dataset"
        ;;
        
    3)
        echo ""
        echo "=== Phase 3: Sequential Baselines ==="
        echo ""
        
        # 1D Sequential sweep
        echo "1D Sequential sweep..."
        ./batch_helpers/sweep_conv1d_seq.sh \
            1000000 10000000 1000000 \
            101 1001 100 \
            20 42
        
        # 2D Sequential sweep
        echo "2D Sequential sweep..."
        ./batch_helpers/sweep_conv2d_seq.sh \
            1024 4096 512 \
            1024 4096 512 \
            3 7 2 \
            3 7 2 \
            20 42
        
        echo ""
        echo "✅ Sequential baseline sweeps started"
        ;;
        
    4)
        echo ""
        echo "=== Phase 4: OpenMP Scaling ==="
        echo ""
        
        # 1D OpenMP thread scaling
        echo "1D OpenMP thread scaling..."
        for threads in 1 2 4 8 16; do
            sbatch slurm_helpers/conv1d_omp_param.slurm \
                5000000 501 $threads static "" same zero 42
        done
        
        # 1D OpenMP schedule comparison
        echo "1D OpenMP schedule comparison..."
        for sched in static dynamic guided auto; do
            sbatch slurm_helpers/conv1d_omp_param.slurm \
                5000000 501 8 $sched "" same zero 42
        done
        
        # 2D OpenMP sweeps
        echo "2D OpenMP sweep..."
        ./batch_helpers/sweep_conv2d_omp.sh \
            1024 2048 512 \
            1024 2048 512 \
            5 5 2 \
            5 5 2 \
            20 8 static "" 42
        
        # 2D OpenMP thread scaling
        echo "2D OpenMP thread scaling..."
        for threads in 1 2 4 8 16; do
            sbatch slurm_helpers/conv2d_omp_param.slurm \
                2048 2048 5 5 $threads static "" same zero 42
        done
        
        echo ""
        echo "✅ OpenMP scaling jobs submitted"
        ;;
        
    5)
        echo ""
        echo "=== Phase 5: MPI Scaling ==="
        echo ""
        
        # 1D MPI strong scaling
        echo "1D MPI strong scaling..."
        for np in 1 2 4 8 16; do
            sbatch slurm_helpers/conv1d_mpi_param.slurm \
                20000000 1001 $np same zero 42
        done
        
        # 1D MPI weak scaling
        echo "1D MPI weak scaling..."
        sbatch slurm_helpers/conv1d_mpi_param.slurm 5000000  1001 1  same zero 42
        sbatch slurm_helpers/conv1d_mpi_param.slurm 10000000 1001 2  same zero 42
        sbatch slurm_helpers/conv1d_mpi_param.slurm 20000000 1001 4  same zero 42
        sbatch slurm_helpers/conv1d_mpi_param.slurm 40000000 1001 8  same zero 42
        
        # 2D MPI grid scaling
        echo "2D MPI grid scaling..."
        for np in 1 4 9 16 25; do
            sbatch slurm_helpers/conv2d_mpi_param.slurm \
                4096 4096 7 7 $np same zero 42
        done
        
        # 2D MPI sweep
        echo "2D MPI sweep..."
        ./batch_helpers/sweep_conv2d_mpi.sh \
            2048 4096 1024 \
            2048 4096 1024 \
            5 7 2 \
            5 7 2 \
            20 4 42
        
        echo ""
        echo "✅ MPI scaling jobs submitted"
        ;;
        
    6)
        echo ""
        echo "=== Phase 6: Hybrid Exploration ==="
        echo ""
        
        # 1D Hybrid balance (16 cores)
        echo "1D Hybrid MPI×OMP balance..."
        sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 16 1  static "" same zero 42
        sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 8  2  static "" same zero 42
        sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 4  4  static "" same zero 42
        sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 2  8  static "" same zero 42
        sbatch slurm_helpers/conv1d_mpi_omp_param.slurm 20000000 1001 1  16 static "" same zero 42
        
        # 2D Hybrid balance (16 cores)
        echo "2D Hybrid MPI×OMP balance..."
        sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 16 1 static "" same zero 42
        sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 4  4 static "" same zero 42
        sbatch slurm_helpers/conv2d_mpi_omp_param.slurm 4096 4096 7 7 1  16 static "" same zero 42
        
        # 1D Hybrid sweep
        echo "1D Hybrid sweep..."
        ./batch_helpers/sweep_conv1d_mpi_omp.sh \
            5000000 20000000 5000000 \
            501 1001 500 \
            20 4 4 static "" 42
        
        # 2D Hybrid sweep
        echo "2D Hybrid sweep..."
        ./batch_helpers/sweep_conv2d_mpi_omp.sh \
            2048 4096 1024 \
            2048 4096 1024 \
            5 7 2 \
            5 7 2 \
            20 4 4 static "" 42
        
        echo ""
        echo "✅ Hybrid exploration jobs submitted"
        ;;
        
    7)
        echo ""
        echo "=== Phase 7: Full Dataset ==="
        echo ""
        echo "This will submit ALL phases sequentially."
        read -p "Continue? This will take 15-30 hours total. (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
        
        # Run all phases
        $0 <<< "1"
        echo "Waiting 5 minutes before Phase 2..."
        sleep 300
        
        $0 <<< "3"
        echo "Waiting for Phase 3 to complete..."
        # Wait logic here...
        
        $0 <<< "4"
        $0 <<< "5"
        $0 <<< "6"
        
        echo ""
        echo "✅ Full dataset collection started"
        ;;
        
    8)
        echo ""
        echo "=== Custom Configuration ==="
        echo ""
        echo "Available scripts:"
        echo "  SLURM scripts: slurm_helpers/*.slurm"
        echo "  Batch helpers: batch_helpers/*.sh"
        echo ""
        echo "See DATA_COLLECTION_GUIDE.md for detailed instructions"
        exit 0
        ;;
        
    *)
        echo "Invalid phase number"
        exit 1
        ;;
esac

# Show queue status
echo ""
echo "Current queue status:"
squeue -u $USER

echo ""
echo "=== Monitor Progress ==="
echo "  squeue -u \$USER                    # Check queue"
echo "  ls logs/*.err | wc -l              # Count completed jobs"
echo "  ls metrics/*.csv | wc -l           # Count metrics files"
echo "  tail -f logs/conv*.err             # Watch latest log"
echo ""
echo "=== Next Steps ==="
echo "  See DATA_COLLECTION_GUIDE.md for:"
echo "  - Monitoring job progress"
echo "  - Data quality verification"
echo "  - Analysis and visualization"
echo ""
echo "✅ Data collection started! 🚀"
