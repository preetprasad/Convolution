# 📊 A2 Data Collection Roadmap

```
┌─────────────────────────────────────────────────────────────────────┐
│                    A2 PERFORMANCE DATA COLLECTION                    │
│                         Complete Workflow                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  PREREQUISITES  │
└────────┬────────┘
         │
         ├─► Connect to Kaya: ssh <user>@kaya.hpc.uwa.edu.au
         ├─► Load modules: module load gcc/14.2 openmpi/5.0.5
         ├─► Build programs: make
         └─► Create dirs: mkdir -p logs metrics results

┌──────────────────────────────────────────────────────────────────────┐
│                         COLLECTION PHASES                             │
└──────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Correctness Verification                    Time: ~5 min  │
├────────────────────────────────────────────────────────────────────┤
│ Purpose: Ensure all implementations work correctly                 │
│ Jobs: 8 (small problems)                                          │
│                                                                    │
│ Commands:                                                          │
│   ./start_data_collection.sh  → Select option 1                   │
│                                                                    │
│ What to check:                                                     │
│   ✓ All jobs complete successfully                                │
│   ✓ Logs show GFLOPS metrics                                      │
│   ✓ No errors in stderr                                           │
│                                                                    │
│ Success criteria:                                                  │
│   8/8 jobs complete with positive GFLOPS                          │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Sequential Baselines                      Time: ~2-4 hrs  │
├────────────────────────────────────────────────────────────────────┤
│ Purpose: Establish baseline for speedup calculations              │
│ Jobs: ~540 (100 × 1D + 440 × 2D)                                  │
│                                                                    │
│ Commands:                                                          │
│   ./start_data_collection.sh  → Select option 3                   │
│                                                                    │
│ What to measure:                                                   │
│   • Time vs. problem size (N, K, H, W, kH, kW)                   │
│   • GFLOPS performance                                            │
│   • Memory requirements                                           │
│                                                                    │
│ Success criteria:                                                  │
│   • Clean time scaling with problem size                          │
│   • Consistent GFLOPS across runs                                 │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 3: OpenMP Scaling                            Time: ~3-6 hrs  │
├────────────────────────────────────────────────────────────────────┤
│ Purpose: Measure shared-memory parallelism                        │
│ Jobs: ~250 jobs                                                    │
│                                                                    │
│ Commands:                                                          │
│   ./start_data_collection.sh  → Select option 4                   │
│                                                                    │
│ What to measure:                                                   │
│   • Speedup vs. threads (1,2,4,8,16)                             │
│   • Schedule performance (static/dynamic/guided/auto)             │
│   • Chunk size impact                                             │
│                                                                    │
│ Expected results:                                                  │
│   • Near-linear speedup for small thread counts                   │
│   • Diminishing returns after 8-16 threads                        │
│   • Static schedule often best for regular workloads              │
│                                                                    │
│ Success criteria:                                                  │
│   • Speedup > 1 for all thread counts                            │
│   • Peak efficiency at 4-8 threads                                │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 4: MPI Scaling                               Time: ~4-8 hrs  │
├────────────────────────────────────────────────────────────────────┤
│ Purpose: Measure distributed-memory parallelism                   │
│ Jobs: ~400 jobs                                                    │
│                                                                    │
│ Commands:                                                          │
│   ./start_data_collection.sh  → Select option 5                   │
│                                                                    │
│ What to measure:                                                   │
│   • Strong scaling (fixed problem, vary processes)                │
│   • Weak scaling (scale problem with processes)                   │
│   • 2D grid topology impact (1×1, 2×2, 3×3, 4×4)                 │
│   • Communication overhead                                        │
│                                                                    │
│ Expected results:                                                  │
│   • Good strong scaling up to 16 processes                        │
│   • Better weak scaling efficiency                                │
│   • 2D topology outperforms 1D for 2D problems                   │
│                                                                    │
│ Success criteria:                                                  │
│   • Speedup increases with processes                              │
│   • Weak scaling efficiency > 80%                                 │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Hybrid MPI+OpenMP                         Time: ~4-8 hrs  │
├────────────────────────────────────────────────────────────────────┤
│ Purpose: Find optimal MPI×OMP balance                             │
│ Jobs: ~400 jobs                                                    │
│                                                                    │
│ Commands:                                                          │
│   ./start_data_collection.sh  → Select option 6                   │
│                                                                    │
│ What to measure:                                                   │
│   • Pure MPI vs. Pure OMP vs. Hybrid                             │
│   • MPI×OMP configurations (16×1, 8×2, 4×4, 2×8, 1×16)          │
│   • 2D grid + OpenMP interaction                                  │
│                                                                    │
│ Expected results:                                                  │
│   • Hybrid often better than pure approaches                      │
│   • Optimal balance depends on problem size                       │
│   • 4×4 or 2×8 often best for 16 cores                          │
│                                                                    │
│ Success criteria:                                                  │
│   • Hybrid outperforms pure MPI and pure OMP                      │
│   • Clear optimal configuration identified                        │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 6: Verification & Gap Filling                Time: ~2-4 hrs  │
├────────────────────────────────────────────────────────────────────┤
│ Purpose: Ensure complete, high-quality dataset                    │
│                                                                    │
│ Tasks:                                                             │
│   1. Check for missing data points                                │
│   2. Re-run failed jobs                                           │
│   3. Verify data quality                                          │
│   4. Fill parameter space gaps                                    │
│                                                                    │
│ Commands:                                                          │
│   sacct -S $(date -d '1 week ago' +%Y-%m-%d) --state=FAILED      │
│   grep -l error logs/*.err                                        │
│   cat metrics/*.csv | awk 'NR==1 || !/^RunID/' > merged.csv     │
│                                                                    │
│ Quality checks:                                                    │
│   ✓ No NaN or inf values                                         │
│   ✓ Positive GFLOPS for all runs                                 │
│   ✓ Consistent timing across replicates                          │
│   ✓ Full coverage of parameter space                             │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│                        DATA READY FOR ANALYSIS                     │
└────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS WORKFLOW                             │
└──────────────────────────────────────────────────────────────────────┘

┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  Merge CSVs    │─────►│   Clean Data   │─────►│   Calculate    │
│                │      │                │      │   Metrics      │
│ cat metrics/*  │      │ Remove NaN/inf │      │ Speedup, Eff.  │
└────────────────┘      └────────────────┘      └────────┬───────┘
                                                          │
                                                          ▼
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│   Export to    │◄─────│  Visualize     │◄─────│   Statistical  │
│   LaTeX/Report │      │  Performance   │      │   Analysis     │
│                │      │  Curves        │      │                │
└────────────────┘      └────────────────┘      └────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION EXAMPLES                           │
└──────────────────────────────────────────────────────────────────────┘

1. Strong Scaling Plot:
   X-axis: Number of cores (log scale)
   Y-axis: Speedup
   Lines: Different implementations (seq, omp, mpi, hybrid)
   Ideal: Linear speedup line

2. Efficiency Plot:
   X-axis: Number of cores
   Y-axis: Efficiency (%)
   Target: > 80% for good scaling

3. Weak Scaling Plot:
   X-axis: Number of processes
   Y-axis: Time (should be constant)
   Good: Flat line

4. Hybrid Configuration Heatmap:
   X-axis: MPI processes
   Y-axis: OMP threads
   Color: GFLOPS or speedup
   Goal: Find sweet spot

5. Schedule Comparison:
   X-axis: Problem size
   Y-axis: Time
   Lines: static, dynamic, guided, auto
   Compare: Which schedule wins when?

┌──────────────────────────────────────────────────────────────────────┐
│                        ESTIMATED TIMELINE                             │
└──────────────────────────────────────────────────────────────────────┘

Week 1:
  Day 1: Setup + Correctness (Phase 1)                      ✓ 30 min
  Day 2: Sequential Baselines (Phase 2)                     ✓ 2-4 hrs
  Day 3: OpenMP Scaling (Phase 3)                           ✓ 3-6 hrs
  Day 4: MPI Scaling (Phase 4)                              ✓ 4-8 hrs
  Day 5: Hybrid Exploration (Phase 5)                       ✓ 4-8 hrs
  Day 6: Verification + Gap Filling (Phase 6)               ✓ 2-4 hrs
  Day 7: Buffer for issues / re-runs

Week 2:
  Day 1-2: Data cleaning and merging
  Day 3-4: Analysis and visualization
  Day 5-7: Report writing

Total Time Estimate: 15-30 hours of computation + 2-3 days analysis

┌──────────────────────────────────────────────────────────────────────┐
│                      SUCCESS METRICS                                  │
└──────────────────────────────────────────────────────────────────────┘

Data Collection Success:
  ✓ ~1600 successful job completions
  ✓ < 5% job failure rate
  ✓ No missing parameter space regions
  ✓ CSV files with valid GFLOPS

Analysis Success:
  ✓ Clear speedup trends vs. cores
  ✓ Efficiency > 80% for small core counts
  ✓ Identified optimal configurations
  ✓ Statistical significance in comparisons

Report Success:
  ✓ All figures generated
  ✓ Performance analysis complete
  ✓ Conclusions supported by data
  ✓ Ready for submission

┌──────────────────────────────────────────────────────────────────────┐
│                      USEFUL COMMANDS                                  │
└──────────────────────────────────────────────────────────────────────┘

Monitor:
  squeue -u $USER                          # Check queue
  watch -n 10 'squeue -u $USER | wc -l'   # Auto-refresh count
  tail -f logs/conv*.err                   # Watch logs
  ls metrics/*.csv | wc -l                 # Count metrics

Verify:
  grep GFLOPS logs/*.err | sort -t'=' -k2 -n    # GFLOPS range
  sacct -S $(date -d '1 day ago' +%Y-%m-%d)     # Job history
  grep -i error logs/*.err                       # Find errors

Analyze:
  cat metrics/*.csv > all_metrics.csv            # Merge CSVs
  awk 'NR==1 || !/^RunID/' all_metrics.csv > merged.csv  # Remove dups
  wc -l merged.csv                               # Count points

Clean:
  rm logs/* metrics/* results/*            # Clean all data
  make clean                                # Clean binaries

┌──────────────────────────────────────────────────────────────────────┐
│                      GETTING HELP                                     │
└──────────────────────────────────────────────────────────────────────┘

Documentation:
  📘 DATA_COLLECTION_GUIDE.md       - Complete guide (this file)
  📋 DATA_COLLECTION_QUICKREF.md    - Quick reference card
  🔧 TESTING_FRAMEWORK.md           - Testing framework details
  📜 SLURM_SCRIPTS_REFERENCE.md     - SLURM script reference
  🛠️ MAKEFILE_GUIDE.md              - Makefile usage

Scripts:
  ./start_data_collection.sh        - Interactive setup
  batch_helpers/sweep_*.sh          - Batch submission scripts
  slurm_helpers/*.slurm             - SLURM job scripts

Support:
  📧 Ask on unit discussion board
  👥 Consult with lab demonstrators
  📚 Check unit materials on LMS

┌──────────────────────────────────────────────────────────────────────┐
│                 READY TO BEGIN? 🚀                                    │
│                                                                        │
│  Run: ./start_data_collection.sh                                     │
│                                                                        │
│  Or see: DATA_COLLECTION_QUICKREF.md for manual commands             │
└──────────────────────────────────────────────────────────────────────┘
```
