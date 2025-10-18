/*------------------------------------------------------------------------------
  conv2d_mpi.c — Pure-MPI 2-D convolution/correlation with TRUE 2-D Cartesian topology

  OVERVIEW
    Computes Y = F * G (convolution) or Y = F ⊗ G (correlation) in 2-D using
    MPI with a TRUE 2-D Cartesian topology (MPI_Cart_create with ndims=2) and
    full 2D block decomposition where BOTH height and width are partitioned across
    ranks. Supports SAME/FULL modes, arbitrary stride (sH, sW), selectable padding,
    and both file and RNG inputs. Communication uses phased non-blocking 4-direction
    halo exchange. Timing measures ONLY the arithmetic kernel. Output can be
    human-readable text or binary (default).

  KEY FEATURES
    • Operations:   --conv (default) or --corr
    • Modes:        SAME (centered kernel), FULL (full convolution with overlaps)
    • Padding:      zero | none | const (with -c/--cval)
    • Stride:       -sH/--stride-h M, -sW/--stride-w N (sample every M×N)
    • Inputs:
        - Text  .txt  (assignment format: "H W" header + H×W floats)
        - Binary .bin (header int32 H, int32 W, followed by H×W float32)
        - RNG   -H/-W/-kH/-kW with -se/--seed
        - Optional --parallel-gen: each rank deterministically generates its
          2D block of F by global index; G is generated on root and broadcast.
          (Default: root generates full input and scatters to ranks.)
    
  2D CARTESIAN TOPOLOGY
    • MPI_Cart_create with ndims=2 creates P_rows × P_cols process grid
    • MPI_Dims_create automatically balances grid dimensions for load balancing
    • Each rank has up to 4 neighbors: north, south, east, west (MPI_Cart_shift)
    • 2D coordinates: (my_row_idx, my_col_idx) obtained via MPI_Cart_coords
    • True 2D block decomposition: both H and W are partitioned across ranks
    
  HALO EXCHANGE (4-Direction Phased Communication)
    • Halo widths: HH = KH-1 (vertical), HW = KW-1 (horizontal)
    • Phase 1 (N/S): Vertical neighbors exchange top/bottom halos
      - Per-row sends (separate MPI_Isend per row) to preserve buffer stride
      - Ensures correct 2D array layout with buf_W stride between rows
    • Phase 2 (E/W): Horizontal neighbors exchange left/right halos
      - Column packing/unpacking required (row-major storage)
      - Includes corner halos from Phase 1 for complete boundary coverage
    • Non-blocking: MPI_Irecv/Isend with MPI_Waitall for overlap potential
    
  OUTPUT POSITIONING (Mode-Specific)
    • SAME mode: Non-overlapping output blocks, cumulative positioning
      - Each rank's output block placed sequentially in global output
    • FULL mode: Overlapping output blocks, direct positioning
      - Rank owning input position i produces output position i (direct mapping)
      - Accounts for KH-1 and KW-1 overlap between adjacent blocks
    
  PERFORMANCE & TIMING
    • Timing:       Kernel-only via MPI_Wtime; min/avg/max per-rank reported
    • GFLOP/s:      Computed using Σ(2*KH*KW*local_out_count)/max(time)
    • Scalability:  Can use up to H × W / (KH × KW) ranks (much better than 1D)
    
  I/O & OUTPUT
    • MPI-IO:       Collective parallel read/write using MPI_Type_create_subarray
    • Binary (default): [int32 outH][int32 outW] + outH×outW float32 (rounded 3dp)
    • Text (--text):    Assignment style, rounded to 3 dp
    • Metrics:      Per-run CSV at metrics/metrics_<RUNID>.csv

  BUILD
      mpicc -std=c11 -O2 -Wall -Wextra -Werror -o conv2d_mpi conv2d_mpi.c
      # with debug prints:
      mpicc -DDEBUG_MPI=1 -std=c11 -O2 -Wall -Wextra -Werror -o conv2d_mpi conv2d_mpi.c

  TYPICAL RUNS
    # Binary output (default), convolution:
    mpirun -np 4 ./conv2d_mpi -H 1024 -W 1024 -kH 5 -kW 5 -m same -sH 2 -sW 2 -se 42 -o Y.bin

    # Text output, correlation:
    mpirun -np 4 ./conv2d_mpi --corr -H 512 -W 512 -kH 7 -kW 7 -m full --text -o Y.txt

    # Input from files:
    mpirun -np 4 ./conv2d_mpi -f img.txt -g ker.txt -m same -sH 1 -sW 1 -o Y.bin

    # Parallel generation:
    mpirun -np 8 ./conv2d_mpi -H 4096 -W 4096 -kH 11 -kW 11 --parallel-gen -se 123 -o Y.bin

  ARGUMENTS
      -f, --file PATH        Input image F (text .txt or binary .bin)
      -g, --kernel PATH      Kernel G (text .txt or binary .bin)
      -H, --rows N           Generate F with height N (if -f not given)
      -W, --cols M           Generate F with width M (if -f not given)
      -kH, --krows K         Generate G with height K (if -g not given)
      -kW, --kcols L         Generate G with width L (if -g not given)
      -se, --seed S          RNG seed
      --conv                 Perform convolution (default)
      --corr                 Perform correlation
      -m,  --mode M          same | full   (default: same)
      -p,  --padding P       zero | none | const   (default: zero)
      -c,  --cval V          Constant pad value when -p const
      -sH, --stride-h M      Vertical stride (>=1, default: 1)
      -sW, --stride-w N      Horizontal stride (>=1, default: 1)
      --parallel-gen         Each rank generates its slice of F locally
      --text                 Force text output (default is binary)
      -o,  --out PATH        Output file (required)

  INPUT FORMATS
    Text (.txt):
      line 1: H W
      following: H×W floats (row-major, whitespace-separated)
    Binary (.bin):
      8 bytes: int32 H, int32 W
      4*H*W bytes: float32 values (row-major)

  OUTPUT FORMATS
    Text (--text):
      line 1: outH outW
      following: outH×outW floats (%.3f, row-major)
    Binary (default):
      8 bytes: int32 outH, int32 outW
      4*outH*outW bytes: float32 values (rounded to 3 dp)

  DETERMINISM & COMPARABILITY
    - Fixed seed → deterministic results (both root-gen and --parallel-gen)
    - --parallel-gen uses indexable PRNG; root path uses stdlib rand
    - Outputs rounded to 3 decimal places before writing

  DEBUGGING
    - Build with -DDEBUG_MPI=1 for per-rank logs showing decomposition,
      halo width, generation mode, kernel time, and output positioning.
    - Build with -DDEBUG_HALO=1 for detailed halo exchange verification.

  IMPLEMENTATION NOTES
    • 2D Cartesian topology with automatic grid balancing (MPI_Dims_create)
    • 2D block decomposition: each rank owns a rectangular H_block × W_block region
    • Phased 4-direction halo exchange (N/S then E/W) for complete boundary coverage
    • Per-row halo sends preserve buffer stride in 2D arrays (row-major layout)
    • Column halos require packing/unpacking due to row-major storage
    • FULL mode uses direct position mapping (output_pos = input_pos)
    • SAME mode uses cumulative positioning (non-overlapping blocks)
    • Padding applied in convolution kernel (no fabricated ghost cells for const)
    • Manual 2D scatter/gather for non-contiguous block distribution
    • This implementation prioritizes correctness, clarity, and lecture alignment

------------------------------------------------------------------------------*/

#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <getopt.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>

/* ------------------- Macros & Utils ------------------- */
#define IDX2(i, j, ldW) ((size_t)(i) * (size_t)(ldW) + (size_t)(j))

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static int ends_with(const char *s, const char *suf)
{
    size_t n = strlen(s), m = strlen(suf);
    return (n >= m) && (memcmp(s + (n - m), suf, m) == 0);
}

/* ------------------- Enums ------------------- */
typedef enum
{
    MODE_SAME = 0,
    MODE_FULL = 1
} conv_mode;

typedef enum
{
    PAD_ZERO = 0,
    PAD_NONE = 1,
    PAD_CONST = 2
} pad_mode;

typedef enum
{
    OP_CONV = 0,
    OP_CORR = 1
} op_kind;

/* ------------------- Forward Declarations ------------------- */
void usage(const char *prog);

int parse_args(int argc, char **argv,
               const char **img_path, const char **ker_path, const char **out_path,
               long *H_req, long *W_req, long *KH_req, long *KW_req,
               unsigned long *seed, int *have_seed,
               op_kind *op, conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *sH, int *sW, int *parallel_gen, int *text_mode);

float *read_matrix_2d_txt(const char *path, int *H_out, int *W_out);
void write_matrix_2d_txt(const char *path, const float *A, int H, int W);
float *gen_matrix_2d(int H, int W);
float gen_value_at_index_2d(unsigned long seed, long long gi, long long gj);
float *flip_kernel_2d(const float *G, int KH, int KW);

void ensure_dir(const char *path);
double elapsed_seconds_mpi(double t0, double t1);
void log_metrics(int H, int W, int KH, int KW, int outH, int outW,
                 op_kind op, conv_mode cmode, pad_mode pmode, float cval,
                 int sH, int sW,
                 double elapsed_secs, double gflops,
                 int np, int P_rows, int P_cols);

/* ------------------- CLI Parsing ------------------- */
void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--conv|--corr] "
            "[-f img.txt|.bin | -H H --rows H -W W --cols W] "
            "[-g ker.txt|.bin | -kH KH --krows KH -kW KW --kcols KW] "
            "-o out.txt|out.bin|--out path "
            "[-se seed|--seed seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value] "
            "[-sH stride_h|--stride-h stride_h] "
            "[-sW stride_w|--stride-w stride_w] "
            "[--parallel-gen] [--text]\n",
            prog);
}

int parse_args(int argc, char **argv,
               const char **img_path, const char **ker_path, const char **out_path,
               long *H_req, long *W_req, long *KH_req, long *KW_req,
               unsigned long *seed, int *have_seed,
               op_kind *op, conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *sH, int *sW, int *parallel_gen, int *text_mode)
{
    *img_path = *ker_path = *out_path = NULL;
    *H_req = *W_req = *KH_req = *KW_req = -1;
    *seed = (unsigned long)time(NULL);
    *have_seed = 0;
    *op = OP_CONV;
    *cmode = MODE_SAME;
    *pmode = PAD_ZERO;
    *cval = 0.0f;
    *have_cval = 0;
    *sH = 1;
    *sW = 1;
    *parallel_gen = 0;
    *text_mode = 0;

    /* Pre-scan for custom args */
    int fargc = 1;
    char **fargv = (char **)malloc((size_t)argc * sizeof(char *));
    if (!fargv)
    {
        perror("malloc fargv");
        return 0;
    }
    fargv[0] = argv[0];

    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        if (!strcmp(a, "-kH"))
        {
            if (i + 1 < argc)
                *KH_req = atol(argv[++i]);
            continue;
        }
        if (!strncmp(a, "-kH=", 4))
        {
            *KH_req = atol(a + 4);
            continue;
        }
        if (!strcmp(a, "-kW"))
        {
            if (i + 1 < argc)
                *KW_req = atol(argv[++i]);
            continue;
        }
        if (!strncmp(a, "-kW=", 4))
        {
            *KW_req = atol(a + 4);
            continue;
        }
        if (!strcmp(a, "-sH"))
        {
            if (i + 1 < argc)
                *sH = atoi(argv[++i]);
            continue;
        }
        if (!strncmp(a, "-sH=", 4))
        {
            *sH = atoi(a + 4);
            continue;
        }
        if (!strcmp(a, "-sW"))
        {
            if (i + 1 < argc)
                *sW = atoi(argv[++i]);
            continue;
        }
        if (!strncmp(a, "-sW=", 4))
        {
            *sW = atoi(a + 4);
            continue;
        }
        if (!strcmp(a, "-se"))
        {
            if (i + 1 < argc)
            {
                *seed = (unsigned long)atol(argv[++i]);
                *have_seed = 1;
            }
            continue;
        }
        if (!strncmp(a, "-se=", 4))
        {
            *seed = (unsigned long)atol(a + 4);
            *have_seed = 1;
            continue;
        }
        if (!strcmp(a, "--parallel-gen"))
        {
            *parallel_gen = 1;
            continue;
        }
        if (!strcmp(a, "--text"))
        {
            *text_mode = 1;
            continue;
        }
        fargv[fargc++] = argv[i];
    }

    static struct option long_opts[] = {
        {"conv", no_argument, 0, 10},
        {"corr", no_argument, 0, 11},
        {"file", required_argument, 0, 'f'},
        {"kernel", required_argument, 0, 'g'},
        {"out", required_argument, 0, 'o'},
        {"seed", required_argument, 0, 's'},
        {"mode", required_argument, 0, 'm'},
        {"padding", required_argument, 0, 'p'},
        {"cval", required_argument, 0, 'c'},
        {"rows", required_argument, 0, 'H'},
        {"cols", required_argument, 0, 'W'},
        {"krows", required_argument, 0, 1},
        {"kcols", required_argument, 0, 2},
        {"stride-h", required_argument, 0, 3},
        {"stride-w", required_argument, 0, 4},
        {0, 0, 0, 0}};

    int opt, idx = 0;
    opterr = 0;
    while ((opt = getopt_long(fargc, fargv, "f:g:o:s:m:p:c:H:W:", long_opts, &idx)) != -1)
    {
        switch (opt)
        {
        case 10:
            *op = OP_CONV;
            break;
        case 11:
            *op = OP_CORR;
            break;
        case 'f':
            *img_path = optarg;
            break;
        case 'g':
            *ker_path = optarg;
            break;
        case 'o':
            *out_path = optarg;
            break;
        case 's':
            *seed = (unsigned long)atol(optarg);
            *have_seed = 1;
            break;
        case 'm':
            if (!strcmp(optarg, "same"))
                *cmode = MODE_SAME;
            else if (!strcmp(optarg, "full"))
                *cmode = MODE_FULL;
            else
            {
                fprintf(stderr, "unknown mode '%s'\n", optarg);
                free(fargv);
                return 0;
            }
            break;
        case 'p':
            if (!strcmp(optarg, "zero"))
                *pmode = PAD_ZERO;
            else if (!strcmp(optarg, "none"))
                *pmode = PAD_NONE;
            else if (!strcmp(optarg, "const"))
                *pmode = PAD_CONST;
            else
            {
                fprintf(stderr, "unknown padding '%s'\n", optarg);
                free(fargv);
                return 0;
            }
            break;
        case 'c':
            *cval = (float)atof(optarg);
            *have_cval = 1;
            break;
        case 'H':
            *H_req = atol(optarg);
            break;
        case 'W':
            *W_req = atol(optarg);
            break;
        case 1: /* --krows */
            *KH_req = atol(optarg);
            break;
        case 2: /* --kcols */
            *KW_req = atol(optarg);
            break;
        case 3: /* --stride-h */
            *sH = atoi(optarg);
            break;
        case 4: /* --stride-w */
            *sW = atoi(optarg);
            break;
        default:
            free(fargv);
            return 0;
        }
    }
    free(fargv);

    if (!*out_path)
    {
        fprintf(stderr, "Missing -o/--out\n");
        return 0;
    }
    if (!*img_path && (*H_req <= 0 || *W_req <= 0))
    {
        fprintf(stderr, "Missing -H/--rows and/or -W/--cols\n");
        return 0;
    }
    if (!*ker_path && (*KH_req <= 0 || *KW_req <= 0))
    {
        fprintf(stderr, "Missing -kH/--krows and/or -kW/--kcols\n");
        return 0;
    }
    if (*pmode == PAD_CONST && !*have_cval)
        fprintf(stderr, "warning: -p const without -c/--cval; using cval=0.0\n");
    if (*sH < 1 || *sW < 1)
    {
        fprintf(stderr, "stride must be >= 1\n");
        return 0;
    }

    return 1;
}

/* ------------------- I/O Functions ------------------- */
float *read_matrix_2d_txt(const char *path, int *H_out, int *W_out)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        perror(path);
        return NULL;
    }
    int H = 0, W = 0;
    if (fscanf(fp, "%d %d", &H, &W) != 2 || H <= 0 || W <= 0)
    {
        fprintf(stderr, "bad header in %s\n", path);
        fclose(fp);
        return NULL;
    }
    float *A = (float *)malloc((size_t)H * (size_t)W * sizeof(float));
    if (!A)
    {
        fprintf(stderr, "oom reading %s\n", path);
        fclose(fp);
        return NULL;
    }
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            if (fscanf(fp, "%f", &A[IDX2(i, j, W)]) != 1)
            {
                fprintf(stderr, "bad body in %s\n", path);
                free(A);
                fclose(fp);
                return NULL;
            }
    fclose(fp);
    *H_out = H;
    *W_out = W;
    return A;
}

void write_matrix_2d_txt(const char *path, const float *A, int H, int W)
{
    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        perror(path);
        return;
    }
    fprintf(fp, "%d %d\n", H, W);
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
            fprintf(fp, (j + 1 == W) ? "%.3f\n" : "%.3f ", A[IDX2(i, j, W)]);
    }
    fclose(fp);
}

float *gen_matrix_2d(int H, int W)
{
    if (H <= 0 || W <= 0)
        return NULL;
    float *A = (float *)malloc((size_t)H * (size_t)W * sizeof(float));
    if (!A)
        return NULL;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
        {
            float u01 = (float)rand() / (float)RAND_MAX;
            A[IDX2(i, j, W)] = -1.0f + 2.0f * u01;
        }
    return A;
}

/* Indexable PRNG for parallel generation */
static inline unsigned long long splitmix64(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x = x ^ (x >> 31);
    return x;
}

float gen_value_at_index_2d(unsigned long seed, long long gi, long long gj)
{
    unsigned long long x = ((unsigned long long)seed << 32) ^ ((unsigned long long)gi << 16) ^ (unsigned long long)gj;
    unsigned long long r = splitmix64(x);
    float u01 = (float)((r >> 40) & 0xFFFFFF) / (float)0x1000000;
    return -1.0f + 2.0f * u01;
}

float *flip_kernel_2d(const float *G, int KH, int KW)
{
    float *Gf = (float *)malloc((size_t)KH * (size_t)KW * sizeof(float));
    if (!Gf)
        return NULL;
    for (int u = 0; u < KH; u++)
        for (int v = 0; v < KW; v++)
            Gf[IDX2(u, v, KW)] = G[IDX2(KH - 1 - u, KW - 1 - v, KW)];
    return Gf;
}

/* ------------------- Utility Functions ------------------- */
void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        mkdir(path, 0755);
}

double elapsed_seconds_mpi(double t0, double t1)
{
    return t1 - t0;
}

void log_metrics(int H, int W, int KH, int KW, int outH, int outW,
                 op_kind op, conv_mode cmode, pad_mode pmode, float cval,
                 int sH, int sW,
                 double elapsed_secs, double gflops,
                 int np, int P_rows, int P_cols)
{
    ensure_dir("metrics/");
    const char *slurm = getenv("SLURM_JOB_ID");
    char runid[128];
    if (slurm && slurm[0])
        snprintf(runid, sizeof(runid), "SLURM_%s", slurm);
    else
    {
        time_t t = time(NULL);
        struct tm tm;
        localtime_r(&t, &tm);
        pid_t pid = getpid();
        strftime(runid, sizeof(runid), "LOCAL_%Y%m%d_%H%M%S", &tm);
        size_t L = strlen(runid);
        snprintf(runid + L, sizeof(runid) - L, "_%d", (int)pid);
    }
    char fname[256];
    snprintf(fname, sizeof(fname), "metrics/metrics_%s.csv", runid);
    FILE *csv = fopen(fname, "w");
    if (!csv)
    {
        perror(fname);
        return;
    }
    fprintf(csv, "RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,sH,sW,time,gflops,np,P_rows,P_cols\n");
    fprintf(csv, "%s,%d,%d,%d,%d,%d,%d,%s,%s,%s,%.9g,%d,%d,%.9f,%.6f,%d,%d,%d\n",
            runid, H, W, KH, KW, outH, outW,
            (op == OP_CONV ? "conv" : "corr"),
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            sH, sW, elapsed_secs, gflops, np, P_rows, P_cols);
    fclose(csv);
}

/* ------------------- Main ------------------- */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Create 2-D Cartesian communicator (2D block decomposition) */
    MPI_Comm comm;
    int dims[2] = {0, 0};  /* Let MPI decide optimal 2D grid: [P_rows, P_cols] */
    int coords[2];         /* My position in grid: [my_row_idx, my_col_idx] */
    {
        MPI_Dims_create(world_size, 2, dims);
        int periods[2] = {0, 0};  /* No wraparound */
        int reorder = 1;  /* Allow reordering for better locality */
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm);
        if (comm == MPI_COMM_NULL)
            comm = MPI_COMM_WORLD;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, rank, 2, coords);

    const int P_rows = dims[0];
    const int P_cols = dims[1];
    const int my_row_idx = coords[0];
    const int my_col_idx = coords[1];

    /* Find 4 neighbors (N/S/E/W) */
    int north = MPI_PROC_NULL, south = MPI_PROC_NULL;
    int west = MPI_PROC_NULL, east = MPI_PROC_NULL;
    MPI_Cart_shift(comm, 0, 1, &north, &south);  /* Dimension 0: vertical */
    MPI_Cart_shift(comm, 1, 1, &west, &east);     /* Dimension 1: horizontal */

    const char *img_path = NULL, *ker_path = NULL, *out_path = NULL;
    long H_req = -1, W_req = -1, KH_req = -1, KW_req = -1;
    unsigned long seed = (unsigned long)time(NULL);
    int have_seed = 0;
    op_kind op = OP_CONV;
    conv_mode cmode = MODE_SAME;
    pad_mode pmode = PAD_ZERO;
    float cval = 0.0f;
    int have_cval = 0;
    int sH = 1, sW = 1;
    int parallel_gen = 0;
    int text_mode = 0;

    if (!parse_args(argc, argv, &img_path, &ker_path, &out_path,
                    &H_req, &W_req, &KH_req, &KW_req,
                    &seed, &have_seed,
                    &op, &cmode, &pmode, &cval, &have_cval,
                    &sH, &sW, &parallel_gen, &text_mode))
    {
        if (rank == 0)
            usage(argv[0]);
        MPI_Abort(comm, 1);
    }

    int H = 0, W = 0, KH = 0, KW = 0;
    float *F_root = NULL, *G_root = NULL;

    const int f_is_bin = (img_path && ends_with(img_path, ".bin"));
    const int g_is_bin = (ker_path && ends_with(ker_path, ".bin"));

    /* Root reads/generates input (binary reads handled separately below) */
    if (rank == 0)
    {
        if (img_path && !f_is_bin)
        {
            F_root = read_matrix_2d_txt(img_path, &H, &W);
            if (!F_root)
            {
                fprintf(stderr, "Failed to read %s\n", img_path);
                MPI_Abort(comm, 1);
            }
        }
        else if (img_path && f_is_bin)
        {
            /* Binary input F: read header only on root for now */
            MPI_File fh;
            MPI_Status st;
            int rc = MPI_File_open(MPI_COMM_SELF, (char *)img_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
            if (rc != MPI_SUCCESS)
            {
                fprintf(stderr, "Failed to open binary %s\n", img_path);
                MPI_Abort(comm, 1);
            }
            int hdr[2] = {0, 0};
            MPI_File_read_at(fh, 0, hdr, 2, MPI_INT, &st);
            H = hdr[0];
            W = hdr[1];
            MPI_File_close(&fh);
        }
        else if (!img_path)
        {
            H = (int)H_req;
            W = (int)W_req;
            if (!parallel_gen)
            {
                srand((unsigned)seed);
                F_root = gen_matrix_2d(H, W);
                if (!F_root)
                {
                    fprintf(stderr, "Failed to generate F\n");
                    MPI_Abort(comm, 1);
                }
            }
        }

        if (ker_path && !g_is_bin)
        {
            G_root = read_matrix_2d_txt(ker_path, &KH, &KW);
            if (!G_root)
            {
                fprintf(stderr, "Failed to read %s\n", ker_path);
                MPI_Abort(comm, 1);
            }
        }
        else if (ker_path && g_is_bin)
        {
            /* Binary input G: read on root */
            MPI_File fh;
            MPI_Status st;
            int rc = MPI_File_open(MPI_COMM_SELF, (char *)ker_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
            if (rc != MPI_SUCCESS)
            {
                fprintf(stderr, "Failed to open binary %s\n", ker_path);
                MPI_Abort(comm, 1);
            }
            int hdr[2] = {0, 0};
            MPI_File_read_at(fh, 0, hdr, 2, MPI_INT, &st);
            KH = hdr[0];
            KW = hdr[1];
            if (KH <= 0 || KW <= 0)
            {
                fprintf(stderr, "bad header in binary %s\n", ker_path);
                MPI_File_close(&fh);
                MPI_Abort(comm, 1);
            }
            G_root = (float *)malloc((size_t)KH * (size_t)KW * sizeof(float));
            if (!G_root)
            {
                fprintf(stderr, "OOM reading binary %s\n", ker_path);
                MPI_File_close(&fh);
                MPI_Abort(comm, 1);
            }
            MPI_Offset base = (MPI_Offset)(2 * sizeof(int));
            MPI_File_read_at(fh, base, G_root, KH * KW, MPI_FLOAT, &st);
            MPI_File_close(&fh);
        }
        else if (!ker_path)
        {
            KH = (int)KH_req;
            KW = (int)KW_req;
            G_root = gen_matrix_2d(KH, KW);
            if (!G_root)
            {
                fprintf(stderr, "Failed to generate G\n");
                MPI_Abort(comm, 1);
            }
        }
    }

    /* Broadcast dimensions and config */
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&KH, 1, MPI_INT, 0, comm);
    MPI_Bcast(&KW, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cmode, 1, MPI_INT, 0, comm);
    MPI_Bcast(&pmode, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cval, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(&sH, 1, MPI_INT, 0, comm);
    MPI_Bcast(&sW, 1, MPI_INT, 0, comm);
    MPI_Bcast(&parallel_gen, 1, MPI_INT, 0, comm);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, comm);
    MPI_Bcast(&op, 1, MPI_INT, 0, comm);

    /* 2D Block decomposition */
    /* Vertical (rows) */
    int base_rows = H / P_rows;
    int rem_rows = H % P_rows;
    int my_block_H = base_rows + (my_row_idx < rem_rows ? 1 : 0);
    
    int my_H_start = 0;
    for (int r = 0; r < my_row_idx; r++) {
        int rows_for_r = base_rows + (r < rem_rows ? 1 : 0);
        my_H_start += rows_for_r;
    }

    /* Horizontal (columns) */
    int base_cols = W / P_cols;
    int rem_cols = W % P_cols;
    int my_block_W = base_cols + (my_col_idx < rem_cols ? 1 : 0);
    
    int my_W_start = 0;
    for (int c = 0; c < my_col_idx; c++) {
        int cols_for_c = base_cols + (c < rem_cols ? 1 : 0);
        my_W_start += cols_for_c;
    }

#ifdef DEBUG_MPI
    if (rank == 0) {
        printf("2D Cartesian Grid: %d × %d ranks (P_rows × P_cols)\n", P_rows, P_cols);
    }
    printf("[%d] coords=(%d,%d) block=[%d:%d, %d:%d] size=%dx%d neighbors=(N:%d S:%d E:%d W:%d)\n",
           rank, my_row_idx, my_col_idx,
           my_H_start, my_H_start + my_block_H - 1,
           my_W_start, my_W_start + my_block_W - 1,
           my_block_H, my_block_W,
           north, south, east, west);
#endif

    /* Allocate local block storage */
    float *F_local = (float *)malloc((size_t)my_block_H * (size_t)my_block_W * sizeof(float));
    if (!F_local)
    {
        fprintf(stderr, "[%d] OOM F_local\n", rank);
        MPI_Abort(comm, 1);
    }

    /* Handle F input: binary parallel read, scatter, or parallel-gen */
    if (img_path && f_is_bin)
    {
        /* Binary parallel read using MPI-IO */
        MPI_File fh;
        MPI_Status st;
        int rc = MPI_File_open(comm, (char *)img_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        if (rc != MPI_SUCCESS)
        {
            fprintf(stderr, "[%d] MPI_File_open failed for %s\n", rank, img_path);
            MPI_Abort(comm, 1);
        }

        /* Verify header */
        int hdr[2] = {0, 0};
        if (rank == 0)
        {
            MPI_File_read_at(fh, 0, hdr, 2, MPI_INT, &st);
        }
        MPI_Bcast(hdr, 2, MPI_INT, 0, comm);
        if (hdr[0] != H || hdr[1] != W)
        {
            fprintf(stderr, "[%d] Binary header mismatch in %s\n", rank, img_path);
            MPI_File_close(&fh);
            MPI_Abort(comm, 1);
        }

        /* Read my 2D block: use subarray type for non-contiguous access */
        int global_sizes[2] = {H, W};
        int local_sizes[2] = {my_block_H, my_block_W};
        int starts[2] = {my_H_start, my_W_start};
        
        MPI_Datatype file_type;
        MPI_Type_create_subarray(2, global_sizes, local_sizes, starts,
                                 MPI_ORDER_C, MPI_FLOAT, &file_type);
        MPI_Type_commit(&file_type);
        
        /* Set file view (skip 8-byte header: 2 ints) */
        MPI_Offset header_offset = 2 * sizeof(int);
        MPI_File_set_view(fh, header_offset, MPI_FLOAT, file_type, "native", MPI_INFO_NULL);
        
        /* Collective read */
        MPI_File_read_all(fh, F_local, my_block_H * my_block_W, MPI_FLOAT, &st);
        
        MPI_Type_free(&file_type);
        MPI_File_close(&fh);
    }
    else if (parallel_gen)
    {
        /* Parallel generation using indexable PRNG - 2D block */
        for (int li = 0; li < my_block_H; li++)
            for (int lj = 0; lj < my_block_W; lj++)
            {
                long long gi = (long long)(my_H_start + li);
                long long gj = (long long)(my_W_start + lj);
                F_local[IDX2(li, lj, my_block_W)] = gen_value_at_index_2d(seed, gi, gj);
            }
    }
    else
    {
        /* Manual 2D block scatter from root */
        if (rank == 0)
        {
            /* Root sends blocks to all ranks (including itself) */
            for (int dest_rank = 0; dest_rank < size; dest_rank++)
            {
                /* Calculate dest rank's coordinates and block info */
                int dest_coords[2];
                MPI_Cart_coords(comm, dest_rank, 2, dest_coords);
                int dest_row_idx = dest_coords[0];
                int dest_col_idx = dest_coords[1];
                
                /* Calculate dest's vertical block */
                int dest_block_H = base_rows + (dest_row_idx < rem_rows ? 1 : 0);
                int dest_H_start = 0;
                for (int r = 0; r < dest_row_idx; r++) {
                    dest_H_start += base_rows + (r < rem_rows ? 1 : 0);
                }
                
                /* Calculate dest's horizontal block */
                int dest_block_W = base_cols + (dest_col_idx < rem_cols ? 1 : 0);
                int dest_W_start = 0;
                for (int c = 0; c < dest_col_idx; c++) {
                    dest_W_start += base_cols + (c < rem_cols ? 1 : 0);
                }
                
                /* Pack block from F_root */
                float *block = (float *)malloc((size_t)dest_block_H * (size_t)dest_block_W * sizeof(float));
                if (!block) {
                    fprintf(stderr, "[%d] OOM scatter block\n", rank);
                    MPI_Abort(comm, 1);
                }
                
                for (int i = 0; i < dest_block_H; i++) {
                    for (int j = 0; j < dest_block_W; j++) {
                        block[i * dest_block_W + j] = F_root[(dest_H_start + i) * W + (dest_W_start + j)];
                    }
                }
                
                if (dest_rank == 0) {
                    /* Copy to self */
                    memcpy(F_local, block, (size_t)dest_block_H * (size_t)dest_block_W * sizeof(float));
                } else {
                    /* Send to other rank */
                    MPI_Send(block, dest_block_H * dest_block_W, MPI_FLOAT,
                             dest_rank, 100, comm);
                }
                free(block);
            }
        }
        else
        {
            /* Non-root receives its block */
            MPI_Recv(F_local, my_block_H * my_block_W, MPI_FLOAT,
                     0, 100, comm, MPI_STATUS_IGNORE);
        }
    }

    if (rank == 0 && F_root)
    {
        free(F_root);
        F_root = NULL;
    }

    /* Broadcast kernel G */
    float *G = (float *)malloc((size_t)KH * (size_t)KW * sizeof(float));
    if (!G)
    {
        fprintf(stderr, "[%d] OOM G\n", rank);
        MPI_Abort(comm, 1);
    }

    if (rank == 0)
        memcpy(G, G_root, (size_t)KH * (size_t)KW * sizeof(float));
    MPI_Bcast(G, KH * KW, MPI_FLOAT, 0, comm);

    if (rank == 0 && G_root)
    {
        free(G_root);
        G_root = NULL;
    }

    /* Flip kernel if convolution */
    float *G_use = G;
    float *G_flip = NULL;
    if (op == OP_CONV)
    {
        G_flip = flip_kernel_2d(G, KH, KW);
        if (!G_flip)
        {
            fprintf(stderr, "[%d] OOM G_flip\n", rank);
            MPI_Abort(comm, 1);
        }
        G_use = G_flip;
    }

    /* Mode-aware halo widths: SAME needs only kernel radius, FULL needs full overlap */
    int HH, HW;  /* Vertical and horizontal halo widths */
    if (cmode == MODE_SAME) {
        /* For SAME mode, we only need kernel radius to compute centered output */
        HH = (KH > 1 ? KH / 2 : 0);  /* floor(KH/2) */
        HW = (KW > 1 ? KW / 2 : 0);  /* floor(KW/2) */
    } else {
        /* For FULL mode, we need full kernel overlap */
        HH = (KH > 0 ? KH - 1 : 0);
        HW = (KW > 0 ? KW - 1 : 0);
    }
    
    const int buf_H = my_block_H + 2 * HH;
    const int buf_W = my_block_W + 2 * HW;
    
    float *buf = (float *)calloc((size_t)buf_H * (size_t)buf_W, sizeof(float));
    if (!buf)
    {
        fprintf(stderr, "[%d] OOM buf\n", rank);
        MPI_Abort(comm, 1);
    }

    /* Copy local data into buffer center */
    for (int li = 0; li < my_block_H; li++)
        for (int lj = 0; lj < my_block_W; lj++)
            buf[IDX2(li + HH, lj + HW, buf_W)] = F_local[IDX2(li, lj, my_block_W)];

    int edge_rows = (HH <= my_block_H ? HH : my_block_H);
    int edge_cols = (HW <= my_block_W ? HW : my_block_W);

    /* Compute neighbor block sizes (everyone knows the decomposition) */
    int north_block_H = (my_row_idx > 0)
        ? (base_rows + ((my_row_idx - 1) < rem_rows ? 1 : 0)) : 0;
    int south_block_H = (my_row_idx + 1 < P_rows)
        ? (base_rows + ((my_row_idx + 1) < rem_rows ? 1 : 0)) : 0;
    int west_block_W  = (my_col_idx > 0)
        ? (base_cols + ((my_col_idx - 1) < rem_cols ? 1 : 0)) : 0;
    int east_block_W  = (my_col_idx + 1 < P_cols)
        ? (base_cols + ((my_col_idx + 1) < rem_cols ? 1 : 0)) : 0;

    /* Limit posted receives to what neighbors can actually send */
    int north_halo_rows = (north != MPI_PROC_NULL) 
        ? (HH < north_block_H ? HH : north_block_H) : 0;
    int south_halo_rows = (south != MPI_PROC_NULL) 
        ? (HH < south_block_H ? HH : south_block_H) : 0;
    int west_halo_cols  = (west != MPI_PROC_NULL) 
        ? (HW < west_block_W  ? HW : west_block_W ) : 0;
    int east_halo_cols  = (east != MPI_PROC_NULL) 
        ? (HW < east_block_W  ? HW : east_block_W ) : 0;

    /* ===== PHASE 1: NORTH / SOUTH (contiguous rows) ===== */
    MPI_Request ns_reqs[8];  /* Up to 2*HH sends + 2*HH recvs for N/S */
    int ns_req_idx = 0;
    
    if (north != MPI_PROC_NULL) {
        /* Recv north halo - only as many rows as neighbor can send */
        for (int i = 0; i < north_halo_rows; i++) {
            MPI_Irecv(&buf[IDX2(i, HW, buf_W)], my_block_W, MPI_FLOAT, 
                      north, 202 + i, comm, &ns_reqs[ns_req_idx++]);
        }
        /* Send my top boundary */
        float *top_send = (float *)malloc((size_t)HH * (size_t)my_block_W * sizeof(float));
        for (int i = 0; i < edge_rows; i++)
            for (int j = 0; j < my_block_W; j++)
                top_send[i * my_block_W + j] = F_local[IDX2(i, j, my_block_W)];
#ifdef DEBUG_HALO
        fprintf(stderr, "[%d] Send to NORTH (%d): %d rows from local rows [0:%d]\n",
                rank, north, edge_rows, edge_rows-1);
#endif
        /* Send rows separately to match receive */
        for (int i = 0; i < edge_rows; i++) {
            MPI_Isend(&top_send[i * my_block_W], my_block_W, MPI_FLOAT, 
                      north, 201 + i, comm, &ns_reqs[ns_req_idx++]);
        }
        free(top_send);
    }

    if (south != MPI_PROC_NULL) {
        /* Recv south halo - only as many rows as neighbor can send */
        for (int i = 0; i < south_halo_rows; i++) {
            MPI_Irecv(&buf[IDX2(HH + my_block_H + i, HW, buf_W)], my_block_W, MPI_FLOAT, 
                      south, 201 + i, comm, &ns_reqs[ns_req_idx++]);
        }
        /* Send my bottom boundary */
        float *bot_send = (float *)malloc((size_t)HH * (size_t)my_block_W * sizeof(float));
        int send_start = (my_block_H >= HH ? my_block_H - HH : 0);
        for (int i = 0; i < edge_rows; i++)
            for (int j = 0; j < my_block_W; j++)
                bot_send[i * my_block_W + j] = F_local[IDX2(send_start + i, j, my_block_W)];
#ifdef DEBUG_HALO
        fprintf(stderr, "[%d] Send to SOUTH (%d): %d rows from local rows [%d:%d], first 4: %.3f %.3f %.3f %.3f\n",
                rank, south, edge_rows, send_start, send_start + edge_rows - 1,
                bot_send[0], bot_send[1], bot_send[2], bot_send[3]);
#endif
        /* Send rows separately to match receive */
        for (int i = 0; i < edge_rows; i++) {
            MPI_Isend(&bot_send[i * my_block_W], my_block_W, MPI_FLOAT, 
                      south, 202 + i, comm, &ns_reqs[ns_req_idx++]);
        }
        free(bot_send);
    }

    /* Wait for N/S exchange to complete before W/E (needed for corners) */
    MPI_Waitall(ns_req_idx, ns_reqs, MPI_STATUSES_IGNORE);

#ifdef DEBUG_HALO
    /* Debug: print north halo (first HH rows) - should match ranks[my_row_idx-1]'s bottom rows */
    if (north != MPI_PROC_NULL && rank == 1) {
        fprintf(stderr, "[1] North halo after N/S exchange (should be rank 0's rows 2-3):\n");
        for (int i = 0; i < HH; i++) {
            fprintf(stderr, "  halo row %d: ", i);
            for (int j = 0; j < my_block_W && j < 8; j++) {
                fprintf(stderr, "%.3f ", buf[IDX2(i, HW + j, buf_W)]);
            }
            fprintf(stderr, "\n");
        }
    }
    /* Debug: print my local data (center of buffer) */
    for (int r = 0; r < size; r++) {
        MPI_Barrier(comm);
        if (rank == r) {
            fprintf(stderr, "[%d] F_local (my data, global rows [%d:%d]):\n",
                    rank, my_H_start, my_H_start + my_block_H - 1);
            for (int i = 0; i < my_block_H; i++) {
                fprintf(stderr, "  row %d: ", my_H_start + i);
                for (int j = 0; j < my_block_W && j < 8; j++) {
                    fprintf(stderr, "%.3f ", F_local[IDX2(i, j, my_block_W)]);
                }
                fprintf(stderr, "\n");
            }
        }
    }
#endif

    /* ===== PHASE 2: WEST / EAST (using derived datatypes - no pack/unpack) ===== */
    MPI_Request ew_reqs[4];
    int ew_req_idx = 0;
    
    /* Create column vector types for W/E halos to eliminate packing overhead */
    MPI_Datatype west_recv_type, east_recv_type, west_send_type, east_send_type;
    
    if (west != MPI_PROC_NULL) {
        /* Recv west halo directly into buffer using strided vector type */
        MPI_Type_vector(buf_H, west_halo_cols, buf_W, MPI_FLOAT, &west_recv_type);
        MPI_Type_commit(&west_recv_type);
        MPI_Irecv(&buf[IDX2(0, 0, buf_W)], 1, west_recv_type, 
                  west, 204, comm, &ew_reqs[ew_req_idx++]);
        
        /* Send west boundary directly from buffer using strided vector type */
        MPI_Type_vector(buf_H, edge_cols, buf_W, MPI_FLOAT, &west_send_type);
        MPI_Type_commit(&west_send_type);
        MPI_Isend(&buf[IDX2(0, HW, buf_W)], 1, west_send_type, 
                  west, 203, comm, &ew_reqs[ew_req_idx++]);
    }

    if (east != MPI_PROC_NULL) {
        /* Recv east halo directly into buffer using strided vector type */
        MPI_Type_vector(buf_H, east_halo_cols, buf_W, MPI_FLOAT, &east_recv_type);
        MPI_Type_commit(&east_recv_type);
        MPI_Irecv(&buf[IDX2(0, HW + my_block_W, buf_W)], 1, east_recv_type, 
                  east, 203, comm, &ew_reqs[ew_req_idx++]);
        
        /* Send east boundary directly from buffer using strided vector type */
        int send_col_start = HW + my_block_W - edge_cols;
        MPI_Type_vector(buf_H, edge_cols, buf_W, MPI_FLOAT, &east_send_type);
        MPI_Type_commit(&east_send_type);
        MPI_Isend(&buf[IDX2(0, send_col_start, buf_W)], 1, east_send_type, 
                  east, 204, comm, &ew_reqs[ew_req_idx++]);
    }

    MPI_Waitall(ew_req_idx, ew_reqs, MPI_STATUSES_IGNORE);

    /* Free derived types */
    if (west != MPI_PROC_NULL) {
        MPI_Type_free(&west_recv_type);
        MPI_Type_free(&west_send_type);
    }
    if (east != MPI_PROC_NULL) {
        MPI_Type_free(&east_recv_type);
        MPI_Type_free(&east_send_type);
    }

    /* Global output dimensions */
    const int fullH = H + KH - 1;
    const int fullW = W + KW - 1;
    const int outH = (cmode == MODE_FULL) ? ceil_div(fullH, sH) : ceil_div(H, sH);
    const int outW = (cmode == MODE_FULL) ? ceil_div(fullW, sW) : ceil_div(W, sW);

    /* Calculate my output block dimensions (based on my input block) */
    const int my_fullH = my_block_H + KH - 1;
    const int my_fullW = my_block_W + KW - 1;
    const int my_out_H = (cmode == MODE_FULL) ? ceil_div(my_fullH, sH) : ceil_div(my_block_H, sH);
    const int my_out_W = (cmode == MODE_FULL) ? ceil_div(my_fullW, sW) : ceil_div(my_block_W, sW);
    
    /* Calculate my output block's starting position in global output */
    int my_out_H_start = 0;
    int my_out_W_start = 0;
    
    if (cmode == MODE_FULL) {
        /* In FULL mode with stride, output indices are sampled at 0, sH, 2*sH, ...
         * so the starting output offset must be floored by the stride */
        my_out_H_start = my_H_start / sH;   /* floor division */
        my_out_W_start = my_W_start / sW;   /* floor division */
    } else {
        /* In SAME mode, cumulative sum of output block sizes */
        for (int r = 0; r < my_row_idx; r++) {
            int block_h_r = base_rows + (r < rem_rows ? 1 : 0);
            int out_h_r = ceil_div(block_h_r, sH);
            my_out_H_start += out_h_r;
        }
        
        for (int c = 0; c < my_col_idx; c++) {
            int block_w_c = base_cols + (c < rem_cols ? 1 : 0);
            int out_w_c = ceil_div(block_w_c, sW);
            my_out_W_start += out_w_c;
        }
    }

#ifdef DEBUG_MPI
    for (int r = 0; r < size; r++)
    {
        MPI_Barrier(comm);
        if (rank == r)
        {
            fprintf(stderr,
                    "[rank %d/%d] coords=(%d,%d) %s: in_block=[%d:%d, %d:%d] (%dx%d) | "
                    "out_block=[%d:%d, %d:%d] (%dx%d) | HH=%d HW=%d | gen=%s\n",
                    rank, size, my_row_idx, my_col_idx, 
                    (cmode == MODE_FULL ? "FULL" : "SAME"),
                    my_H_start, my_H_start + my_block_H - 1,
                    my_W_start, my_W_start + my_block_W - 1,
                    my_block_H, my_block_W,
                    my_out_H_start, my_out_H_start + my_out_H - 1,
                    my_out_W_start, my_out_W_start + my_out_W - 1,
                    my_out_H, my_out_W,
                    HH, HW, (parallel_gen ? "parallel" : "root"));
            fflush(stderr);
        }
    }
    MPI_Barrier(comm);
#endif

    float *Y_local = (float *)calloc((size_t)my_out_H * (size_t)my_out_W, sizeof(float));
    if (!Y_local)
    {
        fprintf(stderr, "[%d] OOM Y_local\n", rank);
        MPI_Abort(comm, 1);
    }

    /* Kernel timing */
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    if (cmode == MODE_SAME)
    {
        /* Determine anchor based on operation */
        const int cH = (op == OP_CONV) ? (KH / 2) : ((KH - 1) / 2);
        const int cW = (op == OP_CONV) ? (KW / 2) : ((KW - 1) / 2);

        for (int i_out = 0; i_out < my_out_H; i_out++)
        {
            int gi = (my_out_H_start + i_out) * sH; /* global row in F */
            for (int j_out = 0; j_out < my_out_W; j_out++)
            {
                int gj = (my_out_W_start + j_out) * sW; /* global col in F */
                double acc = 0.0;

                for (int u = 0; u < KH; u++)
                {
                    for (int v = 0; v < KW; v++)
                    {
                        int fi = gi + (u - cH);
                        int fj = gj + (v - cW);

                        if (fi >= 0 && fi < H && fj >= 0 && fj < W)
                        {
                            /* Map to local buffer coordinates (with halos) */
                            int local_i = fi - my_H_start;
                            int local_j = fj - my_W_start;
                            if (local_i >= -HH && local_i < my_block_H + HH &&
                                local_j >= -HW && local_j < my_block_W + HW)
                            {
                                int buf_i = local_i + HH;
                                int buf_j = local_j + HW;
                                acc += (double)buf[IDX2(buf_i, buf_j, buf_W)] * (double)G_use[IDX2(u, v, KW)];
                            }
                        }
                        else if (pmode == PAD_CONST)
                        {
                            acc += (double)cval * (double)G_use[IDX2(u, v, KW)];
                        }
                        /* PAD_NONE and PAD_ZERO: out-of-range contributes 0 */
                    }
                }
                Y_local[IDX2(i_out, j_out, my_out_W)] = (float)acc;
            }
        }
    }
    else /* MODE_FULL */
    {
        for (int i_out = 0; i_out < my_out_H; i_out++)
        {
            int oi = (my_out_H_start + i_out) * sH; /* global output index */
            for (int j_out = 0; j_out < my_out_W; j_out++)
            {
                int oj = (my_out_W_start + j_out) * sW;
                double acc = 0.0;

                /* Determine valid overlap range */
                int i0 = (oi < KH - 1) ? 0 : (oi - (KH - 1));
                int i1 = (oi < H) ? oi : (H - 1);
                int j0 = (oj < KW - 1) ? 0 : (oj - (KW - 1));
                int j1 = (oj < W) ? oj : (W - 1);

                for (int i = i0; i <= i1; i++)
                {
                    for (int j = j0; j <= j1; j++)
                    {
                        int u = oi - i;
                        int v = oj - j;
                        if (u >= 0 && u < KH && v >= 0 && v < KW)
                        {
                            /* Map to local buffer (with halos) */
                            int local_i = i - my_H_start;
                            int local_j = j - my_W_start;
                            if (local_i >= -HH && local_i < my_block_H + HH &&
                                local_j >= -HW && local_j < my_block_W + HW)
                            {
                                int buf_i = local_i + HH;
                                int buf_j = local_j + HW;
                                acc += (double)buf[IDX2(buf_i, buf_j, buf_W)] * (double)G_use[IDX2(u, v, KW)];
                            }
                        }
                    }
                }
                Y_local[IDX2(i_out, j_out, my_out_W)] = (float)acc;
            }
        }
    }

    double local_secs = MPI_Wtime() - t0;

#ifdef DEBUG_MPI
    fprintf(stderr, "[rank %d/%d] kernel_time=%.9f s (out_cnt=%d)\n",
            rank, size, local_secs, my_out_H * my_out_W);
#endif

    /* Performance reductions */
    double local_flops = 2.0 * (double)KH * (double)KW * (double)(my_out_H * my_out_W);
    double sum_flops = 0.0, max_secs = 0.0, min_secs = 0.0, sum_secs = 0.0;
    MPI_Reduce(&local_flops, &sum_flops, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_secs, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_secs, &min_secs, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_secs, &sum_secs, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    /* Pre-round to 3 decimal places */
    for (int i = 0; i < my_out_H * my_out_W; i++)
    {
        double scaled = Y_local[i] * 1000.0;
        double rounded = nearbyint(scaled);
        Y_local[i] = (float)(rounded * 0.001);
    }

    /* Output */
    if (text_mode)
    {
        /* Manual 2D block gather to root */
        float *Y_root = NULL;
        if (rank == 0)
        {
            Y_root = (float *)calloc((size_t)outH * (size_t)outW, sizeof(float));
            if (!Y_root)
            {
                fprintf(stderr, "[root] OOM Y_root\n");
                MPI_Abort(comm, 1);
            }
            
            /* Receive blocks from all ranks (including self) */
            for (int src_rank = 0; src_rank < size; src_rank++)
            {
                /* Calculate src rank's output block info */
                int src_coords[2];
                MPI_Cart_coords(comm, src_rank, 2, src_coords);
                int src_row_idx = src_coords[0];
                int src_col_idx = src_coords[1];
                
                /* Calculate src's input block */
                int src_block_H = base_rows + (src_row_idx < rem_rows ? 1 : 0);
                int src_block_W = base_cols + (src_col_idx < rem_cols ? 1 : 0);
                
                /* Calculate src's output block */
                int src_fullH = src_block_H + KH - 1;
                int src_fullW = src_block_W + KW - 1;
                int src_out_H = (cmode == MODE_FULL) ? ceil_div(src_fullH, sH) : ceil_div(src_block_H, sH);
                int src_out_W = (cmode == MODE_FULL) ? ceil_div(src_fullW, sW) : ceil_div(src_block_W, sW);
                
                /* Calculate src's output starting position */
                int src_out_H_start = 0;
                int src_out_W_start = 0;
                
                if (cmode == MODE_FULL) {
                    /* In FULL mode, calculate input starting position */
                    for (int r = 0; r < src_row_idx; r++) {
                        int bh = base_rows + (r < rem_rows ? 1 : 0);
                        src_out_H_start += bh;
                    }
                    for (int c = 0; c < src_col_idx; c++) {
                        int bw = base_cols + (c < rem_cols ? 1 : 0);
                        src_out_W_start += bw;
                    }
                } else {
                    /* In SAME mode, cumulative sum of output sizes */
                    for (int r = 0; r < src_row_idx; r++) {
                        int bh = base_rows + (r < rem_rows ? 1 : 0);
                        int oh = ceil_div(bh, sH);
                        src_out_H_start += oh;
                    }
                    for (int c = 0; c < src_col_idx; c++) {
                        int bw = base_cols + (c < rem_cols ? 1 : 0);
                        int ow = ceil_div(bw, sW);
                        src_out_W_start += ow;
                    }
                }
                
                float *block = (float *)malloc((size_t)src_out_H * (size_t)src_out_W * sizeof(float));
                if (!block) {
                    fprintf(stderr, "[root] OOM gather block\n");
                    MPI_Abort(comm, 1);
                }
                
                if (src_rank == 0) {
                    /* Copy from self */
                    memcpy(block, Y_local, (size_t)src_out_H * (size_t)src_out_W * sizeof(float));
                } else {
                    /* Receive from other rank */
                    MPI_Recv(block, src_out_H * src_out_W, MPI_FLOAT,
                             src_rank, 101, comm, MPI_STATUS_IGNORE);
                }
                
                /* Unpack block into Y_root */
                for (int i = 0; i < src_out_H; i++) {
                    for (int j = 0; j < src_out_W; j++) {
                        Y_root[(src_out_H_start + i) * outW + (src_out_W_start + j)] = 
                            block[i * src_out_W + j];
                    }
                }
                free(block);
            }
        }
        else
        {
            /* Non-root sends its block */
            MPI_Send(Y_local, my_out_H * my_out_W, MPI_FLOAT,
                     0, 101, comm);
        }

        if (rank == 0)
        {
            write_matrix_2d_txt(out_path, Y_root, outH, outW);
            double gflops = (max_secs > 0.0) ? (sum_flops / (max_secs * 1e9)) : 0.0;
            double avg_secs = sum_secs / (double)size;
            fprintf(stderr,
                    "H=%d W=%d KH=%d KW=%d outH=%d outW=%d op=%s mode=%s pad=%s cval=%.6g sH=%d sW=%d | "
                    "ranks=%d (2D: %dx%d) | conv_time=%.9f s | %.3f GFLOP/s | perRank(min/avg/max)=%.9f/%.9f/%.9f s\n",
                    H, W, KH, KW, outH, outW,
                    (op == OP_CONV ? "conv" : "corr"),
                    (cmode == MODE_FULL ? "full" : "same"),
                    (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                    (pmode == PAD_CONST ? cval : 0.0),
                    sH, sW, size, P_rows, P_cols, max_secs, gflops, min_secs, avg_secs, max_secs);
            log_metrics(H, W, KH, KW, outH, outW, op, cmode, pmode, cval, sH, sW, max_secs, gflops,
                       size, P_rows, P_cols);
            free(Y_root);
        }
    }
    else
    {
        /* Binary MPI-IO with 2D subarray */
        MPI_File fh;
        int mpierr = MPI_File_open(comm, (char *)out_path,
                                   MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                   MPI_INFO_NULL, &fh);
        if (mpierr != MPI_SUCCESS)
        {
            if (rank == 0)
                fprintf(stderr, "MPI_File_open failed\n");
            MPI_Abort(comm, 1);
        }

        /* Write header (outH, outW) */
        if (rank == 0)
        {
            int header[2] = {outH, outW};
            MPI_Status st;
            MPI_File_write_at(fh, 0, header, 2, MPI_INT, &st);
        }
        MPI_Barrier(comm);

        /* Create subarray datatype for my output block */
        int global_out_sizes[2] = {outH, outW};
        int local_out_sizes[2] = {my_out_H, my_out_W};
        int out_starts[2] = {my_out_H_start, my_out_W_start};
        
        MPI_Datatype out_file_type;
        MPI_Type_create_subarray(2, global_out_sizes, local_out_sizes, out_starts,
                                 MPI_ORDER_C, MPI_FLOAT, &out_file_type);
        MPI_Type_commit(&out_file_type);

        /* Set file view (skip 8-byte header) */
        MPI_Offset header_bytes = 2 * sizeof(int);
        MPI_File_set_view(fh, header_bytes, MPI_FLOAT, out_file_type, "native", MPI_INFO_NULL);

        /* Collective write */
        MPI_Status st;
        MPI_File_write_all(fh, Y_local, my_out_H * my_out_W, MPI_FLOAT, &st);

        MPI_Type_free(&out_file_type);
        MPI_File_close(&fh);

        if (rank == 0)
        {
            double gflops = (max_secs > 0.0) ? (sum_flops / (max_secs * 1e9)) : 0.0;
            double avg_secs = sum_secs / (double)size;
            fprintf(stderr,
                    "H=%d W=%d KH=%d KW=%d outH=%d outW=%d op=%s mode=%s pad=%s cval=%.6g sH=%d sW=%d | "
                    "ranks=%d (2D: %dx%d) | conv_time=%.9f s | %.3f GFLOP/s | perRank(min/avg/max)=%.9f/%.9f/%.9f s\n",
                    H, W, KH, KW, outH, outW,
                    (op == OP_CONV ? "conv" : "corr"),
                    (cmode == MODE_FULL ? "full" : "same"),
                    (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                    (pmode == PAD_CONST ? cval : 0.0),
                    sH, sW, size, P_rows, P_cols, max_secs, gflops, min_secs, avg_secs, max_secs);
            log_metrics(H, W, KH, KW, outH, outW, op, cmode, pmode, cval, sH, sW, max_secs, gflops,
                       size, P_rows, P_cols);
        }
    }

    /* Cleanup */
    free(F_local);
    free(buf);
    free(G);
    if (G_flip)
        free(G_flip);
    free(Y_local);

    if (comm != MPI_COMM_WORLD)
        MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}
