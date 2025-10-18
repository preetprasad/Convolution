/*------------------------------------------------------------------------------
  conv2d_mpi.c — Pure-MPI 2-D convolution/correlation with 2-D domain decomposition

  OVERVIEW
    Computes Y = F * G (convolution) or Y = F ⊗ G (correlation) in 2-D using
    MPI with a 2-D Cartesian topology and halo exchange. Supports SAME/FULL
    modes, arbitrary stride (sH, sW), selectable padding, and both file and
    RNG inputs. Communication overlaps with computation via non-blocking halo
    exchange. Timing measures ONLY the arithmetic kernel. Output can be
    human-readable text or binary (default).

  KEY FEATURES (aligned with conv1d_mpi.c and conv2d.c)
    • Operations:   --conv (default) or --corr
    • Modes:        SAME (centered kernel), FULL (full convolution)
    • Padding:      zero | none | const (with -c/--cval)
    • Stride:       -sH/--stride-h M, -sW/--stride-w N (sample every M×N)
    • Inputs:
        - Text  .txt  (assignment format: "H W" header + H×W floats)
        - Binary .bin (header int32 H, int32 W, followed by H×W float32)
        - RNG   -H/-W/-kH/-kW with -se/--seed
        - Optional --parallel-gen: each rank deterministically generates its
          slice of F by global index; G is generated on root and broadcast.
          (Default: root generates full input and scatters to ranks.)
    • Topology:     2-D Cartesian MPI grid (row-based decomposition for simplicity)
    • Halo exchange: Non-blocking ghost-cell exchange with MPI_Irecv/Isend/Waitall.
      Halo width: HH = KH-1 (top/bottom).
    • Timing:       kernel-only via MPI_Wtime; min/avg/max per-rank reported.
    • Performance:  GFLOP/s computed using Σ(2*KH*KW*local_out_count)/max(time).
    • Output:
        - Binary (default): [int32 outH][int32 outW] + outH×outW float32 (rounded 3dp)
        - Text   (--text):  assignment style, rounded to 3 dp
      3-decimal rounding applied before writing for consistency.
    • MPI-IO:       Collective parallel write of each rank's output slice using
      explicit byte offsets and MPI_Type_contiguous().
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
      halo width, generation mode, kernel time.

  NOTES / LIMITATIONS
    - Row-based decomposition (each rank owns contiguous rows of F)
    - Halo exchange only for top/bottom boundaries (vertical)
    - Padding applied in inner loop (no fabricated halos for const padding)
    - This implementation prioritizes clarity and lecture alignment

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
                 double elapsed_secs, double gflops);

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
                 double elapsed_secs, double gflops)
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
    fprintf(csv, "RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,sH,sW,time,gflops\n");
    fprintf(csv, "%s,%d,%d,%d,%d,%d,%d,%s,%s,%s,%.9g,%d,%d,%.9f,%.6f\n",
            runid, H, W, KH, KW, outH, outW,
            (op == OP_CONV ? "conv" : "corr"),
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            sH, sW, elapsed_secs, gflops);
    fclose(csv);
}

/* ------------------- Main ------------------- */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Create 1-D Cartesian communicator (row-based decomposition) */
    MPI_Comm comm;
    {
        int dims[1] = {world_size};
        int periods[1] = {0};
        int reorder = 0;
        MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &comm);
        if (comm == MPI_COMM_NULL)
            comm = MPI_COMM_WORLD;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Find top/bottom neighbors */
    int top = MPI_PROC_NULL, bottom = MPI_PROC_NULL;
    MPI_Cart_shift(comm, 0, 1, &top, &bottom);

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

    /* Row decomposition */
    int base_rows = H / size;
    int rem_rows = H % size;
    int my_row_start = rank * base_rows + (rank < rem_rows ? rank : rem_rows);
    int my_row_count = base_rows + (rank < rem_rows ? 1 : 0);

    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc((size_t)size * sizeof(int));
        displs = (int *)malloc((size_t)size * sizeof(int));
        int off = 0;
        for (int r = 0; r < size; r++)
        {
            int cnt = base_rows + (r < rem_rows ? 1 : 0);
            sendcounts[r] = cnt * W;
            displs[r] = off;
            off += cnt * W;
        }
    }

    float *F_local = (float *)malloc((size_t)my_row_count * (size_t)W * sizeof(float));
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

        /* Read my rows: header is 2*int32, then H*W float32 row-major */
        MPI_Offset base = (MPI_Offset)(2 * sizeof(int));
        MPI_Offset my_offset = base + (MPI_Offset)my_row_start * (MPI_Offset)W * (MPI_Offset)sizeof(float);
        MPI_Datatype seg;
        MPI_Type_contiguous(my_row_count * W, MPI_FLOAT, &seg);
        MPI_Type_commit(&seg);
        MPI_File_read_at_all(fh, my_offset, F_local, 1, seg, &st);
        MPI_Type_free(&seg);
        MPI_File_close(&fh);
    }
    else if (parallel_gen)
    {
        /* Parallel generation using indexable PRNG */
        for (int li = 0; li < my_row_count; li++)
            for (int j = 0; j < W; j++)
            {
                long long gi = (long long)(my_row_start + li);
                long long gj = (long long)j;
                F_local[IDX2(li, j, W)] = gen_value_at_index_2d(seed, gi, gj);
            }
    }
    else
    {
        /* Scatter from root */
        MPI_Scatterv(F_root, sendcounts, displs, MPI_FLOAT,
                     F_local, my_row_count * W, MPI_FLOAT,
                     0, comm);
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

    /* Halo exchange: top/bottom */
    const int HH = (KH > 0 ? KH - 1 : 0);
    const int buf_rows = my_row_count + 2 * HH;
    float *buf = (float *)calloc((size_t)buf_rows * (size_t)W, sizeof(float));
    if (!buf)
    {
        fprintf(stderr, "[%d] OOM buf\n", rank);
        MPI_Abort(comm, 1);
    }

    /* Copy local data into buffer center */
    for (int li = 0; li < my_row_count; li++)
        memcpy(&buf[IDX2(li + HH, 0, W)], &F_local[IDX2(li, 0, W)], (size_t)W * sizeof(float));

    /* Non-blocking halo exchange */
    int edge_rows = (HH <= my_row_count ? HH : my_row_count);
    MPI_Request reqs[4];

    /* Top halo: recv our top ghost, send our top boundary */
    MPI_Irecv(&buf[IDX2(0, 0, W)], HH * W, MPI_FLOAT, top, 102, comm, &reqs[0]);
    MPI_Isend(&F_local[IDX2(0, 0, W)], edge_rows * W, MPI_FLOAT, top, 101, comm, &reqs[1]);

    /* Bottom halo: recv our bottom ghost, send our bottom boundary */
    MPI_Irecv(&buf[IDX2(HH + my_row_count, 0, W)], HH * W, MPI_FLOAT, bottom, 101, comm, &reqs[2]);
    int send_start = (my_row_count >= HH ? my_row_count - HH : 0);
    MPI_Isend(&F_local[IDX2(send_start, 0, W)], edge_rows * W, MPI_FLOAT, bottom, 102, comm, &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    /* Output dimensions */
    const int fullH = H + KH - 1;
    const int fullW = W + KW - 1;
    const int outH = (cmode == MODE_FULL) ? ceil_div(fullH, sH) : ceil_div(H, sH);
    const int outW = (cmode == MODE_FULL) ? ceil_div(fullW, sW) : ceil_div(W, sW);

    /* Decompose output rows across ranks */
    int base_out_rows = outH / size;
    int rem_out_rows = outH % size;
    int my_out_row_start = rank * base_out_rows + (rank < rem_out_rows ? rank : rem_out_rows);
    int my_out_row_count = base_out_rows + (rank < rem_out_rows ? 1 : 0);

#ifdef DEBUG_MPI
    for (int r = 0; r < size; r++)
    {
        MPI_Barrier(comm);
        if (rank == r)
        {
            fprintf(stderr,
                    "[rank %d/%d] %s: H=%d W=%d KH=%d KW=%d sH=%d sW=%d | "
                    "f_rows=[%d..%d) (cnt=%d) | out_rows=[%d..%d) (cnt=%d) | HH=%d | gen=%s\n",
                    rank, size, (cmode == MODE_FULL ? "FULL" : "SAME"),
                    H, W, KH, KW, sH, sW,
                    my_row_start, my_row_start + my_row_count, my_row_count,
                    my_out_row_start, my_out_row_start + my_out_row_count, my_out_row_count,
                    HH, (parallel_gen ? "parallel" : "root"));
            fflush(stderr);
        }
    }
    MPI_Barrier(comm);
#endif

    float *Y_local = (float *)calloc((size_t)my_out_row_count * (size_t)outW, sizeof(float));
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

        for (int i_out = 0; i_out < my_out_row_count; i_out++)
        {
            int gi = (my_out_row_start + i_out) * sH; /* global row in F */
            for (int j_out = 0; j_out < outW; j_out++)
            {
                int gj = j_out * sW; /* global col in F */
                double acc = 0.0;

                for (int u = 0; u < KH; u++)
                {
                    for (int v = 0; v < KW; v++)
                    {
                        int fi = gi + (u - cH);
                        int fj = gj + (v - cW);

                        if (fi >= 0 && fi < H && fj >= 0 && fj < W)
                        {
                            /* Map to local buffer coordinates */
                            int local_i = fi - my_row_start;
                            if (local_i >= -HH && local_i < my_row_count + HH)
                            {
                                int buf_i = local_i + HH;
                                acc += (double)buf[IDX2(buf_i, fj, W)] * (double)G_use[IDX2(u, v, KW)];
                            }
                        }
                        else if (pmode == PAD_CONST)
                        {
                            acc += (double)cval * (double)G_use[IDX2(u, v, KW)];
                        }
                        /* PAD_NONE and PAD_ZERO: out-of-range contributes 0 */
                    }
                }
                Y_local[IDX2(i_out, j_out, outW)] = (float)acc;
            }
        }
    }
    else /* MODE_FULL */
    {
        for (int i_out = 0; i_out < my_out_row_count; i_out++)
        {
            int oi = (my_out_row_start + i_out) * sH; /* global output index */
            for (int j_out = 0; j_out < outW; j_out++)
            {
                int oj = j_out * sW;
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
                            /* Map to local buffer */
                            int local_i = i - my_row_start;
                            if (local_i >= -HH && local_i < my_row_count + HH)
                            {
                                int buf_i = local_i + HH;
                                acc += (double)buf[IDX2(buf_i, j, W)] * (double)G_use[IDX2(u, v, KW)];
                            }
                        }
                    }
                }
                Y_local[IDX2(i_out, j_out, outW)] = (float)acc;
            }
        }
    }

    double local_secs = MPI_Wtime() - t0;

#ifdef DEBUG_MPI
    fprintf(stderr, "[rank %d/%d] kernel_time=%.9f s (out_cnt=%d)\n",
            rank, size, local_secs, my_out_row_count * outW);
#endif

    /* Performance reductions */
    double local_flops = 2.0 * (double)KH * (double)KW * (double)(my_out_row_count * outW);
    double sum_flops = 0.0, max_secs = 0.0, min_secs = 0.0, sum_secs = 0.0;
    MPI_Reduce(&local_flops, &sum_flops, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_secs, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_secs, &min_secs, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_secs, &sum_secs, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    /* Pre-round to 3 decimal places */
    for (int i = 0; i < my_out_row_count * outW; i++)
    {
        double scaled = Y_local[i] * 1000.0;
        double rounded = nearbyint(scaled);
        Y_local[i] = (float)(rounded * 0.001);
    }

    /* Output */
    if (text_mode)
    {
        /* Gather all output to root and write text */
        int *recvcounts = NULL, *rdispls = NULL;
        if (rank == 0)
        {
            recvcounts = (int *)malloc((size_t)size * sizeof(int));
            rdispls = (int *)malloc((size_t)size * sizeof(int));
            int off = 0;
            for (int r = 0; r < size; r++)
            {
                int cnt = base_out_rows + (r < rem_out_rows ? 1 : 0);
                recvcounts[r] = cnt * outW;
                rdispls[r] = off;
                off += cnt * outW;
            }
        }

        float *Y_root = NULL;
        if (rank == 0)
        {
            Y_root = (float *)malloc((size_t)outH * (size_t)outW * sizeof(float));
            if (!Y_root)
            {
                fprintf(stderr, "[root] OOM Y_root\n");
                MPI_Abort(comm, 1);
            }
        }

        MPI_Gatherv(Y_local, my_out_row_count * outW, MPI_FLOAT,
                    Y_root, recvcounts, rdispls, MPI_FLOAT,
                    0, comm);

        if (rank == 0)
        {
            write_matrix_2d_txt(out_path, Y_root, outH, outW);
            double gflops = (max_secs > 0.0) ? (sum_flops / (max_secs * 1e9)) : 0.0;
            double avg_secs = sum_secs / (double)size;
            fprintf(stderr,
                    "H=%d W=%d KH=%d KW=%d outH=%d outW=%d op=%s mode=%s pad=%s cval=%.6g sH=%d sW=%d | "
                    "ranks=%d | conv_time=%.9f s | %.3f GFLOP/s | perRank(min/avg/max)=%.9f/%.9f/%.9f s\n",
                    H, W, KH, KW, outH, outW,
                    (op == OP_CONV ? "conv" : "corr"),
                    (cmode == MODE_FULL ? "full" : "same"),
                    (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                    (pmode == PAD_CONST ? cval : 0.0),
                    sH, sW, size, max_secs, gflops, min_secs, avg_secs, max_secs);
            log_metrics(H, W, KH, KW, outH, outW, op, cmode, pmode, cval, sH, sW, max_secs, gflops);
            free(Y_root);
            free(recvcounts);
            free(rdispls);
        }
    }
    else
    {
        /* Binary MPI-IO */
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

        /* Compute byte offsets */
        int *recvcounts = NULL, *rdispls = NULL;
        if (rank == 0)
        {
            recvcounts = (int *)malloc((size_t)size * sizeof(int));
            rdispls = (int *)malloc((size_t)size * sizeof(int));
            int off = 0;
            for (int r = 0; r < size; r++)
            {
                int cnt = base_out_rows + (r < rem_out_rows ? 1 : 0);
                recvcounts[r] = cnt * outW;
                rdispls[r] = off;
                off += cnt * outW;
            }
        }

        int my_disp = 0;
        if (rank != 0)
        {
            MPI_Recv(&my_disp, 1, MPI_INT, 0, 900 + rank, comm, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int r = 1; r < size; r++)
                MPI_Send(&rdispls[r], 1, MPI_INT, r, 900 + r, comm);
        }

        MPI_Offset header_bytes = 2 * (MPI_Offset)sizeof(int);
        MPI_Offset my_byte_offset = header_bytes + (MPI_Offset)my_disp * (MPI_Offset)sizeof(float);

        MPI_Datatype segtype;
        MPI_Type_contiguous(my_out_row_count * outW, MPI_FLOAT, &segtype);
        MPI_Type_commit(&segtype);

        MPI_Status st;
        MPI_File_write_at_all(fh, my_byte_offset, Y_local, 1, segtype, &st);

        MPI_Type_free(&segtype);
        MPI_File_close(&fh);

        if (rank == 0)
        {
            double gflops = (max_secs > 0.0) ? (sum_flops / (max_secs * 1e9)) : 0.0;
            double avg_secs = sum_secs / (double)size;
            fprintf(stderr,
                    "H=%d W=%d KH=%d KW=%d outH=%d outW=%d op=%s mode=%s pad=%s cval=%.6g sH=%d sW=%d | "
                    "ranks=%d | conv_time=%.9f s | %.3f GFLOP/s | perRank(min/avg/max)=%.9f/%.9f/%.9f s\n",
                    H, W, KH, KW, outH, outW,
                    (op == OP_CONV ? "conv" : "corr"),
                    (cmode == MODE_FULL ? "full" : "same"),
                    (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                    (pmode == PAD_CONST ? cval : 0.0),
                    sH, sW, size, max_secs, gflops, min_secs, avg_secs, max_secs);
            log_metrics(H, W, KH, KW, outH, outW, op, cmode, pmode, cval, sH, sW, max_secs, gflops);
            free(recvcounts);
            free(rdispls);
        }
    }

    /* Cleanup */
    free(F_local);
    free(buf);
    free(G);
    if (G_flip)
        free(G_flip);
    free(Y_local);
    if (sendcounts)
        free(sendcounts);
    if (displs)
        free(displs);

    if (comm != MPI_COMM_WORLD)
        MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}
