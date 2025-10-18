/*------------------------------------------------------------------------------
  conv1d_mpi.c — Pure-MPI 1-D convolution with true halo exchange

  OVERVIEW
    Computes y = f * g along one dimension using MPI with a halo-exchange
    pattern. Supports SAME (centered) and FULL modes, arbitrary stride,
    selectable padding, and both file and RNG inputs. Communication overlaps
    with computation via non-blocking halo exchange. Timing measures ONLY the
    arithmetic kernel. Output can be human-readable text or binary (default).

  KEY FEATURES (lecture-aligned)
    • Modes:        SAME (centered kernel), FULL (linear convolution)
    • Padding:      zero | none | const (with -c/--cval)
    • Stride:       -st/--stride N  (sample every N in output index space)
    • Inputs:
        - Text  .txt  (assignment format: first line L, second line L floats)
        - Binary .bin (header int32 L, followed by L float32)
        - RNG   -L/-kL with -se/--seed (legacy -s supported)
        - Optional --parallel-gen: each rank deterministically generates its
          slice of f by global index; g is generated on root and broadcast.
          (Default: root generates full input and scatters to ranks.)
    • Halo exchange: true two-sided ghost-cell exchange with MPI_Irecv / Isend
      + MPI_Waitall. Halo width H = K-1.
    • Timing:       kernel-only via MPI_Wtime; min/avg/max per-rank reported.
    • Performance:  GFLOP/s computed using Σ(2*K*local_out_count)/max(time).
    • Output:
        - Binary (default): [int32 outLen] + outLen float32 (rounded to 3 dp)
        - Text   (--text):  assignment style, rounded to 3 dp
      3-decimal rounding is applied before writing so text and binary compare
      identically at the printed precision.
    • MPI-IO:       Collective parallel write of each rank’s output slice using
      explicit byte offsets and MPI_Type_contiguous().
    • Metrics:      Per-run CSV at metrics/metrics_<RUNID>.csv containing
      RunID, N, K, outLen, mode, padding, cval, time(s), GFLOP/s.

  BUILD
      mpicc -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c
      # with debug prints:
      mpicc -DDEBUG_MPI=1 -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c

  TYPICAL RUNS
    # Binary output (default):
    mpirun -np 4 ./conv1d_mpi -L 1024 -kL 5 -m same -st 2 -se 42 -o y.bin

    # Human-readable text output:
    mpirun -np 4 ./conv1d_mpi -L 1024 -kL 5 -m same -st 2 -se 42 --text -o y.txt

    # Input from files:
    mpirun -np 4 ./conv1d_mpi -f f.txt -g g.txt -m full -st 2 -o y.bin
    mpirun -np 4 ./conv1d_mpi -f f.bin -g g.txt -m same -st 1 -o y.bin

    # Parallel generation of f (deterministic per global index):
    mpirun -np 8 ./conv1d_mpi -L 10000000 -kL 31 --parallel-gen -se 123 -o y.bin

  ARGUMENTS
      -f, --file PATH        Input signal f (text .txt or binary .bin)
      -g, --kernel PATH      Kernel g (text .txt or binary .bin)
      -L, --len N            Generate f of length N (if -f not given)
      -kL, --klen K          Generate g of length K (if -g not given)
      -se, --seed S          RNG seed (legacy -s also accepted)
      -m,  --mode M          same | full   (default: same)
      -p,  --padding P       zero | none | const   (default: zero)
      -c,  --cval V          Constant pad value when -p const
      -st, --stride N        Output stride (>=1)
      --parallel-gen         Each rank generates its slice of f locally
      --text                 Force text output (default is binary)
      -o,  --out PATH        Output file (required)

  INPUT FORMATS
    Text (.txt):
      line 1: integer L
      line 2: L whitespace-separated floats
    Binary (.bin):
      4 bytes: int32 L (little-endian typical)
      4*L bytes: float32 values

  OUTPUT FORMATS
    Text (--text):
      line 1: integer outLen
      line 2: outLen floats printed as %.3f
    Binary (default):
      4 bytes: int32 outLen
      4*outLen bytes: float32 values (rounded to 3 dp before write)

  DETERMINISM & COMPARABILITY
    - With a fixed seed, both root-generation and --parallel-gen paths are
      deterministic. --parallel-gen uses an indexable PRNG; root path uses
      stdlib rand with a single seed on root.
    - Outputs are rounded to 3 decimal places before writing (text and binary)
      so 3-decimal comparisons are identical across ranks/modes and match the
      sequential reference that prints to 3 dp.

  DEBUGGING
    - Build with -DDEBUG_MPI=1 for ordered per-rank logs showing:
      core ownership (local_start/count), output index range and count,
      halo width H, generation mode, and per-rank kernel time.

  NOTES / LIMITATIONS
    - Padding is applied in the inner loop; halos are not “fabricated” for
      const padding (global out-of-range contributes cval * g[m] directly).
    - Halo width is conservatively H = K-1 for both SAME and FULL modes.
    - This implementation favors clarity and lecture alignment over maximal
      overlap/compute pipelining. Further optimizations (e.g., double-buffered
      halos, vectorization, or hybrid MPI+OpenMP) can be layered on.

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

/* ------------------- small utils ------------------- */
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static int ends_with(const char *s, const char *suf)
{
    size_t n = strlen(s), m = strlen(suf);
    return (n >= m) && (memcmp(s + (n - m), suf, m) == 0);
}

/* ------------------- enums ------------------- */
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

/* ------------------- forward decls ------------------- */
void usage(const char *prog);
int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *stride, int *parallel_gen, int *text_mode);

float *read_array_1d_txt(const char *path, int *len_out);
void write_array_1d_txt(const char *path, const float *arr, int len);
float *gen_array_1d_seqrand(int n);
float gen_value_at_index(unsigned long seed, long long gi);

void ensure_dir(const char *path);
double elapsed_seconds(struct timespec a, struct timespec b);
void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs, double gflops);

/* ------------------- CLI & utils ------------------- */
void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-f f.txt|.bin | -L N | --len N] "
            "[-g g.txt|.bin | -kL K | --klen K] "
            "-o out.txt|out.bin|--out path "
            "[-se seed|--seed seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value] "
            "[-st stride|--stride stride] "
            "[--parallel-gen] [--text]\n",
            prog);
}

int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *stride, int *parallel_gen, int *text_mode)
{
    *f_path = *g_path = *o_path = NULL;
    *N_req = *K_req = -1;
    *seed = (unsigned long)time(NULL);
    *have_seed = 0;
    *cmode = MODE_SAME;
    *pmode = PAD_ZERO;
    *cval = 0.0f;
    *have_cval = 0;
    *stride = 1;
    *parallel_gen = 0;

    /* pre-scan to accept -L/-kL, -se, -st, --parallel-gen */
    int fargc = 1;
    char **fargv = (char **)malloc((size_t)argc * sizeof(char *));
    if (!fargv)
    {
        perror("malloc");
        return 0;
    }
    fargv[0] = argv[0];

    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        if (!strcmp(a, "-L"))
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-L requires arg\n");
                free(fargv);
                return 0;
            }
            *N_req = strtol(argv[++i], NULL, 10);
            continue;
        }
        if (!strncmp(a, "-L=", 3))
        {
            *N_req = strtol(a + 3, NULL, 10);
            continue;
        }
        if (!strcmp(a, "-kL"))
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-kL requires arg\n");
                free(fargv);
                return 0;
            }
            *K_req = strtol(argv[++i], NULL, 10);
            continue;
        }
        if (!strncmp(a, "-kL=", 4))
        {
            *K_req = strtol(a + 4, NULL, 10);
            continue;
        }
        if (!strcmp(a, "-se"))
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-se requires arg\n");
                free(fargv);
                return 0;
            }
            *seed = strtoul(argv[++i], NULL, 10);
            *have_seed = 1;
            continue;
        }
        if (!strncmp(a, "-se=", 4))
        {
            *seed = strtoul(a + 4, NULL, 10);
            *have_seed = 1;
            continue;
        }
        if (!strcmp(a, "-st"))
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-st requires arg\n");
                free(fargv);
                return 0;
            }
            *stride = (int)strtol(argv[++i], NULL, 10);
            continue;
        }
        if (!strncmp(a, "-st=", 4))
        {
            *stride = (int)strtol(a + 4, NULL, 10);
            continue;
        }
        if (!strcmp(a, "--parallel-gen"))
        {
            *parallel_gen = 1;
            continue;
        }
        fargv[fargc++] = argv[i];
        if (!strcmp(a, "--text"))
        {
            *text_mode = 1;
            continue;
        }
    }

    static struct option long_opts[] = {
        {"file", required_argument, 0, 'f'},
        {"kernel", required_argument, 0, 'g'},
        {"out", required_argument, 0, 'o'},
        {"seed", required_argument, 0, 's'}, /* legacy -s */
        {"len", required_argument, 0, 1},
        {"klen", required_argument, 0, 2},
        {"mode", required_argument, 0, 3},
        {"padding", required_argument, 0, 4},
        {"cval", required_argument, 0, 5},
        {"stride", required_argument, 0, 't'}, /* legacy -t */
        {"parallel-gen", no_argument, 0, 6},
        {"text-mode", no_argument, 0, 7},
        {0, 0, 0, 0}};

    int opt, idx = 0;
    opterr = 0;
    while ((opt = getopt_long(fargc, fargv, "f:g:o:s:m:p:c:t:", long_opts, &idx)) != -1)
    {
        switch (opt)
        {
        case 'f':
            *f_path = optarg;
            break;
        case 'g':
            *g_path = optarg;
            break;
        case 'o':
            *o_path = optarg;
            break;
        case 's':
            *seed = strtoul(optarg, NULL, 10);
            *have_seed = 1;
            break; /* legacy */
        case 'm':
        case 3:
            if (!strcmp(optarg, "same"))
                *cmode = MODE_SAME;
            else if (!strcmp(optarg, "full"))
                *cmode = MODE_FULL;
            else
            {
                usage(fargv[0]);
                free(fargv);
                return 0;
            }
            break;
        case 'p':
        case 4:
            if (!strcmp(optarg, "zero"))
                *pmode = PAD_ZERO;
            else if (!strcmp(optarg, "none"))
                *pmode = PAD_NONE;
            else if (!strcmp(optarg, "const"))
                *pmode = PAD_CONST;
            else
            {
                usage(fargv[0]);
                free(fargv);
                return 0;
            }
            break;
        case 'c':
        case 5:
            *cval = strtof(optarg, NULL);
            *have_cval = 1;
            break;
        case 't':
            *stride = (int)strtol(optarg, NULL, 10);
            break; /* legacy -t accept */
        case 1:
            *N_req = strtol(optarg, NULL, 10);
            break;
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break;
        case 6:
            *parallel_gen = 1;
            break;
        case 7:
            *text_mode = 1;
            break;
        default:
            break;
        }
    }
    free(fargv);

    if (!*o_path)
    {
        usage(argv[0]);
        return 0;
    }
    if (!*f_path && *N_req <= 0)
    {
        fprintf(stderr, "Missing -L/--len for f length\n");
        return 0;
    }
    if (!*g_path && *K_req <= 0)
    {
        fprintf(stderr, "Missing -kL/--klen for g length\n");
        return 0;
    }
    if (*pmode == PAD_CONST && !*have_cval)
        fprintf(stderr, "warning: -p const without -c/--cval; using cval=0.0\n");
    if (*stride <= 0)
    {
        fprintf(stderr, "stride must be >= 1\n");
        return 0;
    }
    return 1;
}

/* assignment text I/O */
float *read_array_1d_txt(const char *path, int *len_out)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        perror(path);
        exit(EXIT_FAILURE);
    }
    int L = 0;
    if (fscanf(fp, "%d", &L) != 1 || L <= 0)
    {
        fprintf(stderr, "bad header in %s\n", path);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    float *a = (float *)malloc((size_t)L * sizeof(float));
    if (!a)
    {
        fprintf(stderr, "OOM reading %s\n", path);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < L; i++)
        if (fscanf(fp, "%f", &a[i]) != 1)
        {
            fprintf(stderr, "bad body in %s at %d\n", path, i);
            free(a);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
    fclose(fp);
    *len_out = L;
    return a;
}
void write_array_1d_txt(const char *path, const float *arr, int len)
{
    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        perror(path);
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "%d\n", len);
    for (int i = 0; i < len; i++)
        fprintf(fp, (i + 1 == len) ? "%.3f\n" : "%.3f ", arr[i]);
    fclose(fp);
}

/* RNG helpers */
float *gen_array_1d_seqrand(int n)
{
    if (n <= 0)
    {
        fprintf(stderr, "invalid n=%d\n", n);
        exit(EXIT_FAILURE);
    }
    float *a = (float *)malloc((size_t)n * sizeof(float));
    if (!a)
    {
        fprintf(stderr, "OOM gen n=%d\n", n);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++)
    {
        float u = (float)rand() / (float)RAND_MAX;
        a[i] = -1.0f + 2.0f * u;
    }
    return a;
}
/* indexable PRNG to support --parallel-gen */
static inline unsigned long long splitmix64(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x = x ^ (x >> 31);
    return x;
}
float gen_value_at_index(unsigned long seed, long long gi)
{
    unsigned long long x = ((unsigned long long)seed << 32) ^ (unsigned long long)gi;
    unsigned long long r = splitmix64(x);
    float u01 = (float)((r >> 40) & 0xFFFFFF) / (float)0x1000000; /* ~24 bits */
    return -1.0f + 2.0f * u01;
}

/* misc */
void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        (void)mkdir(path, 0775);
}
double elapsed_seconds(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}
void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval,
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
    fprintf(csv, "RunID,N,K,outLen,mode,padding,cval,time,gflops\n");
    fprintf(csv, "%s,%d,%d,%d,%s,%s,%.9g,%.9f,%.6f\n",
            runid, N, K, outLen, (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0), elapsed_secs, gflops);
    fclose(csv);
}

/* ------------------- main (with 1-D Cartesian topology) ------------------- */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    /* Base world ranks (before cart, useful if you ever want to compare) */
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Create a 1-D Cartesian communicator, non-periodic, no rank reorder */
    MPI_Comm comm;
    {
        int dims[1] = {world_size};
        int periods[1] = {0}; /* non-periodic: open ends */
        int reorder = 0;      /* keep rank numbers stable */
        MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &comm);
        if (comm == MPI_COMM_NULL)
        {
            /* Fallback (shouldn’t happen for 1D full size) */
            comm = MPI_COMM_WORLD;
        }
    }

    /* From here on, use the cart communicator for everything */
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Find left/right neighbors in the cart topology (MPI_PROC_NULL at edges) */
    int left = MPI_PROC_NULL, right = MPI_PROC_NULL;
    MPI_Cart_shift(comm, /*direction=*/0, /*disp=*/1, &left, &right);

    const char *f_path = NULL, *g_path = NULL, *o_path = NULL;
    long N_req = -1, K_req = -1;
    unsigned long seed = (unsigned long)time(NULL);
    int have_seed = 0;
    conv_mode cmode = MODE_SAME;
    pad_mode pmode = PAD_ZERO;
    float cval = 0.0f;
    int have_cval = 0;
    int stride = 1;
    int parallel_gen = 0;
    int text_mode = 0;

    if (!parse_args(argc, argv, &f_path, &g_path, &o_path,
                    &N_req, &K_req, &seed, &have_seed,
                    &cmode, &pmode, &cval, &have_cval,
                    &stride, &parallel_gen, &text_mode))
    {
        MPI_Abort(comm, 1);
    }

    /* Sizes and kernels */
    int N = 0, K = 0;
    float *f_root = NULL, *g_root = NULL;

    /* Decide input mode per file: text (.txt) => root read; binary (.bin) => parallel read */
    const int f_is_bin = (f_path && ends_with(f_path, ".bin"));
    const int g_is_bin = (g_path && ends_with(g_path, ".bin"));

    if (rank == 0)
    {
        if (f_path && !f_is_bin)
        {
            f_root = read_array_1d_txt(f_path, &N);
        }
        else if (!f_path)
        {
            N = (int)N_req;
            if (!parallel_gen)
            {
                if (have_seed)
                    srand((unsigned)seed);
                else
                    srand((unsigned)seed);
                f_root = gen_array_1d_seqrand(N);
            }
        }
        if (g_path && !g_is_bin)
        {
            g_root = read_array_1d_txt(g_path, &K);
        }
        else if (!g_path)
        {
            K = (int)K_req;
            /* keep the g RNG stream separate from f if you ever reseed */
            g_root = gen_array_1d_seqrand(K);
        }
    }

    /* Broadcast scalar config + seed (so parallel-gen path is deterministic) */
    MPI_Bcast(&N, 1, MPI_INT, 0, comm);
    MPI_Bcast(&K, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cmode, 1, MPI_INT, 0, comm);
    MPI_Bcast(&pmode, 1, MPI_INT, 0, comm);
    MPI_Bcast(&cval, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(&stride, 1, MPI_INT, 0, comm);
    MPI_Bcast(&parallel_gen, 1, MPI_INT, 0, comm);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, comm);

    /* Partition f across ranks: sendcounts/displs (needed for both scatter and parallel reads) */
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc((size_t)size * sizeof(int));
        displs = (int *)malloc((size_t)size * sizeof(int));
        int base = N / size, rem = N % size, off = 0;
        for (int r = 0; r < size; r++)
        {
            int cnt = base + (r < rem ? 1 : 0);
            sendcounts[r] = cnt;
            displs[r] = off;
            off += cnt;
        }
    }
    int local_count = 0, local_start = 0;
    if (rank == 0)
    {
        local_count = sendcounts[0];
        local_start = displs[0];
    }
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, comm);
    MPI_Scatter(displs, 1, MPI_INT, &local_start, 1, MPI_INT, 0, comm);

    float *f_local = (float *)malloc((size_t)local_count * sizeof(float));
    if (!f_local)
    {
        fprintf(stderr, "[%d] OOM f_local\n", rank);
        MPI_Abort(comm, 1);
    }

    /* ---- Input for f ---- */
    if (f_path && f_is_bin)
    {
        /* binary parallel read: [int32 N][float32 N] */
        MPI_File fh;
        MPI_Status st;
        int rc = MPI_File_open(comm, (char *)f_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        if (rc != MPI_SUCCESS)
        {
            if (rank == 0)
                fprintf(stderr, "MPI_File_open(f.bin) failed\n");
            MPI_Abort(comm, 1);
        }

        int hdrN = 0;
        if (rank == 0)
            MPI_File_read_at(fh, 0, &hdrN, 1, MPI_INT, &st);
        MPI_Bcast(&hdrN, 1, MPI_INT, 0, comm);
        if (hdrN != N)
        {
            if (rank == 0)
                fprintf(stderr, "f.bin header (%d) != N (%d)\n", hdrN, N);
            MPI_Abort(comm, 1);
        }

        MPI_Offset base = (MPI_Offset)sizeof(int);
        MPI_Offset off_bytes = base + (MPI_Offset)local_start * (MPI_Offset)sizeof(float);
        MPI_Datatype seg;
        MPI_Type_contiguous(local_count, MPI_FLOAT, &seg);
        MPI_Type_commit(&seg);
        MPI_File_read_at_all(fh, off_bytes, f_local, 1, seg, &st);
        MPI_Type_free(&seg);
        MPI_File_close(&fh);
    }
    else if (f_path && !f_is_bin)
    {
        /* text path: root parsed; scatterv floats */
        MPI_Scatterv(f_root, sendcounts, displs, MPI_FLOAT,
                     f_local, local_count, MPI_FLOAT, 0, comm);
    }
    else
    {
        /* generated */
        if (!parallel_gen)
        {
            MPI_Scatterv(f_root, sendcounts, displs, MPI_FLOAT,
                         f_local, local_count, MPI_FLOAT, 0, comm);
        }
        else
        {
            for (int i = 0; i < local_count; i++)
            {
                long long gi = (long long)local_start + i;
                f_local[i] = gen_value_at_index((unsigned long)seed, gi);
            }
        }
    }

    if (rank == 0)
    {
        free(f_root);
        free(sendcounts);
        free(displs);
    }

    /* ---- Handle g: small, easiest path is broadcast; but add optional parallel read if .bin ---- */
    float *g = (float *)malloc((size_t)K * sizeof(float));
    if (!g)
    {
        fprintf(stderr, "[%d] OOM g\n", rank);
        MPI_Abort(comm, 1);
    }

    if (g_path && g_is_bin)
    {
        MPI_File fh;
        MPI_Status st;
        int rc = MPI_File_open(comm, (char *)g_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        if (rc != MPI_SUCCESS)
        {
            if (rank == 0)
                fprintf(stderr, "MPI_File_open(g.bin) failed\n");
            MPI_Abort(comm, 1);
        }
        int hdrK = 0;
        if (rank == 0)
            MPI_File_read_at(fh, 0, &hdrK, 1, MPI_INT, &st);
        MPI_Bcast(&hdrK, 1, MPI_INT, 0, comm);
        if (hdrK != K)
        {
            if (rank == 0)
                fprintf(stderr, "g.bin header (%d) != K (%d)\n", hdrK, K);
            MPI_Abort(comm, 1);
        }

        MPI_Offset base = (MPI_Offset)sizeof(int);
        MPI_File_read_at_all(fh, base, g, K, MPI_FLOAT, &st);
        MPI_File_close(&fh);
    }
    else
    {
        if (rank == 0)
        {
            if (g_root)
                memcpy(g, g_root, (size_t)K * sizeof(float));
            else
                for (int i = 0; i < K; i++)
                    g[i] = 0.0f;
        }
        MPI_Bcast(g, K, MPI_FLOAT, 0, comm);
    }
    if (rank == 0 && g_root)
        free(g_root);

    /* ---- Build ghost buffer + non-blocking halo exchange ---- */
    const int H = (K > 0 ? K - 1 : 0);
    const int buf_len = local_count + 2 * H;
    float *buf = (float *)malloc((size_t)buf_len * sizeof(float));
    if (!buf)
    {
        fprintf(stderr, "[%d] OOM buf\n", rank);
        MPI_Abort(comm, 1);
    }
    for (int i = 0; i < buf_len; i++)
        buf[i] = 0.0f;
    memcpy(buf + H, f_local, (size_t)local_count * sizeof(float));

    int edge = (H <= local_count ? H : local_count);
    MPI_Request reqs[4];
    /* Left halo: recv our left ghost, send our left boundary */
    MPI_Irecv(buf, H, MPI_FLOAT, left, 102, comm, &reqs[0]);
    MPI_Isend(f_local, edge, MPI_FLOAT, left, 101, comm, &reqs[1]);
    /* Right halo: recv our right ghost, send our right boundary */
    MPI_Irecv(buf + (H + local_count), H, MPI_FLOAT, right, 101, comm, &reqs[2]);
    MPI_Isend((local_count >= H ? f_local + (local_count - H) : f_local),
              edge, MPI_FLOAT, right, 102, comm, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    /* ---- Output size + decomposition (in y index-space j_out = 0..outLen-1) ---- */
    const int fullLen = N + K - 1;
    const int outLen = (cmode == MODE_FULL) ? ceil_div(fullLen, stride)
                                            : ceil_div(N, stride);
    int base_out = outLen / size, rem_out = outLen % size;
    int my_out_start = rank * base_out + (rank < rem_out ? rank : rem_out);
    int my_out_count = base_out + (rank < rem_out ? 1 : 0);
    int my_out_end = my_out_start + my_out_count;

#ifdef DEBUG_MPI
    for (int r = 0; r < size; r++)
    {
        MPI_Barrier(comm);
        if (rank == r)
        {
            fprintf(stderr,
                    "[rank %d/%d] %s: N=%d K=%d stride=%d | n_range=[%d..%d) (out_cnt=%d) | "
                    "f_core=[%d..%d] (core_cnt=%d) | H=%d | gen=%s\n",
                    rank, size, (cmode == MODE_FULL ? "FULL" : "SAME"),
                    N, K, stride,
                    my_out_start * stride,
                    (my_out_end > 0 ? (my_out_end - 1) * stride : -1) + 1,
                    my_out_count,
                    (int)((rank * (N / size)) + (rank < (N % size) ? rank : (N % size))),
                    (int)((rank * (N / size)) + (rank < (N % size) ? rank : (N % size))) + local_count - 1,
                    local_count, H, (parallel_gen ? "parallel" : "root"));
            fflush(stderr);
        }
    }
    MPI_Barrier(comm);
#endif

    float *y_local = (float *)malloc((size_t)my_out_count * sizeof(float));
    if (!y_local)
    {
        fprintf(stderr, "[%d] OOM y_local\n", rank);
        MPI_Abort(comm, 1);
    }

    /* ---- Kernel-only timing ---- */
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    if (cmode == MODE_SAME)
    {
        const int c = K / 2;
        for (int j = 0; j < my_out_count; j++)
        {
            const int n = (my_out_start + j) * stride; /* 0..N-1 */
            double acc = 0.0;
            for (int m = 0; m < K; m++)
            {
                int idx = n - (m - c); /* global f index */
                if (idx >= 0 && idx < N)
                {
                    int in_local = idx - ((N / size) * rank + (rank < (N % size) ? rank : (N % size))) + H;
                    if (in_local >= 0 && in_local < buf_len)
                    {
                        acc += (double)buf[in_local] * (double)g[m];
                    }
                }
                else if (pmode == PAD_CONST)
                {
                    acc += (double)cval * (double)g[m];
                }
            }
            y_local[j] = (float)acc;
        }
    }
    else /* MODE_FULL */
    {
        for (int j = 0; j < my_out_count; j++)
        {
            const int n = (my_out_start + j) * stride; /* 0..N+K-2 */
            double acc = 0.0;
            int i_lo = n - (K - 1);
            if (i_lo < 0)
                i_lo = 0;
            int i_hi = (n < N - 1) ? n : (N - 1);
            for (int i = i_lo; i <= i_hi; i++)
            {
                int in_local = i - ((N / size) * rank + (rank < (N % size) ? rank : (N % size))) + H;
                if (in_local >= 0 && in_local < buf_len)
                {
                    int m = n - i; /* 0..K-1 */
                    acc += (double)buf[in_local] * (double)g[m];
                }
            }
            y_local[j] = (float)acc;
        }
    }

    double local_secs = MPI_Wtime() - t0;

#ifdef DEBUG_MPI
    fprintf(stderr, "[rank %d/%d] kernel_time=%.9f s (out_cnt=%d)\n",
            rank, size, local_secs, my_out_count);
#endif

    /* reductions for perf */
    double local_flops = 2.0 * (double)K * (double)my_out_count;
    double sum_flops = 0.0, max_secs = 0.0, min_secs = 0.0, sum_secs = 0.0;
    MPI_Reduce(&local_flops, &sum_flops, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_secs, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_secs, &min_secs, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_secs, &sum_secs, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    /* ---- Output ----
       If --text, gather and write assignment-style text (rounded 3dp).
       Else default to binary MPI-IO: [int32 outLen] + float32*outLen (rounded 3dp). */
    if (text_mode)
    {
        int *recvcounts = NULL, *rdispls = NULL;
        if (rank == 0)
        {
            recvcounts = (int *)malloc((size_t)size * sizeof(int));
            rdispls = (int *)malloc((size_t)size * sizeof(int));
        }
        MPI_Gather(&my_out_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);
        if (rank == 0)
        {
            int off = 0;
            for (int r = 0; r < size; r++)
            {
                rdispls[r] = off;
                off += recvcounts[r];
            }
        }

        for (int j = 0; j < my_out_count; ++j)
        {
            double scaled = y_local[j] * 1000.0;
            double rounded = nearbyint(scaled);
            y_local[j] = (float)(rounded * 0.001);
        }

        float *y_root = NULL;
        if (rank == 0)
        {
            y_root = (float *)malloc((size_t)outLen * sizeof(float));
            if (!y_root)
            {
                fprintf(stderr, "[root] OOM y_root\n");
                MPI_Abort(comm, 1);
            }
        }
        MPI_Gatherv(y_local, my_out_count, MPI_FLOAT,
                    y_root, recvcounts, rdispls, MPI_FLOAT,
                    0, comm);

        if (rank == 0)
        {
            write_array_1d_txt(o_path, y_root, outLen);
            double gflops = (max_secs > 0.0) ? (sum_flops / (max_secs * 1e9)) : 0.0;
            double avg_secs = sum_secs / (double)size;
            fprintf(stderr,
                    "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g stride=%d | ranks=%d | "
                    "conv_time=%.9f s | %.3f GFLOP/s | perRank(min/avg/max)=%.9f/%.9f/%.9f s\n",
                    N, K, outLen,
                    (cmode == MODE_FULL ? "full" : "same"),
                    (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                    (pmode == PAD_CONST ? cval : 0.0),
                    stride, size, max_secs, gflops, min_secs, avg_secs, max_secs);
            log_metrics(N, K, outLen, cmode, pmode, cval, max_secs, gflops);
            free(y_root);
            free(recvcounts);
            free(rdispls);
        }
    }
    else
    {
        /* Binary MPI-IO: [int32 outLen][float32*outLen], rounded 3 dp */
        MPI_File fh;
        MPI_Status st;
        int mpierr = MPI_File_open(comm, (char *)o_path,
                                   MPI_MODE_CREATE | MPI_MODE_WRONLY,
                                   MPI_INFO_NULL, &fh);
        if (mpierr != MPI_SUCCESS)
        {
            if (rank == 0)
                fprintf(stderr, "MPI_File_open failed\n");
            MPI_Abort(comm, 1);
        }

        for (int j = 0; j < my_out_count; ++j)
        {
            double scaled = y_local[j] * 1000.0;
            double rounded = nearbyint(scaled);
            y_local[j] = (float)(rounded * 0.001);
        }

        if (rank == 0)
        {
            MPI_File_write_at(fh, 0, (void *)&outLen, 1, MPI_INT, &st);
        }
        MPI_Barrier(comm);

        int *recvcounts = NULL, *rdispls = NULL;
        if (rank == 0)
        {
            recvcounts = (int *)malloc((size_t)size * sizeof(int));
            rdispls = (int *)malloc((size_t)size * sizeof(int));
        }
        MPI_Gather(&my_out_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);
        if (rank == 0)
        {
            int off = 0;
            for (int r = 0; r < size; r++)
            {
                rdispls[r] = off;
                off += recvcounts[r];
            }
        }

        MPI_Offset header_bytes = (MPI_Offset)sizeof(int);
        MPI_Offset my_byte_offset = header_bytes;
        if (rank != 0)
        {
            int disp = 0;
            MPI_Recv(&disp, 1, MPI_INT, 0, 900 + rank, comm, MPI_STATUS_IGNORE);
            my_byte_offset += (MPI_Offset)disp * (MPI_Offset)sizeof(float);
        }
        else
        {
            for (int r = 1; r < size; r++)
                MPI_Send(&rdispls[r], 1, MPI_INT, r, 900 + r, comm);
        }

        MPI_Datatype segtype;
        MPI_Type_contiguous(my_out_count, MPI_FLOAT, &segtype);
        MPI_Type_commit(&segtype);
        MPI_File_write_at_all(fh, my_byte_offset, y_local, 1, segtype, &st);
        MPI_Type_free(&segtype);
        MPI_File_close(&fh);

        if (rank == 0)
        {
            double gflops = (max_secs > 0.0) ? (sum_flops / (max_secs * 1e9)) : 0.0;
            double avg_secs = sum_secs / (double)size;
            fprintf(stderr,
                    "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g stride=%d | ranks=%d | "
                    "conv_time=%.9f s | %.3f GFLOP/s | perRank(min/avg/max)=%.9f/%.9f/%.9f s\n",
                    N, K, outLen,
                    (cmode == MODE_FULL ? "full" : "same"),
                    (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                    (pmode == PAD_CONST ? cval : 0.0),
                    stride, size, max_secs, gflops, min_secs, avg_secs, max_secs);
            log_metrics(N, K, outLen, cmode, pmode, cval, max_secs, gflops);
            free(recvcounts);
            free(rdispls);
        }
    }

    /* ---- Cleanup ---- */
    free(f_local);
    free(buf);
    free(g);
    free(y_local);

    if (comm != MPI_COMM_WORLD)
        MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}