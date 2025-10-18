/**
 * @file conv1d_mpi.c
 * @brief MPI-enabled 1-D Convolution with multiple modes, padding strategies, STRIDE,
 *        reproducible RNG, assignment-compliant I/O, and per-run CSV metrics (incl. GFLOP/s).
 *
 * This program implements TRUE 1-D convolution (y = f * g) with support for:
 *   - Output length modes (SAME/FULL)
 *   - Padding policies (zero/none/const)
 *   - Stride-based downsampling (-st/--stride)
 *   - Reproducible RNG for generated inputs (-se/--seed; legacy -s supported)
 *   - Assignment-style I/O (read/write text arrays with header length)
 *   - Accurate timing of only the convolution kernel
 *   - Per-run CSV metrics with GFLOP/s and # of MPI ranks
 *
 * ### Output length modes (-m/--mode):
 *   - same (default): output length = N (input length)       [with stride: ceil(N / t)]
 *   - full          : output length = N + K - 1              [with stride: ceil((N+K-1) / t)]
 *
 * ### Padding policies (-p/--padding), for SAME mode:
 *   - zero  (default): out-of-range values treated as 0.0f
 *   - none           : out-of-range contributions ignored
 *   - const          : out-of-range values replaced with a user constant (-c/--cval)
 *
 * ### Stride (-st/--stride):
 *   - Controls how densely outputs are sampled.
 *   - stride = 1 (default): compute every output position.
 *   - stride > 1: compute every stride-th position (downsamples the convolution).
 *   - Affects output lengths:
 *       SAME → ceil(N / stride)
 *       FULL → ceil((N + K - 1) / stride)
 *
 * ### Input options:
 *   - Read arrays from text files:
 *       -f <f.txt> | --file <f.txt>        : input signal f
 *       -g <g.txt> | --kernel <g.txt>      : kernel g
 *   - Generate random arrays with uniform floats in [-1, 1]:
 *       -L <N>   | --len <N>   : generate input f of length N
 *       -kL <K>  | --klen <K>  : generate kernel g of length K
 *   - RNG reproducibility:
 *       -se <seed> | --seed <seed> : deterministic seed (defaults to time if omitted)
 *       (Legacy short -s is still accepted for back-compat.)
 *
 * ### Output:
 *   - Assignment-compliant text file:
 *       Line 1: integer length L
 *       Line 2: L floats with 3 decimal places, space-separated
 *   - Written via:
 *       -o <out.txt> | --out <out.txt>
 *
 * ### CSV metrics logging:
 *   - Each run writes a single row of metrics (one file per run) under "metrics/".
 *   - Filenames are unique per SLURM or local run.
 *   - Columns: RunID,N,K,outLen,mode,padding,cval,stride,time,gflops,ranks
 *
 * ### Numerical policy:
 *   - Arrays stored as float32
 *   - Accumulation uses double to reduce rounding error, then cast back to float
 *
 * ### Timing policy:
 *   - Reports ONLY the convolution kernel execution time (MPI_Barrier + MPI_Wtime)
 *   - File I/O and RNG excluded (per assignment requirements)
 *   - GFLOP/s model: flops ≈ 2 * outLen * K; gflops = flops / (time * 1e9).
 *
 * ### Example usage:
 *   mpicc -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c
 *
 *   # SAME length, ZERO padding, random input, seed 42, stride 2, 4 ranks
 *   mpirun -np 4 ./conv1d_mpi -L 1024 -kL 5 -m same -p zero -st 2 -se 42 -o y.txt
 *
 *   # FULL mode convolution with stride 3 from files
 *   mpirun -np 8 ./conv1d_mpi -f f.txt -g g.txt -m full -st 3 -o y.txt
 *
 * @note Output must be identical to the sequential conv1d for any number of ranks.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <getopt.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mpi.h>

/* Toggle verbose per-rank debug logging (stderr) */
#ifndef DEBUG_MPI
#define DEBUG_MPI 0
#endif

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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

/* Forward declarations */
void usage(const char *prog);
int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *stride);

float *read_array_1d(const char *path, int *len_out);
void write_array_1d(const char *path, const float *arr, int len);
float *gen_array_1d(int n);

void ensure_dir(const char *path);
void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval, int stride,
                 double elapsed_secs, double gflops, int ranks);

/* Convolution kernels (sequential math, called on disjoint output ranges) */
static void conv1d_full_stride_range(const float *f, int N,
                                     const float *g, int K,
                                     int stride,
                                     int n_out_start, int n_out_end, /* [start,end) in out-index space */
                                     float *out /* length at least (n_out_end - n_out_start) */)
{
    const int fullLen = N + K - 1;
    int pos = 0;
    for (int n_out = n_out_start; n_out < n_out_end; n_out++, pos++)
    {
        const int n = n_out * stride; /* physical (un-strided) output index */
        double acc = 0.0;

        int i_lo = n - (K - 1);
        if (i_lo < 0)
            i_lo = 0;
        int i_hi = n;
        if (i_hi > N - 1)
            i_hi = N - 1;

        (void)fullLen; /* fullLen referenced in design; not needed in loop body. */
        for (int i = i_lo; i <= i_hi; i++)
        {
            const int m = n - i; /* 0..K-1 */
            acc += (double)f[i] * (double)g[m];
        }
        out[pos] = (float)acc;
    }
}

static void conv1d_same_stride_range(const float *f, int N,
                                     const float *g, int K,
                                     int stride,
                                     pad_mode pmod, float cval,
                                     int n_out_start, int n_out_end, /* [start,end) */
                                     float *out /* length at least (n_out_end - n_out_start) */)
{
    const int c = K / 2;
    int pos = 0;
    for (int n_out = n_out_start; n_out < n_out_end; n_out++, pos++)
    {
        const int n = n_out * stride; /* physical (un-strided) output index in [0..N-1] if stride==1 */
        double acc = 0.0;
        for (int m = 0; m < K; m++)
        {
            const int idx = n - (m - c);
            if (idx >= 0 && idx < N)
            {
                acc += (double)f[idx] * (double)g[m];
            }
            else if (pmod == PAD_CONST)
            {
                acc += (double)cval * (double)g[m];
            }
            /* PAD_ZERO: add 0 (no-op). PAD_NONE: skip. */
        }
        out[pos] = (float)acc;
    }
}

/* ------------------------------ Usage ------------------------------ */
void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-f f.txt | -L N | --len N] "
            "[-g g.txt | -kL K | --klen K] "
            "-o out.txt|--out out.txt "
            "[-se seed|--seed seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value] "
            "[-st stride|--stride stride]\n",
            prog);
}

/* ------------------------------ CLI parsing ------------------------------ */
int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *stride)
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

    /* Pre-filter to accept -L/-kL and multi-letter short opts -se/-st */
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
                fprintf(stderr, "-L requires an argument\n");
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
                fprintf(stderr, "-kL requires an argument\n");
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
                fprintf(stderr, "-se requires an argument\n");
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
                fprintf(stderr, "-st requires an argument\n");
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

        fargv[fargc++] = argv[i];
    }

    static struct option long_opts[] = {
        {"file", required_argument, 0, 'f'},
        {"kernel", required_argument, 0, 'g'},
        {"out", required_argument, 0, 'o'},
        {"seed", required_argument, 0, 's'}, /* long --seed (back-compat) */
        {"len", required_argument, 0, 1},
        {"klen", required_argument, 0, 2},
        {"mode", required_argument, 0, 3},
        {"padding", required_argument, 0, 4},
        {"cval", required_argument, 0, 5},
        {"stride", required_argument, 0, 't'}, /* long --stride (back-compat) */
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
            break; /* legacy short -s */
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
        case 't': /* legacy short stride (optional back-compat) */
            *stride = (int)strtol(optarg, NULL, 10);
            break;
        case 1:
            *N_req = strtol(optarg, NULL, 10);
            break; /* --len */
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break; /* --klen */
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

/* ------------------------------ I/O helpers ------------------------------ */
float *read_array_1d(const char *path, int *len_out)
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
        fprintf(stderr, "bad header in %s (expected positive integer length)\n", path);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    float *arr = (float *)malloc((size_t)L * sizeof(float));
    if (!arr)
    {
        fprintf(stderr, "OOM reading %s (L=%d)\n", path, L);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < L; i++)
    {
        if (fscanf(fp, "%f", &arr[i]) != 1)
        {
            fprintf(stderr, "bad body in %s at index %d\n", path, i);
            free(arr);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
    }
    fclose(fp);
    *len_out = L;
    return arr;
}

void write_array_1d(const char *path, const float *arr, int len)
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

float *gen_array_1d(int n)
{
    if (n <= 0)
    {
        fprintf(stderr, "invalid length n=%d\n", n);
        exit(EXIT_FAILURE);
    }
    float *a = (float *)malloc((size_t)n * sizeof(float));
    if (!a)
    {
        fprintf(stderr, "OOM generating array (n=%d)\n", n);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++)
    {
        float u01 = (float)rand() / (float)RAND_MAX; /* [0,1] */
        a[i] = -1.0f + 2.0f * u01;                   /* [-1,1] */
    }
    return a;
}

/* ------------------------------ Utils ------------------------------ */
static double now_mpi(void) { return MPI_Wtime(); }

void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
    {
        (void)mkdir(path, 0775);
    }
}

void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval, int stride,
                 double elapsed_secs, double gflops, int ranks)
{
    ensure_dir("metrics/");

    const char *slurm = getenv("SLURM_JOB_ID");
    char runid[128];
    if (slurm && slurm[0])
    {
        snprintf(runid, sizeof(runid), "SLURM_%s", slurm);
    }
    else
    {
        time_t t = time(NULL);
        struct tm tm;
        localtime_r(&t, &tm);
        pid_t pid = getpid();
        strftime(runid, sizeof(runid), "LOCAL_%Y%m%d_%H%M%S", &tm);
        size_t len = strlen(runid);
        snprintf(runid + len, sizeof(runid) - len, "_%d", (int)pid);
    }

    char fname[256];
    snprintf(fname, sizeof(fname), "metrics/metrics_%s.csv", runid);
    FILE *csv = fopen(fname, "w");
    if (!csv)
    {
        perror(fname);
        return;
    }

    fprintf(csv, "RunID,N,K,outLen,mode,padding,cval,stride,time,gflops,ranks\n");
    fprintf(csv, "%s,%d,%d,%d,%s,%s,%.9g,%d,%.9f,%.6f,%d\n",
            runid, N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            stride, elapsed_secs, gflops, ranks);

    fclose(csv);
}

/* ------------------------------ Main (MPI) ------------------------------ */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *f_path = NULL, *g_path = NULL, *o_path = NULL;
    long N_req = -1, K_req = -1;
    unsigned long seed = (unsigned long)time(NULL);
    int have_seed = 0;
    conv_mode cmode = MODE_SAME;
    pad_mode pmode = PAD_ZERO;
    float cval = 0.0f;
    int have_cval = 0;
    int stride = 1;

    /* Parse CLI only on rank 0, then broadcast */
    int ok = 1;
    if (rank == 0)
    {
        ok = parse_args(argc, argv, &f_path, &g_path, &o_path,
                        &N_req, &K_req, &seed, &have_seed,
                        &cmode, &pmode, &cval, &have_cval, &stride);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok)
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Broadcast scalar config */
    int cmode_i = 0, pmode_i = 0;
    if (rank == 0)
    {
        cmode_i = (int)cmode;
        pmode_i = (int)pmode;
    }
    MPI_Bcast(&cmode_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pmode_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cmode = (conv_mode)cmode_i;
    pmode = (pad_mode)pmode_i;

    MPI_Bcast(&stride, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    /* Root does I/O or RNG; broadcast N, K and arrays */
    int N = 0, K = 0;
    float *f = NULL, *g = NULL;

    if (rank == 0)
    {
        if (have_seed)
            srand((unsigned)seed);
        else
            srand((unsigned)seed);

        if (f_path)
            f = read_array_1d(f_path, &N);
        else
        {
            N = (int)N_req;
            f = gen_array_1d(N);
        }

        if (g_path)
            g = read_array_1d(g_path, &K);
        else
        {
            K = (int)K_req;
            g = gen_array_1d(K);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N <= 0 || K <= 0)
    {
        if (rank == 0)
            fprintf(stderr, "Invalid N or K after broadcast.\n");
        if (f)
            free(f);
        if (g)
            free(g);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Allocate and broadcast arrays */
    if (rank != 0)
    {
        f = (float *)malloc((size_t)N * sizeof(float));
        g = (float *)malloc((size_t)K * sizeof(float));
        if (!f || !g)
        {
            fprintf(stderr, "[rank %d] OOM allocating f/g\n", rank);
            if (f)
                free(f);
            if (g)
                free(g);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }
    MPI_Bcast(f, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g, K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Compute global outLen (stride-aware) */
    const int outLen = (cmode == MODE_FULL) ? ceil_div(N + K - 1, stride)
                                            : ceil_div(N, stride);

    /* Partition output across ranks (nearly equal blocks) */
    int *counts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    if (!counts || !displs)
    {
        fprintf(stderr, "[rank %d] OOM counts/displs\n", rank);
        free(counts);
        free(displs);
        free(f);
        free(g);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int base = outLen / size;
    const int rem = outLen % size;
    for (int r = 0; r < size; r++)
        counts[r] = base + (r < rem ? 1 : 0);
    displs[0] = 0;
    for (int r = 1; r < size; r++)
        displs[r] = displs[r - 1] + counts[r - 1];

#if DEBUG_MPI
    fprintf(stderr, "[rank %d/%d] outLen=%d stride=%d | my_range=[%d..%d) count=%d\n",
            rank, size, outLen, stride,
            displs[rank], displs[rank] + counts[rank], counts[rank]);
#endif

    /* Local output buffer for this rank's block */
    float *out_local = (counts[rank] > 0)
                           ? (float *)malloc((size_t)counts[rank] * sizeof(float))
                           : NULL;
    if (counts[rank] > 0 && !out_local)
    {
        fprintf(stderr, "[rank %d] OOM out_local count=%d\n", rank, counts[rank]);
        free(counts);
        free(displs);
        free(f);
        free(g);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Barrier to exclude input broadcast from kernel timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = now_mpi();

    /* Compute local block */
    const int n_out_start = displs[rank];
    const int n_out_end = n_out_start + counts[rank];
    if (counts[rank] > 0)
    {
        if (cmode == MODE_FULL)
        {
            conv1d_full_stride_range(f, N, g, K, stride,
                                     n_out_start, n_out_end,
                                     out_local);
        }
        else
        {
            conv1d_same_stride_range(f, N, g, K, stride, pmode, cval,
                                     n_out_start, n_out_end,
                                     out_local);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = now_mpi();
    const double secs = t1 - t0;

    /* Gather results to root */
    float *out = NULL;
    if (rank == 0)
    {
        out = (float *)malloc((size_t)outLen * sizeof(float));
        if (!out)
        {
            fprintf(stderr, "[rank 0] OOM outLen=%d\n", outLen);
            free(out_local);
            free(counts);
            free(displs);
            free(f);
            free(g);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    MPI_Gatherv(out_local, counts[rank], MPI_FLOAT,
                out, counts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /* FLOP model and logging (root only) */
    if (rank == 0)
    {
        double flops = 2.0 * (double)outLen * (double)K;
        double gflops = (secs > 0.0) ? (flops / secs / 1e9) : 0.0;

        fprintf(stderr,
                "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g stride=%d | ranks=%d | conv_time=%.9f s | %.3f GFLOP/s\n",
                N, K, outLen,
                (cmode == MODE_FULL ? "full" : "same"),
                (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                (pmode == PAD_CONST ? cval : 0.0),
                stride, size, secs, gflops);

        /* Write output array */
        write_array_1d(o_path, out, outLen);

        /* Per-run CSV metrics */
        log_metrics(N, K, outLen, cmode, pmode, cval, stride, secs, gflops, size);
    }

    /* Cleanup */
    if (out)
        free(out);
    if (out_local)
        free(out_local);
    free(counts);
    free(displs);
    free(f);
    free(g);

    MPI_Finalize();
    return EXIT_SUCCESS;
}