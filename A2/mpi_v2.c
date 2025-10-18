/**
 * @file conv1d_mpi.c
 * @brief MPI-enabled 1-D Convolution with SAME/FULL modes, padding, STRIDE,
 *        reproducible RNG, assignment-compliant text I/O (gather mode),
 *        optional MPI-IO (collective/independent) binary output via FILE VIEWS,
 *        kernel-only timing, Cartesian topology diagnostics,
 *        domain decomposition with halos (no broadcast of full f),
 *        controlled decomposition policy, MPI-IO hints, and CSV metrics.
 *
 * Now includes SAFE ultra-fine overlap (enabled by default):
 *   - Compute core "interior" while halos are in flight.
 *   - As each halo arrives, compute that side's border immediately using
 *     segment-aware kernels (no premature use of an assembled buffer).
 *   - Disable with -DULTRA_OVERLAP=0 if you prefer the simpler fallback path.
 *
 * Defaults: fblock (halo) decomposition, gather→text output file.
 *
 * Usage:
 *   mpirun -np <P> ./conv1d_mpi [inputs] -o out.txt
 *         [-io g|c|i]        # gather | collective MPI-IO | independent MPI-IO
 *         [-dp fblock|out]   # decomposition policy (default: fblock)
 *         [-m same|full] [-p zero|none|const] [-c cval] [-st stride] [-se seed]
 *
 * Env:
 *   CONV1D_BLOCK_SIZE     integer >0 cap on per-rank OUTPUT block (used by -dp out)
 *   CONV1D_CART_REORDER   0|1 allow MPI rank reordering in Cartesian topology (default 0)
 *   CONV_MPI_IO_HINTS     comma list "key=val,key=val" passed to MPI_File_open info
 *
 * Build:
 *   mpicc -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c
 *   # with debug planning prints:
 *   mpicc -DDEBUG_MPI=1 -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c
 */

#define _POSIX_C_SOURCE 200809L
#ifndef ULTRA_OVERLAP
#define ULTRA_OVERLAP 1 /* set to 0 to disable ultra-fine overlap */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <getopt.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>

/* Toggle verbose per-rank debug logging */
#ifndef DEBUG_MPI
#define DEBUG_MPI 0
#endif

/* ---------- small utils ---------- */
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline int max2(int a, int b) { return (a > b) ? a : b; }
static inline int min2(int a, int b) { return (a < b) ? a : b; }

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
    IO_GATHER = 0,
    IO_COLL = 1,
    IO_INDEP = 2
} io_mode_t;
typedef enum
{
    DP_FBLOCK = 0,
    DP_OUT = 1
} decomp_policy_t; /* fblock = halo (default) */

/* ---------- Fwds ---------- */
static void usage(const char *prog);
static int parse_args(int argc, char **argv,
                      const char **f_path, const char **g_path, const char **o_path,
                      long *N_req, long *K_req,
                      unsigned long *seed, int *have_seed,
                      conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
                      int *stride, io_mode_t *iomode, decomp_policy_t *dpol);

static float *read_array_1d(const char *path, int *len_out);
static void write_array_1d(const char *path, const float *arr, int len);
static float *gen_array_1d(int n);

static void ensure_dir(const char *path);
static void log_metrics(int N, int K, int outLen,
                        conv_mode cmode, pad_mode pmode, float cval, int stride,
                        double conv_time_max, double gflops, int ranks, io_mode_t iomode,
                        double halo_time_max, double total_time_max,
                        int hbL_min, double hbL_avg, int hbL_max, long long hbL_total,
                        int hbR_min, double hbR_avg, int hbR_max, long long hbR_total,
                        double interior_time_max, double border_time_max);

static char *bcast_dup_cstr(const char *s, int root, MPI_Comm comm);
static char *make_bin_path(const char *o_path);
static const char *iomode_str(io_mode_t m);
static const char *dpol_str(decomp_policy_t p);
static MPI_Info build_io_info_from_env(void);

/* ===== Kernels that read from local buffers ===== */

/* SAME (stride-aware) using a single contiguous local view [base_g .. base_g+len_local-1] */
static void conv1d_same_stride_range_local(const float *f_local, int base_g, int len_local,
                                           int N, const float *g, int K, int stride,
                                           pad_mode pmod, float cval,
                                           int n_out_start, int n_out_end, float *out)
{
    const int c = K / 2;
    const int g_lo = base_g;
    const int g_hi = base_g + len_local - 1;

    int pos = 0;
    for (int n_out = n_out_start; n_out < n_out_end; ++n_out, ++pos)
    {
        const int n = n_out * stride;
        double acc = 0.0;
        for (int m = 0; m < K; ++m)
        {
            const int idx = n - (m - c); /* global index into f */
            if (idx >= 0 && idx < N)
            {
                if (idx >= g_lo && idx <= g_hi)
                {
                    acc += (double)f_local[idx - g_lo] * (double)g[m];
                }
                else
                {
                    if (pmod == PAD_CONST)
                        acc += (double)cval * (double)g[m];
                }
            }
            else
            {
                if (pmod == PAD_CONST)
                    acc += (double)cval * (double)g[m];
            }
        }
        out[pos] = (float)acc;
    }
}

/* FULL (stride-aware) using a single contiguous local view */
static void conv1d_full_stride_range_local(const float *f_local, int base_g, int len_local,
                                           int N, const float *g, int K, int stride,
                                           int n_out_start, int n_out_end, float *out)
{
    const int g_lo = base_g;
    const int g_hi = base_g + len_local - 1;

    int pos = 0;
    for (int n_out = n_out_start; n_out < n_out_end; ++n_out, ++pos)
    {
        const int n = n_out * stride;
        double acc = 0.0;

        int i_lo = n - (K - 1);
        if (i_lo < 0)
            i_lo = 0;
        int i_hi = n;
        if (i_hi > N - 1)
            i_hi = N - 1;

        for (int i = i_lo; i <= i_hi; ++i)
        {
            const int m = n - i;
            if (i >= g_lo && i <= g_hi)
            {
                acc += (double)f_local[i - g_lo] * (double)g[m];
            }
        }
        out[pos] = (float)acc;
    }
}

/* ===== Segment-aware kernels for ultra-fine overlap =====
   We pass three segments (left halo, core, right halo) and a base_g that points
   to the global index covered by left[0] (or core[0] if left_len==0).
   These kernels *do not* require an assembled contiguous buffer. */

static inline int seg_pick_and_load(const float *left, int left_len,
                                    const float *core, int core_len,
                                    const float *right, int right_len,
                                    int base_g, int idx_g, float *val_out)
{
    /* returns 1 if idx_g is in any segment and writes *val_out; 0 otherwise */
    int left_lo = base_g;
    int left_hi = base_g + left_len - 1;
    int core_lo = left_lo + left_len;
    int core_hi = core_lo + core_len - 1;
    int right_lo = core_lo + core_len;
    int right_hi = right_lo + right_len - 1;

    if (left_len > 0 && idx_g >= left_lo && idx_g <= left_hi)
    {
        *val_out = left[idx_g - left_lo];
        return 1;
    }
    if (core_len > 0 && idx_g >= core_lo && idx_g <= core_hi)
    {
        *val_out = core[idx_g - core_lo];
        return 1;
    }
    if (right_len > 0 && idx_g >= right_lo && idx_g <= right_hi)
    {
        *val_out = right[idx_g - right_lo];
        return 1;
    }
    return 0;
}

static void conv1d_same_stride_range_segments(const float *left, int left_len,
                                              const float *core, int core_len,
                                              const float *right, int right_len,
                                              int base_g, int N,
                                              const float *g, int K, int stride,
                                              pad_mode pmod, float cval,
                                              int n_out_start, int n_out_end, float *out)
{
    const int c = K / 2;
    int pos = 0;
    for (int n_out = n_out_start; n_out < n_out_end; ++n_out, ++pos)
    {
        const int n = n_out * stride; /* physical 0..N-1 */
        double acc = 0.0;
        for (int m = 0; m < K; ++m)
        {
            const int idx = n - (m - c);
            if (idx >= 0 && idx < N)
            {
                float fv;
                if (seg_pick_and_load(left, left_len, core, core_len, right, right_len, base_g, idx, &fv))
                    acc += (double)fv * (double)g[m];
                else if (pmod == PAD_CONST)
                    acc += (double)cval * (double)g[m];
            }
            else
            {
                if (pmod == PAD_CONST)
                    acc += (double)cval * (double)g[m];
            }
        }
        out[pos] = (float)acc;
    }
}

static void conv1d_full_stride_range_segments(const float *left, int left_len,
                                              const float *core, int core_len,
                                              const float *right, int right_len,
                                              int base_g, int N,
                                              const float *g, int K, int stride,
                                              int n_out_start, int n_out_end, float *out)
{
    int pos = 0;
    for (int n_out = n_out_start; n_out < n_out_end; ++n_out, ++pos)
    {
        const int n = n_out * stride; /* 0..N+K-2 */
        double acc = 0.0;
        int i_lo = n - (K - 1);
        if (i_lo < 0)
            i_lo = 0;
        int i_hi = n;
        if (i_hi > N - 1)
            i_hi = N - 1;
        for (int i = i_lo; i <= i_hi; ++i)
        {
            float fv;
            if (seg_pick_and_load(left, left_len, core, core_len, right, right_len, base_g, i, &fv))
            {
                const int m = n - i;
                acc += (double)fv * (double)g[m];
            }
        }
        out[pos] = (float)acc;
    }
}

/* ---------- Usage ---------- */
static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-f f.txt | -L N | --len N] "
            "[-g g.txt | -kL K | --klen K] "
            "-o out.txt|--out out.txt "
            "[-se seed|--seed seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value] "
            "[-st stride|--stride stride] "
            "[-io g|c|i|--ioformat gather|collective|independent] "
            "[-dp fblock|out|--decomp fblock|out]\n",
            prog);
}

/* ---------- CLI Parsing ---------- */
static int parse_args(int argc, char **argv,
                      const char **f_path, const char **g_path, const char **o_path,
                      long *N_req, long *K_req,
                      unsigned long *seed, int *have_seed,
                      conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
                      int *stride, io_mode_t *iomode, decomp_policy_t *dpol)
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
    *iomode = IO_GATHER;
    *dpol = DP_FBLOCK;

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

        if (!strcmp(a, "-io"))
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-io requires an argument (g|c|i)\n");
                free(fargv);
                return 0;
            }
            const char *v = argv[++i];
            if (!strcmp(v, "g"))
                *iomode = IO_GATHER;
            else if (!strcmp(v, "c"))
                *iomode = IO_COLL;
            else if (!strcmp(v, "i"))
                *iomode = IO_INDEP;
            else
            {
                fprintf(stderr, "invalid -io %s\n", v);
                free(fargv);
                return 0;
            }
            continue;
        }
        if (!strncmp(a, "-io=", 4))
        {
            const char *v = a + 4;
            if (!strcmp(v, "g"))
                *iomode = IO_GATHER;
            else if (!strcmp(v, "c"))
                *iomode = IO_COLL;
            else if (!strcmp(v, "i"))
                *iomode = IO_INDEP;
            else
            {
                fprintf(stderr, "invalid -io %s\n", v);
                free(fargv);
                return 0;
            }
            continue;
        }

        if (!strcmp(a, "-dp"))
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-dp requires an argument (fblock|out)\n");
                free(fargv);
                return 0;
            }
            const char *v = argv[++i];
            if (!strcmp(v, "fblock"))
                *dpol = DP_FBLOCK;
            else if (!strcmp(v, "out"))
                *dpol = DP_OUT;
            else
            {
                fprintf(stderr, "invalid -dp %s\n", v);
                free(fargv);
                return 0;
            }
            continue;
        }
        if (!strncmp(a, "-dp=", 4))
        {
            const char *v = a + 4;
            if (!strcmp(v, "fblock"))
                *dpol = DP_FBLOCK;
            else if (!strcmp(v, "out"))
                *dpol = DP_OUT;
            else
            {
                fprintf(stderr, "invalid -dp %s\n", v);
                free(fargv);
                return 0;
            }
            continue;
        }

        fargv[fargc++] = argv[i];
    }

    static struct option long_opts[] = {
        {"file", required_argument, 0, 'f'},
        {"kernel", required_argument, 0, 'g'},
        {"out", required_argument, 0, 'o'},
        {"seed", required_argument, 0, 's'}, /* legacy short -s */
        {"len", required_argument, 0, 1},
        {"klen", required_argument, 0, 2},
        {"mode", required_argument, 0, 3},
        {"padding", required_argument, 0, 4},
        {"cval", required_argument, 0, 5},
        {"stride", required_argument, 0, 't'},
        {"ioformat", required_argument, 0, 6},
        {"decomp", required_argument, 0, 7},
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
            break; /* legacy -s */
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
            break;
        case 1:
            *N_req = strtol(optarg, NULL, 10);
            break;
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break;
        case 6:
        {
            const char *v = optarg;
            if (!strcmp(v, "gather"))
                *iomode = IO_GATHER;
            else if (!strcmp(v, "collective"))
                *iomode = IO_COLL;
            else if (!strcmp(v, "independent"))
                *iomode = IO_INDEP;
            else
            {
                fprintf(stderr, "invalid --ioformat %s\n", v);
                free(fargv);
                return 0;
            }
            break;
        }
        case 7:
        {
            const char *v = optarg;
            if (!strcmp(v, "fblock"))
                *dpol = DP_FBLOCK;
            else if (!strcmp(v, "out"))
                *dpol = DP_OUT;
            else
            {
                fprintf(stderr, "invalid --decomp %s\n", v);
                free(fargv);
                return 0;
            }
            break;
        }
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

/* ---------- I/O helpers ---------- */
static float *read_array_1d(const char *path, int *len_out)
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
        fprintf(stderr, "bad header in %s (expected positive int length)\n", path);
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

static void write_array_1d(const char *path, const float *arr, int len)
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

static float *gen_array_1d(int n)
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

/* ---------- Misc utils ---------- */
static void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
    {
        (void)mkdir(path, 0775);
    }
}

static const char *iomode_str(io_mode_t m)
{
    switch (m)
    {
    case IO_GATHER:
        return "gather";
    case IO_COLL:
        return "collective";
    case IO_INDEP:
        return "independent";
    default:
        return "unknown";
    }
}

static const char *dpol_str(decomp_policy_t p)
{
    return (p == DP_FBLOCK) ? "fblock" : "out";
}

/* CSV logger with extended stats */
static void log_metrics(int N, int K, int outLen,
                        conv_mode cmode, pad_mode pmode, float cval, int stride,
                        double conv_time_max, double gflops, int ranks, io_mode_t iomode,
                        double halo_time_max, double total_time_max,
                        int hbL_min, double hbL_avg, int hbL_max, long long hbL_total,
                        int hbR_min, double hbR_avg, int hbR_max, long long hbR_total,
                        double interior_time_max, double border_time_max)
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
    fprintf(csv,
            "RunID,N,K,outLen,mode,padding,cval,stride,conv_time_max,halo_time_max,total_time_max,gflops,ranks,io,"
            "hbL_min,hbL_avg,hbL_max,hbL_total,hbR_min,hbR_avg,hbR_max,hbR_total,interior_time_max,border_time_max\n");
    fprintf(csv,
            "%s,%d,%d,%d,%s,%s,%.9g,%d,%.9f,%.9f,%.9f,%.6f,%d,%s,"
            "%d,%.6f,%d,%lld,%d,%.6f,%d,%lld,%.9f,%.9f\n",
            runid, N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            stride, conv_time_max, halo_time_max, total_time_max, gflops, ranks, iomode_str(iomode),
            hbL_min, hbL_avg, hbL_max, (long long)hbL_total,
            hbR_min, hbR_avg, hbR_max, (long long)hbR_total,
            interior_time_max, border_time_max);
    fclose(csv);
}

static char *bcast_dup_cstr(const char *s, int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int len = 0;
    if (rank == root)
        len = (int)strlen(s);
    MPI_Bcast(&len, 1, MPI_INT, root, comm);
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf)
    {
        fprintf(stderr, "[rank %d] OOM bcast filename\n", rank);
        MPI_Abort(comm, 1);
    }
    if (rank == root)
        memcpy(buf, s, (size_t)len);
    MPI_Bcast(buf, len, MPI_CHAR, root, comm);
    buf[len] = '\0';
    return buf;
}

static char *make_bin_path(const char *o_path)
{
    size_t L = strlen(o_path);
    char *p = (char *)malloc(L + 5);
    if (!p)
    {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    memcpy(p, o_path, L);
    memcpy(p + L, ".bin", 5);
    return p;
}

static void parse_kv_and_set(MPI_Info info, const char *kv)
{
    const char *eq = strchr(kv, '=');
    if (!eq)
        return;
    size_t klen = (size_t)(eq - kv);
    if (klen == 0)
        return;
    char key[128], val[256];
    if (klen >= sizeof(key))
        return;
    memcpy(key, kv, klen);
    key[klen] = '\0';
    strncpy(val, eq + 1, sizeof(val) - 1);
    val[sizeof(val) - 1] = '\0';
    if (key[0] && val[0])
        MPI_Info_set(info, key, val);
}

static MPI_Info build_io_info_from_env(void)
{
    const char *h = getenv("CONV_MPI_IO_HINTS");
    if (!h || !*h)
        return MPI_INFO_NULL;

    MPI_Info info;
    MPI_Info_create(&info);
    const char *s = h;
    while (*s)
    {
        const char *c = strchr(s, ',');
        if (c)
        {
            char tmp[384];
            size_t len = (size_t)(c - s);
            if (len >= sizeof(tmp))
                len = sizeof(tmp) - 1;
            memcpy(tmp, s, len);
            tmp[len] = '\0';
            parse_kv_and_set(info, tmp);
            s = c + 1;
        }
        else
        {
            parse_kv_and_set(info, s);
            break;
        }
    }
    return info;
}

/* ---------- Main ---------- */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Cartesian 1-D communicator (optional reorder via env) */
    int reorder = 0;
    const char *env_reo = getenv("CONV1D_CART_REORDER");
    if (env_reo && *env_reo)
        reorder = atoi(env_reo) ? 1 : 0;

    MPI_Comm cart_comm = MPI_COMM_NULL;
    {
        int dims[1] = {0};
        MPI_Dims_create(size, 1, dims);
        int periods[1] = {0};
        MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &cart_comm);
    }
    int left = MPI_PROC_NULL, right = MPI_PROC_NULL;
    if (cart_comm != MPI_COMM_NULL)
        MPI_Cart_shift(cart_comm, 0, 1, &left, &right);

    /* Parse CLI on rank 0, broadcast config */
    const char *f_path = NULL, *g_path = NULL, *o_path = NULL;
    long N_req = -1, K_req = -1;
    unsigned long seed = (unsigned long)time(NULL);
    int have_seed = 0;
    conv_mode cmode = MODE_SAME;
    pad_mode pmode = PAD_ZERO;
    float cval = 0.0f;
    int have_cval = 0;
    int stride = 1;
    io_mode_t iomode = IO_GATHER;
    decomp_policy_t dpol = DP_FBLOCK;

    int ok = 1;
    if (rank == 0)
    {
        ok = parse_args(argc, argv, &f_path, &g_path, &o_path,
                        &N_req, &K_req, &seed, &have_seed,
                        &cmode, &pmode, &cval, &have_cval,
                        &stride, &iomode, &dpol);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok)
    {
        if (cart_comm != MPI_COMM_NULL)
            MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int cmode_i = 0, pmode_i = 0, iomode_i = 0, dpol_i = 0;
    if (rank == 0)
    {
        cmode_i = (int)cmode;
        pmode_i = (int)pmode;
        iomode_i = (int)iomode;
        dpol_i = (int)dpol;
    }
    MPI_Bcast(&cmode_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pmode_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stride, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iomode_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dpol_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cmode = (conv_mode)cmode_i;
    pmode = (pad_mode)pmode_i;
    iomode = (io_mode_t)iomode_i;
    dpol = (decomp_policy_t)dpol_i;

    /* Root obtains arrays / lengths */
    int N = 0, K = 0;
    float *f0 = NULL, *g = NULL;

    if (rank == 0)
    {
        srand((unsigned)seed);
        if (f_path)
            f0 = read_array_1d(f_path, &N);
        else
        {
            N = (int)N_req;
            f0 = gen_array_1d(N);
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
        if (f0)
            free(f0);
        if (g)
            free(g);
        if (cart_comm != MPI_COMM_NULL)
            MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Broadcast kernel g to all (small compared to f) */
    if (rank != 0)
    {
        g = (float *)malloc((size_t)K * sizeof(float));
        if (!g)
        {
            fprintf(stderr, "[rank %d] OOM kernel\n", rank);
            if (cart_comm)
                MPI_Comm_free(&cart_comm);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }
    MPI_Bcast(g, K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Compute outLen (stride-aware) */
    const int outLen = (cmode == MODE_FULL) ? ceil_div(N + K - 1, stride)
                                            : ceil_div(N, stride);
    if (size > outLen && rank == 0)
    {
        fprintf(stderr, "note: ranks=%d > outLen=%d; some ranks will be idle for output.\n", size, outLen);
    }

    /* OUTPUT decomposition arrays (displs/counts) usable for gather/MPI-IO view */
    int *y_counts = (int *)calloc((size_t)size, sizeof(int));
    int *y_displs = (int *)calloc((size_t)size, sizeof(int));
    if (!y_counts || !y_displs)
    {
        fprintf(stderr, "[rank %d] OOM y_counts/displs\n", rank);
        free(y_counts);
        free(y_displs);
        if (f0)
            free(f0);
        if (g)
            free(g);
        if (cart_comm != MPI_COMM_NULL)
            MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    float *out_local = NULL;

    /* Local f buffers (DP_FBLOCK) */
    float *f_core = NULL;
    float *halo_left = NULL, *halo_right = NULL;
    float *f_local = NULL; /* only used when ULTRA_OVERLAP==0 (assemble) */
    int f_core_len = 0;
    int f_core_start = 0;
    int halo_left_len = 0, halo_right_len = 0;
    int base_g = 0;    /* global index covered by left[0] (or core[0] if no left) */
    int len_local = 0; /* only for assembled path */
    double halo_time = 0.0;
    int halo_bytes_left_local = 0;
    int halo_bytes_right_local = 0;
    double total_time = 0.0;
    double total_t0 = 0.0;

    double interior_time = 0.0, border_time = 0.0;

    if (dpol == DP_FBLOCK)
    {
        /* f-domain split */
        int *f_counts = (int *)malloc((size_t)size * sizeof(int));
        int *f_displs = (int *)malloc((size_t)size * sizeof(int));
        if (!f_counts || !f_displs)
        {
            fprintf(stderr, "[rank %d] OOM f_counts/displs\n", rank);
            free(f_counts);
            free(f_displs);
            free(y_counts);
            free(y_displs);
            if (f0)
                free(f0);
            if (g)
                free(g);
            if (cart_comm)
                MPI_Comm_free(&cart_comm);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        const int base = N / size, rem = N % size;
        for (int r = 0; r < size; ++r)
            f_counts[r] = base + (r < rem ? 1 : 0);
        f_displs[0] = 0;
        for (int r = 1; r < size; ++r)
            f_displs[r] = f_displs[r - 1] + f_counts[r - 1];

        f_core_len = f_counts[rank];
        f_core_start = f_displs[rank];

        if (f_core_len > 0)
        {
            f_core = (float *)malloc((size_t)f_core_len * sizeof(float));
            if (!f_core)
            {
                fprintf(stderr, "[rank %d] OOM f_core\n", rank);
                free(f_counts);
                free(f_displs);
                free(y_counts);
                free(y_displs);
                if (f0)
                    free(f0);
                if (g)
                    free(g);
                if (cart_comm)
                    MPI_Comm_free(&cart_comm);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        }

        if (rank == 0)
        {
            MPI_Scatterv(f0, f_counts, f_displs, MPI_FLOAT,
                         (f_core_len ? f_core : NULL), f_core_len, MPI_FLOAT,
                         0, MPI_COMM_WORLD);
            free(f0);
            f0 = NULL;
        }
        else
        {
            MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT,
                         (f_core_len ? f_core : NULL), f_core_len, MPI_FLOAT,
                         0, MPI_COMM_WORLD);
        }

        /* halo sizes (up to K-1) */
        const int max_halo = (K > 0) ? (K - 1) : 0;
        halo_left_len = min2(f_core_start, max_halo);
        halo_right_len = min2(N - (f_core_start + f_core_len), max_halo);

        if (halo_left_len > 0)
            halo_left = (float *)malloc((size_t)halo_left_len * sizeof(float));
        if (halo_right_len > 0)
            halo_right = (float *)malloc((size_t)halo_right_len * sizeof(float));
        if ((halo_left_len > 0 && !halo_left) ||
            (halo_right_len > 0 && !halo_right))
        {
            fprintf(stderr, "[rank %d] OOM halos\n", rank);
            if (halo_left)
                free(halo_left);
            if (halo_right)
                free(halo_right);
            if (f_core)
                free(f_core);
            free(f_counts);
            free(f_displs);
            free(y_counts);
            free(y_displs);
            if (g)
                free(g);
            if (cart_comm)
                MPI_Comm_free(&cart_comm);
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        /* Base of segments for segment-aware kernels */
        base_g = f_core_start - halo_left_len;

        /* ====== Asynchronous halo exchange with warm-up ====== */
        float dummy = 0.0f;
        const int send_left_count = min2(f_core_len, halo_left_len);
        const int send_right_count = min2(f_core_len, halo_right_len);

        void *sendL_buf = (send_left_count > 0) ? (void *)f_core : (void *)&dummy;
        void *sendR_buf = (send_right_count > 0) ? (void *)(f_core + (f_core_len - send_right_count)) : (void *)&dummy;
        void *recvL_buf = (halo_left_len > 0) ? (void *)halo_left : (void *)&dummy;
        void *recvR_buf = (halo_right_len > 0) ? (void *)halo_right : (void *)&dummy;

        /* Warm-up */
        {
            MPI_Request warmup_reqs[4];
            int wc = 0;
            if (halo_right_len > 0)
                MPI_Irecv(recvR_buf, halo_right_len, MPI_FLOAT, right, 800, MPI_COMM_WORLD, &warmup_reqs[wc++]);
            if (halo_left_len > 0)
                MPI_Irecv(recvL_buf, halo_left_len, MPI_FLOAT, left, 801, MPI_COMM_WORLD, &warmup_reqs[wc++]);
            if (send_left_count > 0)
                MPI_Isend(sendL_buf, send_left_count, MPI_FLOAT, left, 800, MPI_COMM_WORLD, &warmup_reqs[wc++]);
            if (send_right_count > 0)
                MPI_Isend(sendR_buf, send_right_count, MPI_FLOAT, right, 801, MPI_COMM_WORLD, &warmup_reqs[wc++]);
            if (wc > 0)
                MPI_Waitall(wc, warmup_reqs, MPI_STATUSES_IGNORE);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        /* Start end-to-end timing */
        total_t0 = MPI_Wtime();

        /* Timed halo exchange */
        double halo_t0 = MPI_Wtime();
        MPI_Request req_recv_left = MPI_REQUEST_NULL, req_recv_right = MPI_REQUEST_NULL;
        MPI_Request req_send_left = MPI_REQUEST_NULL, req_send_right = MPI_REQUEST_NULL;

        if (halo_right_len > 0)
            MPI_Irecv(recvR_buf, halo_right_len, MPI_FLOAT, right, 800, MPI_COMM_WORLD, &req_recv_right);
        if (halo_left_len > 0)
            MPI_Irecv(recvL_buf, halo_left_len, MPI_FLOAT, left, 801, MPI_COMM_WORLD, &req_recv_left);
        if (send_left_count > 0)
            MPI_Isend(sendL_buf, send_left_count, MPI_FLOAT, left, 800, MPI_COMM_WORLD, &req_send_left);
        if (send_right_count > 0)
            MPI_Isend(sendR_buf, send_right_count, MPI_FLOAT, right, 801, MPI_COMM_WORLD, &req_send_right);

        /* Determine output distribution (balanced, then trimmed to feasible) */
        int base_out = outLen / size;
        int rem_out = outLen % size;
        int my_out_start = rank * base_out + (rank < rem_out ? rank : rem_out);
        int my_out_count = base_out + (rank < rem_out ? 1 : 0);
        int my_out_end = my_out_start + my_out_count;

        /* Possible physical n supported by my core+halos */
        const int a = f_core_start;
        const int b = f_core_start + f_core_len - 1;
        const int c = K / 2;

        int n_phys_lo = 0, n_phys_hi = -1;
        if (cmode == MODE_SAME)
        {
            int nmin = max2(0, a + (K - 1 - c));
            int nmax = min2(N - 1, b - c);
            if (nmin <= nmax)
            {
                n_phys_lo = nmin;
                n_phys_hi = nmax;
            }
        }
        else
        {
            int nmin = max2(0, a + (K - 1));
            int nmax = min2(b, N + K - 2);
            if (nmin <= nmax)
            {
                n_phys_lo = nmin;
                n_phys_hi = nmax;
            }
        }
        int interior_out_start = -1, interior_out_end = -1;
        if (n_phys_lo <= n_phys_hi)
        {
            interior_out_start = (n_phys_lo + (stride - 1)) / stride;
            interior_out_end = n_phys_hi / stride;
        }

        int n_lo_possible, n_hi_possible;
        if (cmode == MODE_SAME)
        {
            n_lo_possible = a;
            n_hi_possible = b;
            if (n_lo_possible < 0)
                n_lo_possible = 0;
            if (n_hi_possible > N - 1)
                n_hi_possible = N - 1;
        }
        else
        {
            n_lo_possible = a - (K - 1);
            n_hi_possible = b + (K - 1);
            if (n_lo_possible < 0)
                n_lo_possible = 0;
            if (n_hi_possible > (N + K - 2))
                n_hi_possible = (N + K - 2);
        }
        int out_s_possible = (n_lo_possible + (stride - 1)) / stride;
        int out_e_possible = n_hi_possible / stride;

        int out_s_actual = max2(my_out_start, max2(0, out_s_possible));
        int out_e_actual = min2(my_out_end - 1, min2(outLen - 1, out_e_possible));
        int my_final_count = (out_e_actual >= out_s_actual) ? (out_e_actual - out_s_actual + 1) : 0;

        int my_s_tmp = (my_final_count > 0) ? out_s_actual : -1;
        int my_e_tmp = (my_final_count > 0) ? out_e_actual : -1;

        int *starts = (int *)malloc((size_t)size * sizeof(int));
        int *ends = (int *)malloc((size_t)size * sizeof(int));
        MPI_Allgather(&my_s_tmp, 1, MPI_INT, starts, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&my_e_tmp, 1, MPI_INT, ends, 1, MPI_INT, MPI_COMM_WORLD);

        int *final_st = (int *)malloc((size_t)size * sizeof(int));
        int *final_ct = (int *)malloc((size_t)size * sizeof(int));
        int cur = 0;
        for (int r = 0; r < size; ++r)
        {
            int s = starts[r], e = ends[r];
            if (s < 0 || e < 0 || s > e)
            {
                final_st[r] = cur;
                final_ct[r] = 0;
                continue;
            }
            if (s < cur)
                s = cur;
            if (e > outLen - 1)
                e = outLen - 1;
            int ct = (e >= s) ? (e - s + 1) : 0;
            final_st[r] = cur;
            final_ct[r] = ct;
            cur += ct;
        }
        if (cur < outLen)
            final_ct[size - 1] += (outLen - cur);

        for (int r = 0; r < size; ++r)
        {
            y_displs[r] = final_st[r];
            y_counts[r] = final_ct[r];
        }

        free(starts);
        free(ends);
        free(final_st);
        free(final_ct);

        /* Allocate local output */
        if (y_counts[rank] > 0)
        {
            out_local = (float *)malloc((size_t)y_counts[rank] * sizeof(float));
            if (!out_local)
            {
                fprintf(stderr, "[rank %d] OOM out_local\n", rank);
                if (halo_left)
                    free(halo_left);
                if (halo_right)
                    free(halo_right);
                if (f_core)
                    free(f_core);
                free(y_counts);
                free(y_displs);
                if (g)
                    free(g);
                if (cart_comm)
                    MPI_Comm_free(&cart_comm);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
            memset(out_local, 0, (size_t)y_counts[rank] * sizeof(float));
        }

        /* === Compute interior while halos in flight === */
        double t_in0 = MPI_Wtime();
        int my_os = y_displs[rank];
        int my_oe = my_os + y_counts[rank];

        int int_s = -1, int_e = -1; /* inclusive interior within my range */
        if (interior_out_start != -1)
        {
            int_s = max2(my_os, interior_out_start);
            int_e = min2(my_oe - 1, interior_out_end);
        }
        if (y_counts[rank] > 0 && int_s != -1 && int_s <= int_e)
        {
            if (cmode == MODE_SAME)
            {
                conv1d_same_stride_range_local(
                    f_core, f_core_start, f_core_len,
                    N, g, K, stride, pmode, cval,
                    int_s, int_e + 1,
                    out_local + (int_s - my_os));
            }
            else
            {
                conv1d_full_stride_range_local(
                    f_core, f_core_start, f_core_len,
                    N, g, K, stride,
                    int_s, int_e + 1,
                    out_local + (int_s - my_os));
            }
        }
        interior_time = MPI_Wtime() - t_in0;

        /* === Borders: ultra-fine overlap (segment-aware) === */
        double t_b0 = MPI_Wtime();

#if ULTRA_OVERLAP
        if (y_counts[rank] > 0)
        {
            /* If no interior, we must wait for both halos (single pass) */
            if (int_s == -1 || int_s > int_e)
            {
                if (halo_left_len > 0)
                    MPI_Wait(&req_recv_left, MPI_STATUS_IGNORE);
                if (halo_right_len > 0)
                    MPI_Wait(&req_recv_right, MPI_STATUS_IGNORE);

                if (cmode == MODE_SAME)
                {
                    conv1d_same_stride_range_segments(
                        halo_left, halo_left_len, f_core, f_core_len, halo_right, halo_right_len,
                        base_g, N, g, K, stride, pmode, cval,
                        my_os, my_oe, out_local);
                }
                else
                {
                    conv1d_full_stride_range_segments(
                        halo_left, halo_left_len, f_core, f_core_len, halo_right, halo_right_len,
                        base_g, N, g, K, stride,
                        my_os, my_oe, out_local);
                }
            }
            else
            {
                /* Left border: [my_os .. int_s) */
                if (my_os < int_s)
                {
                    if (halo_left_len > 0)
                        MPI_Wait(&req_recv_left, MPI_STATUS_IGNORE);
                    if (cmode == MODE_SAME)
                    {
                        conv1d_same_stride_range_segments(
                            halo_left, halo_left_len, f_core, f_core_len, halo_right, halo_right_len,
                            base_g, N, g, K, stride, pmode, cval,
                            my_os, int_s, out_local + 0);
                    }
                    else
                    {
                        conv1d_full_stride_range_segments(
                            halo_left, halo_left_len, f_core, f_core_len, halo_right, halo_right_len,
                            base_g, N, g, K, stride,
                            my_os, int_s, out_local + 0);
                    }
                }
                /* Right border: (int_e .. my_oe-1] -> [int_e+1 .. my_oe) */
                if (int_e + 1 < my_oe)
                {
                    if (halo_right_len > 0)
                        MPI_Wait(&req_recv_right, MPI_STATUS_IGNORE);
                    if (cmode == MODE_SAME)
                    {
                        conv1d_same_stride_range_segments(
                            halo_left, halo_left_len, f_core, f_core_len, halo_right, halo_right_len,
                            base_g, N, g, K, stride, pmode, cval,
                            int_e + 1, my_oe, out_local + (int_e + 1 - my_os));
                    }
                    else
                    {
                        conv1d_full_stride_range_segments(
                            halo_left, halo_left_len, f_core, f_core_len, halo_right, halo_right_len,
                            base_g, N, g, K, stride,
                            int_e + 1, my_oe, out_local + (int_e + 1 - my_os));
                    }
                }
            }
        }
#else
        /* Fallback: assemble contiguous [halo_left|core|halo_right] then call local kernels */
        if (y_counts[rank] > 0)
        {
            if (halo_left_len > 0)
                MPI_Wait(&req_recv_left, MPI_STATUS_IGNORE);
            if (halo_right_len > 0)
                MPI_Wait(&req_recv_right, MPI_STATUS_IGNORE);

            len_local = halo_left_len + f_core_len + halo_right_len;
            if (len_local > 0)
            {
                f_local = (float *)malloc((size_t)len_local * sizeof(float));
                if (!f_local)
                {
                    fprintf(stderr, "[rank %d] OOM f_local\n", rank);
                    if (halo_left)
                        free(halo_left);
                    if (halo_right)
                        free(halo_right);
                    if (f_core)
                        free(f_core);
                    free(y_counts);
                    free(y_displs);
                    if (g)
                        free(g);
                    if (cart_comm)
                        MPI_Comm_free(&cart_comm);
                    MPI_Finalize();
                    return EXIT_FAILURE;
                }
                int off = 0;
                if (halo_left_len > 0)
                {
                    memcpy(f_local + off, halo_left, (size_t)halo_left_len * sizeof(float));
                    off += halo_left_len;
                }
                if (f_core_len > 0)
                {
                    memcpy(f_local + off, f_core, (size_t)f_core_len * sizeof(float));
                    off += f_core_len;
                }
                if (halo_right_len > 0)
                {
                    memcpy(f_local + off, halo_right, (size_t)halo_right_len * sizeof(float));
                    off += halo_right_len;
                }
            }
            const int base_local = base_g;
            if (cmode == MODE_SAME)
            {
                conv1d_same_stride_range_local(
                    f_local, base_local, len_local, N, g, K, stride, pmode, cval,
                    my_os, my_oe, out_local);
            }
            else
            {
                conv1d_full_stride_range_local(
                    f_local, base_local, len_local, N, g, K, stride,
                    my_os, my_oe, out_local);
            }
        }
#endif
        border_time = MPI_Wtime() - t_b0;

        /* Complete sends */
        if (send_left_count > 0)
            MPI_Wait(&req_send_left, MPI_STATUS_IGNORE);
        if (send_right_count > 0)
            MPI_Wait(&req_send_right, MPI_STATUS_IGNORE);

        halo_time = MPI_Wtime() - halo_t0;

        /* Record message sizes */
        int bytes_send_left = send_left_count * (int)sizeof(float);
        int bytes_recv_right = halo_right_len * (int)sizeof(float);
        int bytes_send_right = send_right_count * (int)sizeof(float);
        int bytes_recv_left = halo_left_len * (int)sizeof(float);
        halo_bytes_left_local = bytes_send_left + bytes_recv_left;
        halo_bytes_right_local = bytes_send_right + bytes_recv_right;

#if DEBUG_MPI
        fprintf(stderr, "[rank %d/%d] neighbors: left=%d right=%d | f_core=[%d..%d] halos L=%d R=%d | base_g=%d\n",
                rank, size, left, right, f_core_start, f_core_start + f_core_len - 1,
                halo_left_len, halo_right_len, base_g);
        if (y_counts[rank] > 0)
        {
            fprintf(stderr, "[plan] rank %d: out_count=%d (offset=%d)\n",
                    rank, y_counts[rank], y_displs[rank]);
        }
#endif

        total_time = MPI_Wtime() - total_t0;

        /* Free temporary buffers we no longer need */
        if (halo_left)
        {
            free(halo_left);
            halo_left = NULL;
        }
        if (halo_right)
        {
            free(halo_right);
            halo_right = NULL;
        }
#if !ULTRA_OVERLAP
        if (f_local)
        {
            free(f_local);
            f_local = NULL;
        }
#endif
        if (f_core)
        {
            free(f_core);
            f_core = NULL;
        }
    }
    else
    { /* DP_OUT: broadcast full f and block by output */
        const int base = (outLen > 0) ? outLen / size : 0;
        const int rem = (outLen > 0) ? outLen % size : 0;
        for (int r = 0; r < size; r++)
            y_counts[r] = base + (r < rem ? 1 : 0);
        y_displs[0] = 0;
        for (int r = 1; r < size; r++)
            y_displs[r] = y_displs[r - 1] + y_counts[r - 1];

        int bs = 0;
        const char *env_bs = getenv("CONV1D_BLOCK_SIZE");
        if (env_bs && *env_bs)
        {
            bs = atoi(env_bs);
            if (bs < 0)
                bs = 0;
        }
        if (bs > 0)
        {
            for (int r = 0; r < size; ++r)
            {
                int start = y_displs[r];
                int remaining = outLen - start;
                y_counts[r] = (remaining > 0) ? (remaining < bs ? remaining : bs) : 0;
            }
        }

        float *f_full = NULL;
        if (rank == 0)
            f_full = f0;
        if (rank != 0)
        {
            f_full = (float *)malloc((size_t)N * sizeof(float));
            if (!f_full)
            {
                fprintf(stderr, "[rank %d] OOM f_full\n", rank);
                if (g)
                    free(g);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        }
        MPI_Bcast(f_full, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (rank == 0)
            f0 = NULL;

        if (y_counts[rank] > 0)
        {
            out_local = (float *)malloc((size_t)y_counts[rank] * sizeof(float));
            if (!out_local)
            {
                fprintf(stderr, "[rank %d] OOM out_local\n", rank);
                if (f_full)
                    free(f_full);
                free(y_counts);
                free(y_displs);
                if (g)
                    free(g);
                if (cart_comm != MPI_COMM_NULL)
                    MPI_Comm_free(&cart_comm);
                MPI_Finalize();
                return EXIT_FAILURE;
            }
            memset(out_local, 0, (size_t)y_counts[rank] * sizeof(float));
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        int my_os = y_displs[rank];
        int my_oe = my_os + y_counts[rank];
        if (y_counts[rank] > 0)
        {
            if (cmode == MODE_FULL)
            {
                conv1d_full_stride_range_local(
                    f_full, 0, N, N, g, K, stride,
                    my_os, my_oe, out_local);
            }
            else
            {
                conv1d_same_stride_range_local(
                    f_full, 0, N, N, g, K, stride, pmode, cval,
                    my_os, my_oe, out_local);
            }
        }
        const double secs = MPI_Wtime() - t0;
        interior_time = secs;
        border_time = 0.0;
        halo_time = 0.0;
        total_time = secs;

        if (f_full)
            free(f_full);
    }

    /* ============================ REDUCTIONS & REPORT ============================ */
    double conv_time_local = interior_time + border_time;

    double conv_t_min = 0, conv_t_max = 0, conv_t_sum = 0;
    double h_min = 0, h_max = 0, h_sum = 0;
    double tt_min = 0, tt_max = 0, tt_sum = 0;
    MPI_Reduce(&conv_time_local, &conv_t_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&conv_time_local, &conv_t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&conv_time_local, &conv_t_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&halo_time, &h_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&halo_time, &h_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&halo_time, &h_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &tt_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &tt_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &tt_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Checksum = sum of abs(out) across all ranks */
    double local_sum_abs = 0.0;
    if (out_local)
    {
        for (int i = 0; i < y_counts[rank]; ++i)
            local_sum_abs += fabs((double)out_local[i]);
    }
    double sum_abs_tot = 0;
    MPI_Reduce(&local_sum_abs, &sum_abs_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Halo bytes stats */
    int hbL_min_local = halo_bytes_left_local;
    int hbL_max_local = halo_bytes_left_local;
    int hbR_min_local = halo_bytes_right_local;
    int hbR_max_local = halo_bytes_right_local;
    long long hbL_total_local = (long long)halo_bytes_left_local;
    long long hbR_total_local = (long long)halo_bytes_right_local;

    int hbL_min = 0, hbL_max = 0, hbR_min = 0, hbR_max = 0;
    long long hbL_tot = 0, hbR_tot = 0;
    int hbL_sum = 0, hbR_sum = 0;
    MPI_Reduce(&hbL_min_local, &hbL_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hbL_max_local, &hbL_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hbR_min_local, &hbR_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hbR_max_local, &hbR_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hbL_total_local, &hbL_tot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hbR_total_local, &hbR_tot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&halo_bytes_left_local, &hbL_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&halo_bytes_right_local, &hbR_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    double hbL_avg = 0.0, hbR_avg = 0.0;
    if (rank == 0)
    {
        hbL_avg = (double)hbL_sum / (double)size;
        hbR_avg = (double)hbR_sum / (double)size;
    }

    /* interior/border times stats */
    double interior_min = 0, interior_max = 0, interior_sum = 0;
    double border_min = 0, border_max = 0, border_sum = 0;
    MPI_Reduce(&interior_time, &interior_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&interior_time, &interior_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&interior_time, &interior_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&border_time, &border_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&border_time, &border_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&border_time, &border_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* FLOPs and report */
    if (rank == 0)
    {
        double flops = 2.0 * (double)outLen * (double)K;
        double gflops = (conv_t_max > 0.0) ? (flops / conv_t_max / 1e9) : 0.0;
        double conv_t_avg = conv_t_sum / (double)size;
        double h_avg = h_sum / (double)size;
        double tt_avg = tt_sum / (double)size;
        double comm_overhead_pct = (conv_t_max > 0.0) ? (h_max / conv_t_max) * 100.0 : 0.0;
        double total_efficiency = (tt_max > 0.0) ? (conv_t_max / tt_max) * 100.0 : 0.0;

        fprintf(stderr,
                "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g stride=%d | ranks=%d | "
                "conv_time[min/avg/max]=%.6e/%.6e/%.6e s | %.3f GFLOP/s | io=%s | decomp=%s | checksum=%.9g\n",
                N, K, outLen,
                (cmode == MODE_FULL ? "full" : "same"),
                (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                (pmode == PAD_CONST ? cval : 0.0),
                stride, size, conv_t_min, conv_t_avg, conv_t_max, gflops, iomode_str(iomode), dpol_str(dpol), sum_abs_tot);

        if (dpol == DP_FBLOCK)
        {
            fprintf(stderr, "[comm] halo_time[min/avg/max]=%.6e/%.6e/%.6e s | overhead=%.1f%% vs computation\n",
                    h_min, h_avg, h_max, comm_overhead_pct);
            fprintf(stderr, "[overlap] interior[min/avg/max]=%.6e/%.6e/%.6e s | border[min/avg/max]=%.6e/%.6e/%.6e s\n",
                    interior_min, interior_sum / (double)size, interior_max,
                    border_min, border_sum / (double)size, border_max);
            fprintf(stderr, "[msg_sizes] left(min/avg/max,total)=(%d/%.1f/%d,%lld) | right(min/avg/max,total)=(%d/%.1f/%d,%lld)\n",
                    hbL_min, hbL_avg, hbL_max, (long long)hbL_tot,
                    hbR_min, hbR_avg, hbR_max, (long long)hbR_tot);
        }
        fprintf(stderr, "[total] end-to-end[min/avg/max]=%.6e/%.6e/%.6e s | computation_efficiency=%.1f%%\n",
                tt_min, tt_avg, tt_max, total_efficiency);

        log_metrics(N, K, outLen, cmode, pmode, cval, stride,
                    conv_t_max, gflops, size, iomode,
                    h_max, tt_max,
                    hbL_min, hbL_avg, hbL_max, hbL_tot,
                    hbR_min, hbR_avg, hbR_max, hbR_tot,
                    interior_max, border_max);
    }

    /* ============================ OUTPUT ============================ */
    if (iomode == IO_GATHER)
    {
        float *out = NULL;
        if (rank == 0)
        {
            out = (float *)malloc((size_t)outLen * sizeof(float));
            if (!out)
                fprintf(stderr, "[rank 0] OOM out gather\n");
        }

        MPI_Gatherv(out_local, y_counts[rank], MPI_FLOAT,
                    out, y_counts, y_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            write_array_1d(o_path, out, outLen);
        }
        if (out)
            free(out);
    }
    else
    {
        /* MPI-IO with file views: [int32 outLen][float out[]] */
        char *o_text = NULL;
        if (rank == 0)
            o_text = (char *)o_path;
        char *o_text_all = bcast_dup_cstr(o_text, 0, MPI_COMM_WORLD);
        char *bin_path = make_bin_path(o_text_all);

        MPI_Info io_info = build_io_info_from_env();

        MPI_File fh;
        int amode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
        int rc = MPI_File_open(MPI_COMM_WORLD, bin_path, amode, io_info, &fh);
        if (io_info != MPI_INFO_NULL)
            MPI_Info_free(&io_info);

        if (rc != MPI_SUCCESS)
        {
            if (rank == 0)
                fprintf(stderr, "MPI_File_open failed for %s\n", bin_path);
            free(o_text_all);
            free(bin_path);
            if (out_local)
                free(out_local);
            free(y_counts);
            free(y_displs);
            if (g)
                free(g);
            if (cart_comm)
                MPI_Comm_free(&cart_comm);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        if (rank == 0)
        {
            MPI_Status st;
            (void)MPI_File_write_at(fh, 0, (void *)&outLen, 1, MPI_INT, &st);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Datatype my_view_type = MPI_DATATYPE_NULL;

        if (y_counts[rank] > 0)
        {
            MPI_Aint blk_disp = (MPI_Aint)y_displs[rank] * (MPI_Aint)sizeof(float);
            int blocklength = y_counts[rank];
            MPI_Type_create_hindexed(1, &blocklength, &blk_disp, MPI_FLOAT, &my_view_type);
            MPI_Type_commit(&my_view_type);
        }
        else
        {
            MPI_Type_contiguous(0, MPI_FLOAT, &my_view_type);
            MPI_Type_commit(&my_view_type);
        }

        MPI_File_set_view(fh, (MPI_Offset)sizeof(int), MPI_FLOAT, my_view_type, "native", MPI_INFO_NULL);

        float io_dummy = 0.0f;
        void *io_buf = (y_counts[rank] > 0) ? (void *)out_local : (void *)&io_dummy;

        if (iomode == IO_COLL)
        {
            (void)MPI_File_write_all(fh, io_buf, y_counts[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
        }
        else
        {
            if (y_counts[rank] > 0)
            {
                (void)MPI_File_write(fh, io_buf, y_counts[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
            }
            MPI_File_sync(fh);
        }

        MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
        MPI_File_close(&fh);

        if (my_view_type != MPI_DATATYPE_NULL)
            MPI_Type_free(&my_view_type);

        free(o_text_all);
        free(bin_path);
    }

    /* Cleanup */
    if (out_local)
        free(out_local);
    free(y_counts);
    free(y_displs);
    if (g)
        free(g);
    if (cart_comm != MPI_COMM_NULL)
        MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}