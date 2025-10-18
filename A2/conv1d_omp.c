/**
 * @file conv1d_omp.c
 * @brief OpenMP-enabled 1-D Convolution with STRIDE, enhanced kernels, smarter chunking, GFLOPS metrics,
 *        assignment-style I/O, RNG, and CLI controls for threads, schedule, and chunk.
 *
 * This program implements TRUE 1-D convolution (y = f * g) with:
 *   • Output length modes (-m/--mode): same (default) or full.
 *   • Padding policies (-p/--padding) for SAME mode: zero (default), none, const (-c/--cval).
 *   • Inputs from files (-f/-g) or generated uniformly in [-1,1] (-L/-kL, optional -se/--seed).
 *   • Accurate timing of ONLY the convolution kernel (I/O/RNG excluded).
 *   • Per-run CSV metrics under metrics/o0/, uniquely named for SLURM vs local runs.
 *   • OpenMP parallelization with deterministic defaults and CLI overrides:
 *        -t, --threads <int>          : number of threads (default: 1)
 *        -S, --schedule <kind>        : static|dynamic|guided|auto (default: static)
 *        -C, --chunk <int>            : schedule chunk size
 *          (default: computed = ceil(work / (threads*4)), work=N (same) or N+K-1 (full))
 *   • STRIDE control:
 *        -st, --stride <int>          : output sampled every <stride> steps (default: 1)
 *          - SAME:   outLen = ceil(N / stride)
 *          - FULL:   outLen = ceil((N + K - 1) / stride)
 *        A stride > 1 reduces output size and compute cost proportionally.
 *
 * Performance quality-of-life:
 *   • Enhanced FULL kernel parallelizes over output n (no atomics).
 *   • Enhanced SAME kernel trims bounds (branch-free inner loop) and uses SIMD reduction.
 *   • First-touch init of output for better NUMA placement.
 *   • GFLOPS report (approx.): 2*outLen*K / time  (SAME still ignores trimmed edge effects).
 *
 * Build:
 *   cc -std=c11 -O3 -march=native -Wall -Wextra -Werror -fopenmp -o conv1d_omp conv1d_omp_stride.c
 *
 * Examples:
 *   ./conv1d_omp -L 1024 -kL 5 -o y.txt -se 42 --threads 8 --schedule guided --chunk 256
 *   ./conv1d_omp -f tests/f.txt -g tests/g.txt -m same -p zero -st 2 -t 4 -o tests/y_out.txt
 *
 * Notes:
 *   • SAME & FULL modes parallelize over the (possibly strided) output index n_out (embarrassingly parallel).
 *   • Defaults are set programmatically to avoid environment surprises:
 *       omp_set_dynamic(0), omp_set_nested(0),
 *       omp_set_num_threads(threads), omp_set_schedule(kind, chunk).
 *   • Compiles without OpenMP by providing stubs (falls back to single-thread).
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

#ifdef _OPENMP
#include <omp.h>
#else
/* --- OpenMP stubs so code compiles and runs single-threaded without OpenMP --- */
typedef int omp_sched_t;
#define omp_sched_static 1
#define omp_sched_dynamic 2
#define omp_sched_guided 3
#define omp_sched_auto 4
static inline void omp_set_dynamic(int x) { (void)x; }
static inline void omp_set_nested(int x) { (void)x; }
static inline void omp_set_num_threads(int x) { (void)x; }
static inline void omp_set_schedule(omp_sched_t s, int c)
{
    (void)s;
    (void)c;
}
static inline void omp_get_schedule(omp_sched_t *s, int *c)
{
    if (s)
        *s = omp_sched_static;
    if (c)
        *c = 1;
}
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
#endif

/* --- Modes and padding policies --- */
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

/* ---- Prototypes ---- */

void usage(const char *prog);
int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *threads, int *have_threads,
               omp_sched_t *sched_kind, int *have_sched,
               int *chunk, int *have_chunk,
               int *stride, int *have_stride);

float *read_array_1d(const char *path, int *len_out);
void write_array_1d(const char *path, const float *arr, int len);
float *gen_array_1d(int n);

void conv1d_full_stride_omp(const float *__restrict f, int N,
                            const float *__restrict g, int K,
                            int stride, float *__restrict out);

void conv1d_same_stride_omp(const float *__restrict f, int N,
                            const float *__restrict g, int K,
                            int stride, float *__restrict out,
                            pad_mode pmod, float cval);

/* Back-compat wrappers (stride=1) */
void conv1d_full_omp(const float *__restrict f, int N,
                     const float *__restrict g, int K,
                     float *__restrict out);

void conv1d_same_omp(const float *__restrict f, int N,
                     const float *__restrict g, int K,
                     float *__restrict out,
                     pad_mode pmod, float cval);

double elapsed_seconds(struct timespec a, struct timespec b);

void ensure_dir(const char *path);

void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs, double gflops,
                 int threads, const char *sched_str, int chunk,
                 int stride);

int parse_schedule_kind(const char *s, omp_sched_t *kind_out, const char **norm_out);
const char *schedule_to_string(omp_sched_t k);

/* ---------------- Implementation ---------------- */

/**
 * @brief Integer ceil-division helper.
 */
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

/**
 * @brief Print usage message to stderr.
 * @param prog executable name (argv[0])
 */
void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-f f.txt | -L N | --len N] "
            "[-g g.txt | -kL K | --klen K] "
            "-o out.txt|--out out.txt "
            "[-se seed|--seed seed] " /* NOTE: -se short, not -s */
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value] "
            "[-t threads|--threads threads] "
            "[-S sched|--schedule sched] "
            "[-C chunk|--chunk chunk] "
            "[-st stride|--stride stride]\n",
            prog);
}

/**
 * @brief Parse textual schedule kind to omp_sched_t and normalized string.
 * @param s input string (static|dynamic|guided|auto)
 * @param kind_out [out] parsed omp_sched_t
 * @param norm_out [out] normalized string literal
 * @return 1 on success; 0 on failure
 */
int parse_schedule_kind(const char *s, omp_sched_t *kind_out, const char **norm_out)
{
    if (!s)
        return 0;
    if (!strcmp(s, "static"))
    {
        if (kind_out)
            *kind_out = omp_sched_static;
        if (norm_out)
            *norm_out = "static";
        return 1;
    }
    if (!strcmp(s, "dynamic"))
    {
        if (kind_out)
            *kind_out = omp_sched_dynamic;
        if (norm_out)
            *norm_out = "dynamic";
        return 1;
    }
    if (!strcmp(s, "guided"))
    {
        if (kind_out)
            *kind_out = omp_sched_guided;
        if (norm_out)
            *norm_out = "guided";
        return 1;
    }
    if (!strcmp(s, "auto"))
    {
        if (kind_out)
            *kind_out = omp_sched_auto;
        if (norm_out)
            *norm_out = "auto";
        return 1;
    }
    return 0;
}

/**
 * @brief Convert omp_sched_t to normalized string.
 * @param k schedule kind
 * @return "static"|"dynamic"|"guided"|"auto" or "unknown"
 */
const char *schedule_to_string(omp_sched_t k)
{
    switch (k)
    {
    case omp_sched_static:
        return "static";
    case omp_sched_dynamic:
        return "dynamic";
    case omp_sched_guided:
        return "guided";
    case omp_sched_auto:
        return "auto";
    default:
        return "unknown";
    }
}

/**
 * @brief Parse CLI arguments and populate configuration/state.
 *
 * Recognizes file/RNG inputs, mode/padding/cval, OpenMP settings (threads, schedule, chunk),
 * stride and output path. Supports --len/-L and --klen/-kL for generated inputs.
 * NOTE: Short flags for seed/stride use two-letter forms: -se and -st (pre-filtered).
 *
 * @return 1 on success; 0 on usage error.
 */
int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *threads, int *have_threads,
               omp_sched_t *sched_kind, int *have_sched,
               int *chunk, int *have_chunk,
               int *stride, int *have_stride)
{
    *f_path = *g_path = *o_path = NULL;
    *N_req = *K_req = -1;
    *seed = (unsigned long)time(NULL);
    *have_seed = 0;
    *cmode = MODE_SAME;
    *pmode = PAD_ZERO;
    *cval = 0.0f;
    *have_cval = 0;
    *threads = 1;
    *have_threads = 0;
    *sched_kind = omp_sched_static;
    *have_sched = 0;
    *chunk = -1;
    *have_chunk = 0;
    *stride = 1;
    *have_stride = 0;

    /* Pre-filter to accept two-letter short options -se and -st (and -L/-kL order-agnostic) */
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
        { /* input length */
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-L requires an argument\n");
                free(fargv);
                return 0;
            }
            *N_req = strtol(argv[++i], NULL, 10);
            continue;
        }
        if (!strcmp(a, "-kL"))
        { /* kernel length */
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-kL requires an argument\n");
                free(fargv);
                return 0;
            }
            *K_req = strtol(argv[++i], NULL, 10);
            continue;
        }
        if (!strcmp(a, "-se"))
        { /* seed short */
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
        { /* stride short */
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-st requires an argument\n");
                free(fargv);
                return 0;
            }
            *stride = (int)strtol(argv[++i], NULL, 10);
            if (*stride < 1)
                *stride = 1;
            *have_stride = 1;
            continue;
        }
        if (!strncmp(a, "-st=", 4))
        {
            *stride = (int)strtol(a + 4, NULL, 10);
            if (*stride < 1)
                *stride = 1;
            *have_stride = 1;
            continue;
        }
        fargv[fargc++] = argv[i];
    }

    static struct option long_opts[] = {
        {"file", required_argument, 0, 'f'},
        {"kernel", required_argument, 0, 'g'},
        {"out", required_argument, 0, 'o'},
        {"seed", required_argument, 0, 6}, /* --seed uses val 6 */
        {"len", required_argument, 0, 1},
        {"klen", required_argument, 0, 2},
        {"mode", required_argument, 0, 3},
        {"padding", required_argument, 0, 4},
        {"cval", required_argument, 0, 5},
        {"threads", required_argument, 0, 't'},
        {"schedule", required_argument, 0, 'S'},
        {"chunk", required_argument, 0, 'C'},
        {"stride", required_argument, 0, 7}, /* --stride uses val 7 */
        {0, 0, 0, 0}};

    int opt, idx = 0;
    opterr = 0;
    while ((opt = getopt_long(fargc, fargv, "f:g:o:m:p:c:t:S:C:", long_opts, &idx)) != -1)
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

        case 6: /* --seed */
            *seed = strtoul(optarg, NULL, 10);
            *have_seed = 1;
            break;

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
            *threads = (int)strtol(optarg, NULL, 10);
            if (*threads < 1)
                *threads = 1;
            *have_threads = 1;
            break;

        case 'S':
        {
            omp_sched_t tmp;
            const char *norm = NULL;
            if (!parse_schedule_kind(optarg, &tmp, &norm))
            {
                usage(fargv[0]);
                free(fargv);
                return 0;
            }
            *sched_kind = tmp;
            *have_sched = 1;
            break;
        }

        case 'C':
            *chunk = (int)strtol(optarg, NULL, 10);
            if (*chunk < 1)
                *chunk = 1;
            *have_chunk = 1;
            break;

        case 1:
            *N_req = strtol(optarg, NULL, 10);
            break; /* --len */
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break; /* --klen */

        case 7: /* --stride */
            *stride = (int)strtol(optarg, NULL, 10);
            if (*stride < 1)
                *stride = 1;
            *have_stride = 1;
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
    return 1;
}

/**
 * @brief Read 1-D array from assignment-style file (length on first line, then values).
 * @param path input file path
 * @param len_out [out] length read
 * @return newly malloc'd float array of length *len_out (caller frees)
 */
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
        fprintf(stderr, "out of memory allocating %d floats for %s\n", L, path);
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

/**
 * @brief Write 1-D array to assignment-style file (length on first line, then values).
 * @param path output file path
 * @param arr pointer to array
 * @param len number of elements
 */
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

/**
 * @brief Generate length-n array with values ~ U([-1,1]).
 * @param n length
 * @return newly malloc'd float array of length n (caller frees)
 */
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
        fprintf(stderr, "out of memory generating array (n=%d)\n", n);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++)
    {
        float u01 = (float)rand() / (float)RAND_MAX;
        a[i] = -1.0f + 2.0f * u01;
    }
    return a;
}

/* -------------------- Convolution Kernels (STRIDE + OpenMP) -------------------- */

/**
 * @brief 1-D convolution in FULL mode with OpenMP, with stride (no atomics).
 *
 * Parallelizes over strided output element n_out in [0..ceil((N+K-1)/stride)-1].
 * Each n_out maps to a full-convolution index n = n_out * stride.
 * For each n, valid i overlap is i in [max(0,n-(K-1)) .. min(n,N-1)], j = n - i.
 *
 * @param f input signal of length N
 * @param N length of input signal
 * @param g kernel of length K
 * @param K length of kernel
 * @param stride stride between consecutive output indices (>=1)
 * @param out output buffer of length ceil((N+K-1)/stride) (pre-allocated)
 */
void conv1d_full_stride_omp(const float *__restrict f, int N,
                            const float *__restrict g, int K,
                            int stride, float *__restrict out)
{
    const int outLen = ceil_div_int(N + K - 1, stride);
#pragma omp parallel for schedule(runtime)
    for (int n_out = 0; n_out < outLen; n_out++)
    {
        const int n = n_out * stride; /* logical position in full conv */
        /* valid i overlap: i in [max(0,n-(K-1)) .. min(n,N-1)] */
        int i0 = n - (K - 1);
        if (i0 < 0)
            i0 = 0;
        int i1 = (n < N - 1) ? n : (N - 1);
        double acc = 0.0;
/* Inner loop vectorization hint; reduction keeps it scalar-accumulating safely. */
#pragma omp simd reduction(+ : acc)
        for (int i = i0; i <= i1; i++)
        {
            const int j = n - i; /* 0..K-1 */
            acc += (double)f[i] * (double)g[j];
        }
        out[n_out] = (float)acc;
    }
}

/**
 * @brief 1-D convolution in SAME mode with OpenMP, with stride (trimmed bounds + SIMD).
 *
 * Parallelizes over strided output element n_out in [0..ceil(N/stride)-1].
 * Each n_out maps to a SAME output index n = n_out * stride (0..N-1). The kernel is
 * centered via c = K/2. We compute tight bounds [m0..m1] so that idx = n - (m - c)
 * stays in [0..N-1], eliminating branches in the inner loop.
 *
 * Padding:
 *   - PAD_ZERO:   ignore out-of-bounds (no-op).
 *   - PAD_NONE:   ignore out-of-bounds (no-op).
 *   - PAD_CONST:  add cval * sum of g outside [m0..m1].
 *
 * @param f input signal of length N
 * @param N length of input signal
 * @param g kernel of length K
 * @param K length of kernel
 * @param stride stride between consecutive output indices (>=1)
 * @param out output buffer of length ceil(N/stride) (pre-allocated)
 * @param pmod padding mode for SAME (zero/none/const)
 * @param cval constant value for PAD_CONST
 */
void conv1d_same_stride_omp(const float *__restrict f, int N,
                            const float *__restrict g, int K,
                            int stride, float *__restrict out,
                            pad_mode pmod, float cval)
{
    const int c = K / 2;
    const int outLen = ceil_div_int(N, stride);
#pragma omp parallel for schedule(runtime)
    for (int n_out = 0; n_out < outLen; n_out++)
    {
        const int n = n_out * stride; /* 0..N-1 (strided) */
        /* m in [m0, m1] ⇒ idx = n - (m - c) stays in [0..N-1] */
        int m0 = 0, m1 = K - 1;

        /* Derive tight bounds for in-bounds idx, where idx = n - (m - c) */
        /* idx>=0 => n - (m - c) >= 0 => m <= n + c  ⇒ m1 = min(K-1, n + c) */
        int m1_tight = n + c;
        if (m1_tight < m1)
            m1 = m1_tight;
        if (m1 > K - 1)
            m1 = K - 1;
        /* idx<N  => n - (m - c) < N => m >= n - (N-1) + c  ⇒ m0 = max(0, n - (N-1) + c) */
        int m0_tight = n - (N - 1) + c;
        if (m0_tight > m0)
            m0 = m0_tight;
        if (m0 < 0)
            m0 = 0;

        double acc = 0.0;
#pragma omp simd reduction(+ : acc)
        for (int m = m0; m <= m1; m++)
        {
            const int idx = n - (m - c); /* in-bounds by construction */
            acc += (double)f[idx] * (double)g[m];
        }

        if (pmod == PAD_CONST)
        {
            double pad_sum = 0.0;
            for (int m = 0; m < m0; m++)
                pad_sum += (double)g[m];
            for (int m = m1 + 1; m < K; m++)
                pad_sum += (double)g[m];
            acc += (double)cval * pad_sum;
        }
        /* PAD_ZERO & PAD_NONE add zero work here */

        out[n_out] = (float)acc;
    }
}

/* -------- Back-compat wrappers (no stride param → stride=1) -------- */

void conv1d_full_omp(const float *__restrict f, int N,
                     const float *__restrict g, int K,
                     float *__restrict out)
{
    conv1d_full_stride_omp(f, N, g, K, /*stride=*/1, out);
}

void conv1d_same_omp(const float *__restrict f, int N,
                     const float *__restrict g, int K,
                     float *__restrict out,
                     pad_mode pmod, float cval)
{
    conv1d_same_stride_omp(f, N, g, K, /*stride=*/1, out, pmod, cval);
}

/**
 * @brief Compute (b - a) wall-clock seconds from two timespecs.
 * @param a start timestamp
 * @param b end timestamp
 * @return seconds as double
 */
double elapsed_seconds(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

/**
 * @brief Ensure a directory exists (mkdir if missing).
 * @param path directory path
 */
void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        (void)mkdir(path, 0775);
}

/**
 * @brief Log run metrics (CSV) into metrics/o0/ with a unique run ID.
 *
 * Fields: RunID,N,K,outLen,mode,padding,cval,time_s,gflops,threads,schedule,chunk,stride
 *
 * @param N length of f
 * @param K length of g
 * @param outLen length of output array
 * @param cmode convolution mode (full/same)
 * @param pmode padding mode (zero/none/const)
 * @param cval  pad constant (if PAD_CONST)
 * @param elapsed_secs kernel time in seconds
 * @param gflops throughput (approx) in GFLOP/s
 * @param threads OpenMP threads
 * @param sched_str schedule string
 * @param chunk OpenMP chunk size
 * @param stride stride used for this run
 */
void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs, double gflops,
                 int threads, const char *sched_str, int chunk,
                 int stride)
{
    ensure_dir("metrics/o0");

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
    snprintf(fname, sizeof(fname), "metrics/o0/metrics_%s.csv", runid);
    FILE *csv = fopen(fname, "w");
    if (!csv)
    {
        perror(fname);
        return;
    }

    fprintf(csv, "RunID,N,K,outLen,mode,padding,cval,time_s,gflops,threads,schedule,chunk,stride\n");
    fprintf(csv, "%s,%d,%d,%d,%s,%s,%.9g,%.9f,%.6f,%d,%s,%d,%d\n",
            runid, N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            elapsed_secs, gflops, threads, sched_str, chunk, stride);

    fclose(csv);
}

/* ----------------------------- Main ----------------------------- */

/**
 * @brief Program entry: orchestrates I/O, OpenMP config, timing, kernel call, and logging.
 * @return EXIT_SUCCESS on success; EXIT_FAILURE otherwise
 */
int main(int argc, char **argv)
{
    const char *f_path = NULL, *g_path = NULL, *o_path = NULL;
    long N_req = -1, K_req = -1;
    unsigned long seed = (unsigned long)time(NULL);
    int have_seed = 0;
    conv_mode cmode = MODE_SAME;
    pad_mode pmode = PAD_ZERO;
    float cval = 0.0f;
    int have_cval = 0;

    int threads = 1, have_threads = 0;
    omp_sched_t sched_kind = omp_sched_static;
    int have_sched = 0;
    int chunk = -1, have_chunk = 0;

    int stride = 1, have_stride = 0;

    if (!parse_args(argc, argv, &f_path, &g_path, &o_path,
                    &N_req, &K_req,
                    &seed, &have_seed,
                    &cmode, &pmode, &cval, &have_cval,
                    &threads, &have_threads,
                    &sched_kind, &have_sched,
                    &chunk, &have_chunk,
                    &stride, &have_stride))
    {
        return EXIT_FAILURE;
    }

    if (have_seed)
        srand((unsigned)seed);
    else
        srand((unsigned)seed);

    /* Prepare f and g: file or RNG */
    int N = 0, K = 0;
    float *f = f_path ? read_array_1d(f_path, &N) : (N = (int)N_req, gen_array_1d(N));
    float *g = g_path ? read_array_1d(g_path, &K) : (K = (int)K_req, gen_array_1d(K));

    /* Compute output length with stride */
    const int outLen = (cmode == MODE_FULL)
                           ? ceil_div_int(N + K - 1, stride)
                           : ceil_div_int(N, stride);

    float *out = (float *)malloc((size_t)outLen * sizeof(float));
    if (!out)
    {
        fprintf(stderr, "out of memory allocating output (outLen=%d)\n", outLen);
        free(f);
        free(g);
        return EXIT_FAILURE;
    }

    /* ---- OpenMP defaults (deterministic, user-overridable via CLI) ---- */
#ifdef _OPENMP
    if (!have_threads)
        threads = 1;    /* default: 1 thread unless specified */
    omp_set_dynamic(0); /* no runtime resizing */
    omp_set_nested(0);  /* no nested parallel regions */
    omp_set_num_threads(threads);

    /* Smarter chunk default if user didn't provide one:
       work = outLen; aim for ~4 chunks per thread. */
    if (!have_chunk)
    {
        long work = (long)outLen;
        long tgt_chunks = (long)threads * 4;
        long ch = (work + tgt_chunks - 1) / tgt_chunks; /* ceil */
        if (ch < 1)
            ch = 1;
        chunk = (int)ch;
    }
    if (!have_sched)
        sched_kind = omp_sched_static;
    omp_set_schedule(sched_kind, chunk);

    /* Query normalized schedule string for logs */
    omp_sched_t qk;
    int qc;
    omp_get_schedule(&qk, &qc);
    const char *sched_str = schedule_to_string(qk);
#else
    /* No OpenMP: force single-threaded semantics */
    threads = 1;
    if (chunk < 1)
        chunk = 1;
    const char *sched_str = "none";
    if (have_threads || have_sched || have_chunk)
    {
        fprintf(stderr, "note: compiled without OpenMP; ignoring --threads/--schedule/--chunk\n");
    }
#endif

    /* First-touch of out[] for NUMA friendliness (outside timing). */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < outLen; i++)
        out[i] = 0.0f;

    /* ---- Time ONLY the convolution kernel ---- */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (cmode == MODE_FULL)
    {
        conv1d_full_stride_omp(f, N, g, K, stride, out);
    }
    else
    {
        conv1d_same_stride_omp(f, N, g, K, stride, out, pmode, cval);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    const double secs = elapsed_seconds(t0, t1);

    /* FLOP model (approx; SAME ignores trimmed edges) and GFLOPS
       With stride, the amount of computed MACs is ~ outLen * K (each -> 2 FLOPs). */
    double flops = 2.0 * (double)outLen * (double)K;
    double gflops = (secs > 0.0) ? (flops / secs / 1e9) : 0.0;

    /* Human-readable timing (stderr) */
    fprintf(stderr,
            "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g | stride=%d | conv_time=%.9f s | %.3f GFLOP/s | OMP{threads=%d sched=%s chunk=%d}\n",
            N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? cval : 0.0),
            stride, secs, gflops,
            threads, sched_str, chunk);

    /* Metrics CSV (safe per-run filename) */
    log_metrics(N, K, outLen, cmode, pmode, cval, secs, gflops, threads, sched_str, chunk, stride);

    /* Write output array (assignment format) */
    write_array_1d(o_path, out, outLen);

    free(f);
    free(g);
    free(out);
    return EXIT_SUCCESS;
}