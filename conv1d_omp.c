/**
 * @file conv1d_omp.c
 * @brief OpenMP-enabled 1-D Convolution with modes, padding, RNG, assignment I/O, CSV metrics,
 *        and CLI controls for threads, schedule, and chunk.
 *
 * This program implements TRUE 1-D convolution (y = f * g) with:
 *   • Output length modes (-m/--mode): same (default) or full.
 *   • Padding policies (-p/--padding) for SAME mode: zero (default), none, const (-c/--cval).
 *   • Inputs from files (-f/-g) or generated uniformly in [-1,1] (-L/-kL, optional -s/--seed).
 *   • Accurate timing of ONLY the convolution kernel (I/O/RNG excluded).
 *   • Per-run CSV metrics under metrics/o0/, uniquely named for SLURM vs local runs.
 *   • OpenMP parallelization with deterministic defaults and CLI overrides:
 *        -t, --threads <int>          : number of threads (default: 1)
 *        -S, --schedule <kind>        : static|dynamic|guided|auto (default: static)
 *        -C, --chunk <int>            : schedule chunk size (default: computed = max(N/threads,1))
 *
 * CSV fields now include OpenMP parameters: threads, schedule, chunk.
 *
 * Build:
 *   cc -std=c11 -O2 -Wall -Wextra -Werror -fopenmp -o conv1d_omp conv1d_omp.c
 *
 * Example:
 *   ./conv1d_omp -L 1024 -kL 5 -o y.txt -s 42 --threads 8 --schedule guided --chunk 256
 *
 * Notes:
 *   • SAME mode parallelizes over output index n (embarrassingly parallel).
 *   • FULL mode parallelizes over i and uses an atomic double accumulation into a double buffer
 *     (then casts to float). This preserves your “double-accumulate” numerical policy.
 *   • Defaults are set programmatically to avoid environment surprises:
 *       omp_set_dynamic(0), omp_set_nested(0),
 *       omp_set_num_threads(threads), omp_set_schedule(kind, chunk).
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
// Stubs so code compiles without OpenMP (single-thread fallback).
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

void usage(const char *prog);
int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *threads, int *have_threads,
               omp_sched_t *sched_kind, int *have_sched,
               int *chunk, int *have_chunk);
float *read_array_1d(const char *path, int *len_out);
void write_array_1d(const char *path, const float *arr, int len);
float *gen_array_1d(int n);
void conv1d_full_omp(const float *f, int N, const float *g, int K, float *out);
void conv1d_same_omp(const float *f, int N, const float *g, int K, float *out,
                     pad_mode pmod, float cval);
double elapsed_seconds(struct timespec a, struct timespec b);
void ensure_dir(const char *path);
void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs,
                 int threads, const char *sched_str, int chunk);

/* Map schedule string to omp_sched_t and back */
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

void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-f f.txt | -L N | --len N] "
            "[-g g.txt | -kL K | --klen K] "
            "-o out.txt|--out out.txt "
            "[-s seed|--seed seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value] "
            "[-t threads|--threads threads] "
            "[-S sched|--schedule sched] "
            "[-C chunk|--chunk chunk]\n",
            prog);
}

int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *threads, int *have_threads,
               omp_sched_t *sched_kind, int *have_sched,
               int *chunk, int *have_chunk)
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

    // Pre-scan for -L / -kL
    for (int i = 1; i + 1 < argc; i++)
    {
        if (strcmp(argv[i], "-L") == 0)
            *N_req = strtol(argv[i + 1], NULL, 10);
        if (strcmp(argv[i], "-kL") == 0)
            *K_req = strtol(argv[i + 1], NULL, 10);
    }

    static struct option long_opts[] = {
        {"file", required_argument, 0, 'f'},
        {"kernel", required_argument, 0, 'g'},
        {"out", required_argument, 0, 'o'},
        {"seed", required_argument, 0, 's'},
        {"len", required_argument, 0, 1},
        {"klen", required_argument, 0, 2},
        {"mode", required_argument, 0, 3},
        {"padding", required_argument, 0, 4},
        {"cval", required_argument, 0, 5},
        {"threads", required_argument, 0, 't'},
        {"schedule", required_argument, 0, 'S'},
        {"chunk", required_argument, 0, 'C'},
        {0, 0, 0, 0}};

    int opt, idx = 0;
    opterr = 0;
    while ((opt = getopt_long(argc, argv, "f:g:o:s:m:p:c:t:S:C:", long_opts, &idx)) != -1)
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
            break;
        case 'm':
        case 3:
            if (!strcmp(optarg, "same"))
                *cmode = MODE_SAME;
            else if (!strcmp(optarg, "full"))
                *cmode = MODE_FULL;
            else
            {
                usage(argv[0]);
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
                usage(argv[0]);
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
                usage(argv[0]);
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
            break; // --len
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break; // --klen
        default:
            break;
        }
    }

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

/* FULL mode with OpenMP:
 * Parallelize over i; accumulate into a DOUBLE buffer with atomic update to preserve numerical policy.
 * Finally cast double buffer to float output.
 */
void conv1d_full_omp(const float *f, int N, const float *g, int K, float *out)
{
    const int outLen = N + K - 1;
    double *outD = (double *)calloc((size_t)outLen, sizeof(double));
    if (!outD)
    {
        fprintf(stderr, "out of memory allocating outD (len=%d)\n", outLen);
        exit(EXIT_FAILURE);
    }

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++)
    {
        const double fi = (double)f[i];
        for (int j = 0; j < K; j++)
        {
            const int n = i + j;
            const double val = fi * (double)g[j];
#pragma omp atomic update
            outD[n] += val;
        }
    }

    // Cast back to float
    for (int n = 0; n < outLen; n++)
        out[n] = (float)outD[n];
    free(outD);
}

/* SAME mode with OpenMP:
 * Parallelize over output index n (each acc is private and independent).
 */
void conv1d_same_omp(const float *f, int N, const float *g, int K, float *out,
                     pad_mode pmod, float cval)
{
    const int c = K / 2;
#pragma omp parallel for schedule(runtime)
    for (int n = 0; n < N; n++)
    {
        double acc = 0.0;
        for (int m = 0; m < K; m++)
        {
            const int idx = n + (m - c);
            if (idx >= 0 && idx < N)
            {
                acc += (double)f[idx] * (double)g[m];
            }
            else
            {
                if (pmod == PAD_CONST)
                {
                    acc += (double)cval * (double)g[m];
                }
                // PAD_ZERO: add 0; PAD_NONE: skip
            }
        }
        out[n] = (float)acc;
    }
}

double elapsed_seconds(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
    {
        (void)mkdir(path, 0775);
    }
}

void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs,
                 int threads, const char *sched_str, int chunk)
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

    fprintf(csv, "RunID,N,K,outLen,mode,padding,cval,time,threads,schedule,chunk\n");
    fprintf(csv, "%s,%d,%d,%d,%s,%s,%.9g,%.9f,%d,%s,%d\n",
            runid, N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            elapsed_secs,
            threads, sched_str, chunk);

    fclose(csv);
}

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

    if (!parse_args(argc, argv, &f_path, &g_path, &o_path,
                    &N_req, &K_req, &seed, &have_seed,
                    &cmode, &pmode, &cval, &have_cval,
                    &threads, &have_threads,
                    &sched_kind, &have_sched,
                    &chunk, &have_chunk))
    {
        return EXIT_FAILURE;
    }

    srand((unsigned)seed);

    // Prepare f
    int N = 0;
    float *f = NULL;
    if (f_path)
        f = read_array_1d(f_path, &N);
    else
    {
        N = (int)N_req;
        f = gen_array_1d(N);
    }

    // Prepare g
    int K = 0;
    float *g = NULL;
    if (g_path)
        g = read_array_1d(g_path, &K);
    else
    {
        K = (int)K_req;
        g = gen_array_1d(K);
    }

    // Allocate output
    const int outLen = (cmode == MODE_FULL) ? (N + K - 1) : N;
    float *out = (float *)malloc((size_t)outLen * sizeof(float));
    if (!out)
    {
        fprintf(stderr, "out of memory allocating output (outLen=%d)\n", outLen);
        free(f);
        free(g);
        return EXIT_FAILURE;
    }

    // ---- OpenMP defaults (deterministic, user-overridable via CLI) ----
#ifdef _OPENMP
    if (!have_threads)
        threads = 1;    // default to 1 thread unless specified
    omp_set_dynamic(0); // no runtime resizing
    omp_set_nested(0);  // no nested parallel regions
    omp_set_num_threads(threads);
    // Default chunk if not provided: balanced static chunks
    if (!have_chunk)
    {
        int base = (N > threads) ? (N / threads) : 1;
        chunk = (base < 1) ? 1 : base;
    }
    // Default schedule kind if not provided: static
    if (!have_sched)
    {
        sched_kind = omp_sched_static;
    }
    omp_set_schedule(sched_kind, chunk);
    // Query normalized schedule string for logs
    omp_sched_t qk;
    int qc;
    omp_get_schedule(&qk, &qc);
    const char *sched_str = schedule_to_string(qk);
#else
    // No OpenMP: force single-threaded semantics
    threads = 1;
    if (chunk < 1)
        chunk = 1;
    const char *sched_str = "none";
    if (have_threads || have_sched || have_chunk)
    {
        fprintf(stderr, "note: compiled without OpenMP; ignoring --threads/--schedule/--chunk\n");
    }
#endif

    // ---- Time ONLY the convolution ----
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (cmode == MODE_FULL)
    {
        conv1d_full_omp(f, N, g, K, out);
    }
    else
    {
        conv1d_same_omp(f, N, g, K, out, pmode, cval);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    const double secs = elapsed_seconds(t0, t1);

    // Human-readable timing (stderr)
    fprintf(stderr,
            "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g | conv_time=%.9f s | OMP{threads=%d sched=%s chunk=%d}\n",
            N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? cval : 0.0),
            secs,
            threads, sched_str, chunk);

    // Metrics CSV (safe per-run filename)
    log_metrics(N, K, outLen, cmode, pmode, cval, secs, threads, sched_str, chunk);

    // Write output array
    write_array_1d(o_path, out, outLen);

    free(f);
    free(g);
    free(out);
    return EXIT_SUCCESS;
}