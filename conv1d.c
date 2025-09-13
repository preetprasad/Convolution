/**
 * @file conv1d.c
 * @brief 1-D Convolution with multiple modes, padding strategies, reproducible RNG, assignment-compliant I/O, and per-run CSV metrics.
 *
 * This program implements TRUE 1-D convolution (y = f * g) with support for
 * multiple output length modes, padding policies, reproducible random input
 * generation, assignment-style I/O, accurate timing, and automatic metrics
 * logging to CSV. It is designed to match project specifications for stress
 * testing on single nodes (e.g., Kaya cluster) while remaining flexible for
 * debugging and reproducibility.
 *
 * ### Output length modes (-m/--mode):
 *   - same (default): output length = N (input length)
 *   - full          : output length = N + K - 1
 *
 * ### Padding policies (-p/--padding), applies only for SAME mode:
 *   - zero  (default): out-of-range values treated as 0.0f
 *   - none           : out-of-range contributions ignored
 *   - const          : out-of-range values replaced with a user constant (-c/--cval)
 *
 * ### Input options:
 *   - Read arrays from text files:
 *       -f <f.txt> | --file <f.txt>        : input signal f
 *       -g <g.txt> | --kernel <g.txt>      : kernel g
 *   - Generate random arrays with uniform floats in [-1, 1]:
 *       -L <N>   | --len <N>   : generate input f of length N
 *       -kL <K>  | --klen <K>  : generate kernel g of length K
 *   - RNG reproducibility:
 *       -s <seed> | --seed <seed> : deterministic seed (defaults to time if omitted)
 *
 * ### Output:
 *   - Assignment-compliant text file:
 *       Line 1: integer length L
 *       Line 2: L floats with 3 decimal places, space-separated
 *   - Written via:
 *       -o <out.txt> | --out <out.txt>
 *
 * ### CSV metrics logging:
 *   - Each run appends a single row of metrics (N, K, outLen, mode, pad, cval, time)
 *   - CSVs stored under a "metrics/" directory:
 *       - On SLURM: metrics/metrics_SLURM_<JOBID>.csv
 *       - Locally:  metrics/metrics_LOCAL_YYYYMMDD_HHMMSS_<PID>.csv
 *   - Safe design: each job/run writes to its own file (no race conditions)
 *   - CSVs can later be merged/analyzed with pandas/matplotlib
 *
 * ### Numerical policy:
 *   - Arrays stored as float32
 *   - Accumulation uses double to reduce rounding error, then cast back to float
 *
 * ### Timing policy:
 *   - Reports ONLY the convolution kernel execution time to stderr
 *   - File I/O and RNG excluded (per assignment requirements)
 *
 * ### Error handling:
 *   - Invalid CLI arguments, missing required inputs
 *   - Bad file formats (wrong header or insufficient data)
 *   - Memory allocation failures
 *   - Clear diagnostics with non-zero exit
 *
 * ### Platform:
 *   - Portable C11, tested with:
 *       - Apple/clang (local development)
 *       - GCC on Linux clusters (Kaya via SLURM)
 *
 * ### Example usage:
 *   # Default assignment mode: SAME length, ZERO padding, random input
 *   ./conv1d -L 1024 -kL 5 -o y.txt -s 42
 *
 *   # SAME mode with constant padding = 1.5
 *   ./conv1d -L 16 -kL 5 -m same -p const -c 1.5 -o y.txt
 *
 *   # FULL mode convolution, padding ignored
 *   ./conv1d -L 16 -kL 3 -m full -o y.txt
 *
 * @note Build with:
 *   cc -std=c11 -O2 -Wall -Wextra -Werror -o conv1d conv1d.c
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

// Forward declarations */
void usage(const char *prog);
int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval);
float *read_array_1d(const char *path, int *len_out);
void write_array_1d(const char *path, const float *arr, int len);
float *gen_array_1d(int n);
void conv1d_full(const float *f, int N, const float *g, int K, float *out);
void conv1d_same(const float *f, int N, const float *g, int K, float *out,
                 pad_mode pmod, float cval);
double elapsed_seconds(struct timespec a, struct timespec b);
void ensure_dir(const char *path);
void log_metrics(int N, int K, int outLen,
                conv_mode cmode, pad_mode pmode, float cval,
                double elapsed_secs);

/**
 * Prints usage information for the program.
 * @param prog  Program name, typically argv[0].
 */
void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-f f.txt | -L N | --len N] "
            "[-g g.txt | -kL K | --klen K] "
            "-o out.txt|--out out.txt "
            "[-s seed|--seed seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value]\n",
            prog);
}

/**
 * Parses command-line arguments using a hybrid strategy:
 *  1) Pre-scan for assignment-style -L / -kL.
 *  2) Use getopt_long for short and long options.
 *
 * On success fills out all OUT parameters and returns 1.
 * On failure prints a diagnostic (and usage when relevant) and returns 0.
 *
 * @param f_path,g_path,o_path  OUT file paths (nullable).
 * @param N_req,K_req           OUT lengths when generating (if no files).
 * @param seed,have_seed        OUT RNG seed and flag if explicitly provided.
 * @param cmode,pmode           OUT convolution mode and padding policy.
 * @param cval,have_cval        OUT constant padding value and presence flag.
 * @return                      1 on success, 0 on invalid/missing args.
 */
int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval)
{
    *f_path = *g_path = *o_path = NULL;
    *N_req = *K_req = -1;
    *seed = (unsigned long)time(NULL);
    *have_seed = 0;
    *cmode = MODE_SAME; // assignment default
    *pmode = PAD_ZERO;  // assignment default
    *cval = 0.0f;
    *have_cval = 0;

    // Pre-scan for -L / -kL regardless of order
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
        {0, 0, 0, 0}};

    int opt, idx = 0;
    opterr = 0;
    while ((opt = getopt_long(argc, argv, "f:g:o:s:m:p:c:", long_opts, &idx)) != -1)
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
        case 1:
            *N_req = strtol(optarg, NULL, 10);
            break; // --len */
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break; // --klen */
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

/**
 * Reads a 1-D array from a text file in the assignment format.
 * On failure prints a diagnostic and terminates the program.
 *
 * @param path     Path to input file.
 * @param len_out  OUT parsed length L (>0).
 * @return         Pointer to malloc'd float array of length L.
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
 * Writes a 1-D array to a text file in assignment format.
 * On failure prints a diagnostic and terminates the program.
 *
 * @param path  Output path.
 * @param arr   Array pointer.
 * @param len   Number of elements.
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
 * Generates a length-n array of floats uniformly in [-1, 1].
 * Terminates on invalid length or allocation failure.
 *
 * @param n  Length (>0).
 * @return   Pointer to malloc'd float array.
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
        float u01 = (float)rand() / (float)RAND_MAX; // [0,1]
        a[i] = -1.0f + 2.0f * u01;                   // [-1,1]
    }
    return a;
}

/**
 * Computes FULL 1-D convolution.
 * Output length is N + K − 1. Padding policy is irrelevant for FULL.
 * Uses a double accumulator then stores back to float.
 *
 * @param f,g  Input arrays (lengths N and K).
 * @param N,K  Input lengths.
 * @param out  Output array of length N+K−1 (must be allocated by caller).
 */
void conv1d_full(const float *f, int N, const float *g, int K, float *out)
{
    const int outLen = N + K - 1;
    memset(out, 0, (size_t)outLen * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        const double fi = (double)f[i];
        for (int j = 0; j < K; j++)
        {
            // out[i+j] form already encodes true convolution (no explicit flip needed).
            const int n = i + j; // 0..N+K-2
            const double acc = (double)out[n] + fi * (double)g[j];
            out[n] = (float)acc;
        }
    }
}

/**
 * Computes SAME-length 1-D convolution with padding control.
 * y[n] = sum_{m=0..K-1} F(n + (m − c)) * g[m], where c = floor(K/2)
 * and F(.) is defined by padding policy.
 *
 * @param f,g    Input arrays (lengths N and K).
 * @param N,K    Input lengths.
 * @param out    Output array of length N (must be allocated by caller).
 * @param pmod   Padding policy (zero|none|const).
 * @param cval   Constant pad value when pmod == PAD_CONST.
 */
void conv1d_same(const float *f, int N, const float *g, int K, float *out,
                 pad_mode pmod, float cval)
{
    const int c = K / 2; // kernel anchor for SAME
    memset(out, 0, (size_t)N * sizeof(float));

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
                // PAD_ZERO: add 0 (no-op). PAD_NONE: skip contribution.
            }
        }
        out[n] = (float)acc;
    }
}

/**
 * Returns elapsed seconds between two CLOCK_MONOTONIC timestamps.
 *
 * @param a  Start timestamp.
 * @param b  End timestamp.
 * @return   Elapsed wall-time in seconds (fractional).
 */
double elapsed_seconds(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

/**
 * Ensures a directory exists; creates it if missing (best-effort).
 *
 * @param path  Directory path (e.g., "metrics").
 */
void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
    {
        (void)mkdir(path, 0775); // ignore EEXIST race
    }
}

/**
 * Writes a single CSV file for this run into "metrics/" with a unique name.
 * • On SLURM:  metrics/metrics_SLURM_<JOBID>.csv
 * • Locally:   metrics/metrics_LOCAL_YYYYMMDD_HHMMSS_<PID>.csv
 * The file contains a header and one data row for this execution.
 *
 * @param N,K,outLen     Problem sizes.
 * @param cmode,pmode    Mode and padding used.
 * @param cval           Constant padding value (if PAD_CONST; printed for completeness).
 * @param elapsed_secs   Convolution time in seconds.
 */
void log_metrics(int N, int K, int outLen,
                           conv_mode cmode, pad_mode pmode, float cval,
                           double elapsed_secs)
{
    ensure_dir("metrics/o0");

    // Build run id from SLURM_JOB_ID or timestamp+PID for local runs
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

    fprintf(csv, "RunID,N,K,outLen,mode,padding,cval,time\n");
    fprintf(csv, "%s,%d,%d,%d,%s,%s,%.9g,%.9f\n",
            runid, N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            elapsed_secs);

    fclose(csv);
}

/**
 * Program entry point:
 *  1) Parse CLI
 *  2) Read or generate inputs
 *  3) Time ONLY the convolution (exclude I/O and RNG)
 *  4) Persist per-run metrics CSV
 *  5) Write output array (assignment format)
 *
 * @return EXIT_SUCCESS on success; EXIT_FAILURE on invalid arguments or OOM.
 */
int main(int argc, char **argv)
{
    const char *f_path = NULL;
    const char *g_path = NULL;
    const char *o_path = NULL;
    long N_req = -1, K_req = -1;
    unsigned long seed = (unsigned long)time(NULL);
    int have_seed = 0;
    conv_mode cmode = MODE_SAME;
    pad_mode pmode = PAD_ZERO;
    float cval = 0.0f;
    int have_cval = 0;

    if (!parse_args(argc, argv, &f_path, &g_path, &o_path,
                    &N_req, &K_req, &seed, &have_seed,
                    &cmode, &pmode, &cval, &have_cval))
    {
        return EXIT_FAILURE;
    }

    srand((unsigned)seed); // deterministic if -s/--seed used

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

    // Time ONLY the convolution
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (cmode == MODE_FULL)
    {
        conv1d_full(f, N, g, K, out);
    }
    else
    {
        conv1d_same(f, N, g, K, out, pmode, cval);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    const double secs = elapsed_seconds(t0, t1);

    // Human-readable timing to stderr
    fprintf(stderr, "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g | conv_time=%.9f s\n",
            N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? cval : 0.0),
            secs);

    // Persist per-run metrics (safe for arrays / local)
    log_metrics(N, K, outLen, cmode, pmode, cval, secs);

    // Write output array
    write_array_1d(o_path, out, outLen);

    free(f);
    free(g);
    free(out);
    return EXIT_SUCCESS;
}