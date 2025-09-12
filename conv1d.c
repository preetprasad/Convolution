//==============================================================================
//  conv1d.c — 1-D Convolution with Mode (-m) and Padding (-p), incl. CONST pad
//==============================================================================
//
//  Description:
//  ------------
//  Computes 1-D TRUE convolution y = f * g with two output length modes:
//    • mode = same : output length N  (assignment default)
//    • mode = full : output length N + K − 1  (useful for debugging)
//  Padding policy controls how values outside input range are treated
//  (only meaningful for mode=same):
//    • padding = zero  : use 0.0f (assignment default)
//    • padding = none  : ignore out-of-range contributions
//    • padding = const : use a user-supplied constant (via -c/--cval)
//
//  Inputs can be read from files or generated randomly (uniform in [-1, 1]).
//  Only the convolution kernel time is printed to stderr (I/O and generation
//  are excluded), per project timing guidance.
//
//  Command-line interface (hybrid):
//    Short / assignment-style / long:
//      -f <f.txt>     | --file <f.txt>        : read f (1-D: N then N floats)
//      -g <g.txt>     | --kernel <g.txt>      : read g (1-D: K then K floats)
//      -o <out.txt>   | --out <out.txt>       : write output (1-D format)
//      -s <seed>      | --seed <seed>         : RNG seed (generation only)
//      -L <N>         | --len <N>             : generate f of length N (if no -f)
//      -kL <K>        | --klen <K>            : generate g of length K (if no -g)
//      -m <mode>      | --mode <mode>         : same|full        (default: same)
//      -p <padding>   | --padding <padding>   : zero|none|const  (default: zero)
//      -c <value>     | --cval <value>        : constant pad value for -p const
//
//  File format (1-D):
//    Line 1 : integer length L (>0)
//    Line 2 : L space-separated floats
//
//  Numerical policy:
//    • Arrays are single-precision floats (float32).
//    • Accumulation uses double for improved stability; stored to float.
//
//  Timing policy:
//    • Report ONLY the convolution time (exclude I/O and random generation).
//
//  Build:
//    cc -std=c11 -O2 -Wall -Wextra -Werror -o conv1d conv1d.c
//
//  Examples:
//    # Assignment-like defaults: SAME + ZERO padding
//    ./conv1d -L 1024 -kL 5 -o y.txt -s 42
//
//    # SAME length with constant padding value 1.5
//    ./conv1d -L 16 -kL 5 -m same -p const -c 1.5 -o y.txt
//
//    # FULL length (padding irrelevant)
//    ./conv1d -L 16 -kL 3 -m full -p none -o y.txt
//
//  Platform:   Apple / Linux (Kaya)
//==============================================================================

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>  // FILE, fopen, fclose, fprintf, fscanf, perror
#include <stdlib.h> // malloc, free, exit, rand, srand, strtol, strtoul, strtof
#include <string.h> // strcmp, memset
#include <time.h>   // time, clock_gettime
#include <errno.h>  // errno
#include <getopt.h> // getopt_long

//------------------------------------------------------------------------------
// Enumerations and forward declarations
//------------------------------------------------------------------------------
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
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval);
float *read_array_1d(const char *path, int *len_out);
void write_array_1d(const char *path, const float *arr, int len);
float *gen_array_1d(int n);
void conv1d_full(const float *f, int N, const float *g, int K, float *out);
void conv1d_same(const float *f, int N, const float *g, int K, float *out,
                 pad_mode pmod, float cval);
double elapsed_seconds(struct timespec a, struct timespec b);

/**
 * Prints the usage information for the program.
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
 *  - Pre-scan for assignment-style -L and -kL.
 *  - getopt_long for -f/-g/-o/-s and long forms (--file, --len, etc.).
 *  - Parses mode (-m/--mode), padding (-p/--padding), and const value (-c/--cval).
 *
 * @return 1 on success, 0 on invalid/missing required arguments.
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
    *cmode = MODE_SAME; // default per assignment
    *pmode = PAD_ZERO;  // default per assignment
    *cval = 0.0f;       // default constant (only used if PAD_CONST)
    *have_cval = 0;

    // ---- Pre-scan solely for -L and -kL so they are honored regardless of getopt_long ----
    for (int i = 1; i + 1 < argc; i++)
    {
        if (strcmp(argv[i], "-L") == 0)
            *N_req = strtol(argv[i + 1], NULL, 10);
        if (strcmp(argv[i], "-kL") == 0)
            *K_req = strtol(argv[i + 1], NULL, 10);
    }

    // Long option table for getopt_long
    static struct option long_opts[] = {
        {"file", required_argument, 0, 'f'},
        {"kernel", required_argument, 0, 'g'},
        {"out", required_argument, 0, 'o'},
        {"seed", required_argument, 0, 's'},
        {"len", required_argument, 0, 1},     // --len
        {"klen", required_argument, 0, 2},    // --klen
        {"mode", required_argument, 0, 3},    // --mode {same|full}
        {"padding", required_argument, 0, 4}, // --padding {zero|none|const}
        {"cval", required_argument, 0, 5},    // --cval <float>
        {0, 0, 0, 0}};

    int opt, long_index = 0;
    opterr = 0; // We will print our own usage on errors

    while ((opt = getopt_long(argc, argv, "f:g:o:s:m:p:c:", long_opts, &long_index)) != -1)
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

        case 'm': // short -m
        case 3:   // --mode
            if (strcmp(optarg, "same") == 0)
                *cmode = MODE_SAME;
            else if (strcmp(optarg, "full") == 0)
                *cmode = MODE_FULL;
            else
            {
                usage(argv[0]);
                return 0;
            }
            break;

        case 'p': // short -p
        case 4:   // --padding
            if (strcmp(optarg, "zero") == 0)
                *pmode = PAD_ZERO;
            else if (strcmp(optarg, "none") == 0)
                *pmode = PAD_NONE;
            else if (strcmp(optarg, "const") == 0)
                *pmode = PAD_CONST;
            else
            {
                usage(argv[0]);
                return 0;
            }
            break;

        case 'c': // short -c
        case 5:   // --cval
            *cval = strtof(optarg, NULL);
            *have_cval = 1;
            break;

        case 1:
            *N_req = strtol(optarg, NULL, 10);
            break; // --len
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break; // --klen
        default:   /* ignore unknowns; pre-scan handled -L/-kL */
            break;
        }
    }

    // Validate requireds
    if (!*o_path)
    {
        usage(argv[0]);
        return 0;
    }
    if (!*f_path && *N_req <= 0)
    {
        usage(argv[0]);
        fprintf(stderr, "Missing -L/--len for f length\n");
        return 0;
    }
    if (!*g_path && *K_req <= 0)
    {
        usage(argv[0]);
        fprintf(stderr, "Missing -kL/--klen for g length\n");
        return 0;
    }

    // Sanity: if -p const but no -c provided, keep default 0.0f but warn.
    if (*pmode == PAD_CONST && !*have_cval)
    {
        fprintf(stderr, "warning: -p const without -c/--cval; using cval=0.0\n");
    }
    // Note: padding has no effect for MODE_FULL; we allow it but could warn:
    if (*cmode == MODE_FULL)
    {
        // Optional warning: commented out to avoid noise.
        // fprintf(stderr, "note: padding ignored for mode=full\n");
    }

    return 1;
}

/**
 * Reads a 1-D array from a text file in assignment format:
 *   Line 1 : integer length L (>0)
 *   Line 2 : L space-separated floats
 *
 * Exits the program on any error.
 * @param path     Input file path.
 * @param len_out  OUT: parsed length.
 * @return         malloc'd float array of length *len_out.
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
 * Writes a 1-D array to a text file in assignment format:
 *   Line 1 : integer length
 *   Line 2 : values with 3 decimals (space-separated)
 *
 * Exits the program on any error.
 * @param path  Output file path.
 * @param arr   Pointer to array to write.
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
    {
        fprintf(fp, (i + 1 == len) ? "%.3f\n" : "%.3f ", arr[i]);
    }

    fclose(fp);
}

/**
 * Generates a length-n array with uniform random floats in [-1, 1].
 * Exits on allocation failure or invalid length.
 * @param n  Length (>0).
 * @return   malloc'd float array of length n.
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
        // Uniform in [0,1] scaled to [-1,1]
        float u01 = (float)rand() / (float)RAND_MAX;
        a[i] = -1.0f + 2.0f * u01;
    }
    return a;
}

/**
 * FULL 1-D convolution (true convolution). Output length is N + K − 1.
 * Implementation:
 *   out[i + j] += f[i] * g[j], i in [0..N-1], j in [0..K-1]
 * Notes:
 *   - Padding mode has no effect for FULL (indices always valid in out).
 *   - Double accumulator reduces rounding error; stored as float.
 *
 * @param f    Input array (len N).
 * @param N    Length of f.
 * @param g    Kernel array (len K).
 * @param K    Length of g.
 * @param out  Output array (len N+K-1). Must be allocated.
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
            const int n = i + j; // 0..N+K-2
            const double acc = (double)out[n] + fi * (double)g[j];
            out[n] = (float)acc;
        }
    }
}

/**
 * SAME 1-D convolution (true convolution). Output length is N.
 * Implementation:
 *   y[n] = sum_{m=0..K-1} F(n + (m − c)) * g[m],  where c = floor(K/2)
 *   with boundary handling controlled by pad_mode:
 *     - PAD_ZERO : F(out-of-range) = 0.0f
 *     - PAD_NONE : skip out-of-range contributions
 *     - PAD_CONST: F(out-of-range) = cval
 *
 * @param f     Input array (len N).
 * @param N     Length of f.
 * @param g     Kernel array (len K).
 * @param K     Length of g.
 * @param out   Output array (len N). Must be allocated.
 * @param pmod  Padding mode.
 * @param cval  Constant pad value when pmod == PAD_CONST.
 */
void conv1d_same(const float *f, int N, const float *g, int K, float *out,
                 pad_mode pmod, float cval)
{
    const int c = K / 2; // anchor
    memset(out, 0, (size_t)N * sizeof(float));

    for (int n = 0; n < N; n++)
    {
        double acc = 0.0;
        for (int m = 0; m < K; m++)
        {
            const int idx = n + (m - c); // index into f

            if (idx >= 0 && idx < N)
            {
                acc += (double)f[idx] * (double)g[m];
            }
            else
            {
                if (pmod == PAD_NONE)
                {
                    // skip contribution
                }
                else if (pmod == PAD_ZERO)
                {
                    // add 0.0f (no-op)
                }
                else if (pmod == PAD_CONST)
                {
                    acc += (double)cval * (double)g[m];
                }
            }
        }
        out[n] = (float)acc;
    }
}

/**
 * Computes elapsed seconds between two CLOCK_MONOTONIC timestamps.
 * @param a  Start timestamp.
 * @param b  End timestamp.
 * @return   Elapsed wall time in seconds (fractional).
 */
double elapsed_seconds(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

/**
 * Program entry point:
 *  1) Parse CLI
 *  2) Read or generate inputs
 *  3) Time ONLY the convolution
 *  4) Write output (assignment format)
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

    // Seed RNG (deterministic if user supplied -s/--seed)
    if (have_seed)
        srand((unsigned)seed);
    else
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

    // Allocate output based on mode
    int outLen = (cmode == MODE_FULL) ? (N + K - 1) : N;
    float *out = (float *)malloc((size_t)outLen * sizeof(float));
    if (!out)
    {
        fprintf(stderr, "out of memory allocating output (outLen=%d)\n", outLen);
        free(f);
        free(g);
        return EXIT_FAILURE;
    }

    // ---- Measure ONLY the convolution (exclude I/O and generation) ----
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
    fprintf(stderr, "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g | conv_time=%.9f s\n",
            N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? cval : 0.0),
            secs);

    // Write output
    write_array_1d(o_path, out, outLen);

    // Cleanup
    free(f);
    free(g);
    free(out);
    return EXIT_SUCCESS;
}