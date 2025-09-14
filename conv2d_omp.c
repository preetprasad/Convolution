/**
 * @file conv2d_omp.c
 * @brief OpenMP-enabled 2-D Correlation/Convolution with SAME/FULL modes, padding policies,
 *        RNG for reproducibility, assignment-style I/O, GFLOPS metrics, and CLI controls
 *        for threads, schedule, and chunk (ported feature parity with the 1-D program).
 *
 * Core math:
 *   • corr2d_* implements mathematical 2-D cross-correlation (no flip).
 *   • Convolution is realized as correlation with a once-off kernel flip: Gf[u,v] = G[KH-1-u, KW-1-v].
 *
 * Operation selection:
 *   --conv (default)  : flip kernel then call corr2d_* (true convolution)
 *   --corr            : call corr2d_* directly (correlation)
 *
 * Output modes (-m/--mode):
 *   same (default) : output H×W
 *   full           : output (H+KH-1)×(W+KW-1)
 *
 * Padding policies (-p/--padding), SAME mode only:
 *   zero (default)
 *   none
 *   const  (requires -c/--cval, contributes cval*G outside)
 *
 * OpenMP controls (deterministic defaults, user-overridable):
 *   -t, --threads <int>     : number of threads (default: 1)
 *   -S, --schedule <kind>   : static|dynamic|guided|auto (default: static)
 *   -C, --chunk <int>       : schedule chunk size (default: ceil(work/(threads*4)))
 *     where work = outH*outW (SAME) or (H+KH-1)*(W+KW-1) (FULL)
 *
 * Input options:
 *   From files (assignment format):
 *     -f, --file <img.txt>    : image  (first line "H W", then H*W floats row-major)
 *     -g, --kernel <ker.txt>  : kernel (first line "KH KW", then KH*KW floats)
 *   Generated U([-1,1]) with reproducible RNG:
 *     -H/--rows <H> -W/--cols <W>
 *     -kH/--krows <KH> -kW/--kcols <KW>
 *     -s/--seed <seed>
 *
 * Output:
 *   -o, --out <out.txt>  (assignment format: header "outH outW", then values)
 *
 * Metrics (CSV; one row per run):
 *   Path: metrics/o0/metrics_{SLURM_$JOBID | LOCAL_YYYYMMDD_HHMMSS_PID}.csv
 *   Columns:
 *     RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,time_s,gflops,threads,schedule,chunk
 *
 * Timing policy:
 *   Only the corr2d_* kernel is timed. Preprocessing (flip), RNG, and I/O are excluded.
 *
 * Build:
 *   cc -std=c11 -O3 -march=native -Wall -Wextra -Werror -fopenmp -o conv2d_omp conv2d_omp.c
 *
 * Example:
 *   ./conv2d_omp -H 1024 -W 1024 -kH 5 -kW 5 -o Y.txt -s 42 \
 *       --threads 8 --schedule guided --chunk 4096
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
/* --- OpenMP stubs so code compiles & runs single-threaded without OpenMP --- */
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

/* ---------------------------- Types & Enums ---------------------------- */

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

/* ------------------------------ Prototypes ------------------------------ */
void usage(const char *prog);

int parse_args(int argc, char **argv,
               const char **img_path, const char **ker_path, const char **out_path,
               long *H_req, long *W_req, long *KH_req, long *KW_req,
               unsigned long *seed, int *have_seed,
               op_kind *op, conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *threads, int *have_threads, omp_sched_t *sched_kind, int *have_sched,
               int *chunk, int *have_chunk);

int parse_schedule_kind(const char *s, omp_sched_t *kind_out, const char **norm_out);
const char *schedule_to_string(omp_sched_t k);

float *read_matrix_2d(const char *path, int *H_out, int *W_out);
void write_matrix_2d(const char *path, const float *A, int H, int W);
float *gen_matrix_2d(int H, int W);

float *flip_kernel_2d(const float *G, int KH, int KW);

void corr2d_full(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y);

void corr2d_same(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y, pad_mode pmod, float cval);

double elapsed_seconds(struct timespec a, struct timespec b);
void ensure_dir(const char *path);

void log_metrics(int H, int W, int KH, int KW, int outH, int outW,
                 op_kind op, conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs, double gflops,
                 int threads, const char *sched_str, int chunk);

/* ------------------------------ Utilities ------------------------------ */
#define IDX2(i, j, ldW) ((size_t)(i) * (size_t)(ldW) + (size_t)(j))

/**
 * @brief Print usage message to stderr.
 * @param prog executable name (argv[0])
 */
void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--conv|--corr] "
            "[-f img.txt | -H H --rows H -W W --cols W] "
            "[-g ker.txt | -kH KH --krows KH -kW KW --kcols KW] "
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

/**
 * @brief Parse schedule kind string (static|dynamic|guided|auto).
 * @param s         Input text.
 * @param kind_out  [out] Resulting omp_sched_t.
 * @param norm_out  [out] Canonical string literal (for logs).
 * @return 1 on success, 0 on error.
 */
int parse_schedule_kind(const char *s, omp_sched_t *k, const char **norm)
{
    if (!s)
        return 0;
#ifdef _OPENMP
    if (!strcmp(s, "static"))
    {
        if (k)
            *k = omp_sched_static;
        if (norm)
            *norm = "static";
        return 1;
    }
    if (!strcmp(s, "dynamic"))
    {
        if (k)
            *k = omp_sched_dynamic;
        if (norm)
            *norm = "dynamic";
        return 1;
    }
    if (!strcmp(s, "guided"))
    {
        if (k)
            *k = omp_sched_guided;
        if (norm)
            *norm = "guided";
        return 1;
    }
    if (!strcmp(s, "auto"))
    {
        if (k)
            *k = omp_sched_auto;
        if (norm)
            *norm = "auto";
        return 1;
    }
#else
    if (!strcmp(s, "static"))
    {
        if (k)
            *k = 1;
        if (norm)
            *norm = "static";
        return 1;
    }
    if (!strcmp(s, "dynamic"))
    {
        if (k)
            *k = 2;
        if (norm)
            *norm = "dynamic";
        return 1;
    }
    if (!strcmp(s, "guided"))
    {
        if (k)
            *k = 3;
        if (norm)
            *norm = "guided";
        return 1;
    }
    if (!strcmp(s, "auto"))
    {
        if (k)
            *k = 4;
        if (norm)
            *norm = "auto";
        return 1;
    }
#endif
    return 0;
}

/**
 * @brief Convert omp_sched_t to normalized string.
 * @param k schedule kind
 * @return "static"|"dynamic"|"guided"|"auto" or "unknown"
 */
const char *schedule_to_string(omp_sched_t k)
{
#ifdef _OPENMP
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
#else
    (void)k;
    return "none";
#endif
}

/* ------------------------------ CLI parsing ------------------------------ */

/**
 * @brief Parse CLI (mirrors 1-D program style).
 *
 * Accepts: file/RNG inputs (-f/-g or -H/-W/-kH/-kW/--krows/--kcols), output path, seed,
 * mode/padding/cval, conv/corr, and OpenMP controls (threads/schedule/chunk).
 * Supports short -kH/-kW by pre-filtering them before getopt_long.
 *
 * @return 1 on success; 0 on validation failure.
 */
int parse_args(int argc, char **argv,
               const char **img_path, const char **ker_path, const char **out_path,
               long *H_req, long *W_req, long *KH_req, long *KW_req,
               unsigned long *seed, int *have_seed,
               op_kind *op, conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *threads, int *have_threads, omp_sched_t *sched_kind, int *have_sched,
               int *chunk, int *have_chunk)
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
    *threads = 1;
    *have_threads = 0;
#ifdef _OPENMP
    *sched_kind = omp_sched_static;
#else
    *sched_kind = 1;
#endif
    *have_sched = 0;
    *chunk = -1;
    *have_chunk = 0;

    /* Pre-filter to accept -kH/-kW short options (like 1-D style) */
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
        if (!strcmp(a, "-kH"))
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-kH requires an argument\n");
                free(fargv);
                return 0;
            }
            *KH_req = strtol(argv[++i], NULL, 10);
            continue;
        }
        if (!strncmp(a, "-kH=", 4))
        {
            *KH_req = strtol(a + 4, NULL, 10);
            continue;
        }
        if (!strcmp(a, "-kW"))
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-kW requires an argument\n");
                free(fargv);
                return 0;
            }
            *KW_req = strtol(argv[++i], NULL, 10);
            continue;
        }
        if (!strncmp(a, "-kW=", 4))
        {
            *KW_req = strtol(a + 4, NULL, 10);
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
        {"threads", required_argument, 0, 't'},
        {"schedule", required_argument, 0, 'S'},
        {"chunk", required_argument, 0, 'C'},
        {0, 0, 0, 0}};

    int opt, idx = 0;
    opterr = 0;
    while ((opt = getopt_long(fargc, fargv, "f:g:o:s:m:p:c:H:W:t:S:C:", long_opts, &idx)) != -1)
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
            *seed = strtoul(optarg, NULL, 10);
            *have_seed = 1;
            break;
        case 'm':
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
            *cval = strtof(optarg, NULL);
            *have_cval = 1;
            break;
        case 'H':
            *H_req = strtol(optarg, NULL, 10);
            break;
        case 'W':
            *W_req = strtol(optarg, NULL, 10);
            break;
        case 1:
            *KH_req = strtol(optarg, NULL, 10);
            break; /* --krows */
        case 2:
            *KW_req = strtol(optarg, NULL, 10);
            break; /* --kcols */
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
        default:
            break;
        }
    }
    free(fargv);

    if (!*out_path)
    {
        usage(argv[0]);
        return 0;
    }
    if (!*img_path && (*H_req <= 0 || *W_req <= 0))
    {
        fprintf(stderr, "Missing -H/--rows and/or -W/--cols for generated image.\n");
        return 0;
    }
    if (!*ker_path && (*KH_req <= 0 || *KW_req <= 0))
    {
        fprintf(stderr, "Missing -kH/--krows and/or -kW/--kcols for generated kernel.\n");
        return 0;
    }
    if (*pmode == PAD_CONST && !*have_cval)
        fprintf(stderr, "warning: -p const without -c/--cval; using cval=0.0\n");
    return 1;
}

/* ----------------------------- I/O helpers ----------------------------- */

/**
 * @brief Read H×W matrix from assignment-style file.
 * @param path   File path.
 * @param H_out  [out] Height (H).
 * @param W_out  [out] Width (W).
 * @return malloc'd float array of size H*W; caller must free.
 */
float *read_matrix_2d(const char *path, int *H_out, int *W_out)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        perror(path);
        exit(EXIT_FAILURE);
    }
    int H = 0, W = 0;
    if (fscanf(fp, "%d %d", &H, &W) != 2 || H <= 0 || W <= 0)
    {
        fprintf(stderr, "bad header in %s (expected positive 'H W')\n", path);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    float *A = (float *)malloc((size_t)H * (size_t)W * sizeof(float));
    if (!A)
    {
        fprintf(stderr, "oom reading %s (%dx%d)\n", path, H, W);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            if (fscanf(fp, "%f", &A[IDX2(i, j, W)]) != 1)
            {
                fprintf(stderr, "bad body in %s at (%d,%d)\n", path, i, j);
                free(A);
                fclose(fp);
                exit(EXIT_FAILURE);
            }
    fclose(fp);
    *H_out = H;
    *W_out = W;
    return A;
}

/**
 * @brief Write H×W matrix to assignment-style file.
 * @param path Output file path.
 * @param A    Matrix pointer.
 * @param H    Height.
 * @param W    Width.
 */
void write_matrix_2d(const char *path, const float *A, int H, int W)
{
    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        perror(path);
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "%d %d\n", H, W);
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
            fprintf(fp, (j + 1 == W) ? "%.3f\n" : "%.3f ", A[IDX2(i, j, W)]);
    }
    fclose(fp);
}

/**
 * @brief Generate H×W random matrix (U([-1,1])).
 * @return malloc'd float array of size H*W; caller must free.
 */
float *gen_matrix_2d(int H, int W)
{
    if (H <= 0 || W <= 0)
    {
        fprintf(stderr, "invalid dims H=%d W=%d\n", H, W);
        exit(EXIT_FAILURE);
    }
    float *A = (float *)malloc((size_t)H * (size_t)W * sizeof(float));
    if (!A)
    {
        fprintf(stderr, "oom generating matrix (%dx%d)\n", H, W);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
        {
            float u01 = (float)rand() / (float)RAND_MAX;
            A[IDX2(i, j, W)] = -1.0f + 2.0f * u01;
        }
    return A;
}

/* --------------------------- Kernel helpers --------------------------- */

/**
 * @brief Flip kernel along both axes (for convolution).
 * @param G   KH×KW kernel.
 * @param KH  Kernel height.
 * @param KW  Kernel width.
 * @return malloc'd flipped kernel; caller must free.
 */
float *flip_kernel_2d(const float *G, int KH, int KW)
{
    float *Gf = (float *)malloc((size_t)KH * (size_t)KW * sizeof(float));
    if (!Gf)
    {
        fprintf(stderr, "oom flipping kernel (%dx%d)\n", KH, KW);
        exit(EXIT_FAILURE);
    }
    for (int u = 0; u < KH; u++)
        for (int v = 0; v < KW; v++)
            Gf[IDX2(u, v, KW)] = G[IDX2(KH - 1 - u, KW - 1 - v, KW)];
    return Gf;
}

/* ------------------------- Correlation kernels ------------------------- */
/**
 * @brief 2-D FULL cross-correlation (no flip inside).
 * @param F   H×W image.
 * @param H,W Dimensions of F.
 * @param G   KH×KW kernel.
 * @param KH,KW Kernel dims.
 * @param Y   Output (H+KH−1)×(W+KW−1).
 */
void corr2d_full(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y)
{
    const int outH = H + KH - 1;
    const int outW = W + KW - 1;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(runtime)
#endif
    for (int oi = 0; oi < outH; oi++)
    {
        for (int oj = 0; oj < outW; oj++)
        {
            const int i0 = (oi - (KH - 1) > 0) ? (oi - (KH - 1)) : 0;
            const int i1 = (oi < (H - 1)) ? oi : (H - 1);
            const int j0 = (oj - (KW - 1) > 0) ? (oj - (KW - 1)) : 0;
            const int j1 = (oj < (W - 1)) ? oj : (W - 1);

            double acc = 0.0;
            for (int i = i0; i <= i1; i++)
            {
                const int u = oi - i; /* 0..KH-1 */
#ifdef _OPENMP
#pragma omp simd reduction(+ : acc)
#endif
                for (int j = j0; j <= j1; j++)
                {
                    const int v = oj - j; /* 0..KW-1 */
                    acc += (double)F[IDX2(i, j, W)] * (double)G[IDX2(u, v, KW)];
                }
            }
            Y[IDX2(oi, oj, outW)] = (float)acc;
        }
    }
}

/**
 * @brief 2-D SAME cross-correlation (no flip inside) with padding.
 * @param F   H×W image.
 * @param H,W Dimensions of F.
 * @param G   KH×KW kernel.
 * @param KH,KW Kernel dims.
 * @param Y   Output H×W.
 * @param pmod Padding policy (zero/none/const).
 * @param cval Constant pad value (if PAD_CONST).
 */
void corr2d_same(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y, pad_mode pmod, float cval)
{
    const int cH = KH / 2, cW = KW / 2;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(runtime)
#endif
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            int u0 = cH - i;
            if (u0 < 0)
                u0 = 0;
            int u1 = (H - 1 + cH) - i;
            if (u1 > KH - 1)
                u1 = KH - 1;
            int v0 = cW - j;
            if (v0 < 0)
                v0 = 0;
            int v1 = (W - 1 + cW) - j;
            if (v1 > KW - 1)
                v1 = KW - 1;

            double acc = 0.0;
            for (int u = u0; u <= u1; u++)
            {
                const int ii = i + (u - cH);
#ifdef _OPENMP
#pragma omp simd reduction(+ : acc)
#endif
                for (int v = v0; v <= v1; v++)
                {
                    const int jj = j + (v - cW);
                    acc += (double)F[IDX2(ii, jj, W)] * (double)G[IDX2(u, v, KW)];
                }
            }

            if (pmod == PAD_CONST)
            {
                double pad_sum = 0.0;
                for (int u = 0; u < u0; u++)
                    for (int v = 0; v < KW; v++)
                        pad_sum += (double)G[IDX2(u, v, KW)];
                for (int u = u1 + 1; u < KH; u++)
                    for (int v = 0; v < KW; v++)
                        pad_sum += (double)G[IDX2(u, v, KW)];
                for (int u = u0; u <= u1; u++)
                {
                    for (int v = 0; v < v0; v++)
                        pad_sum += (double)G[IDX2(u, v, KW)];
                    for (int v = v1 + 1; v < KW; v++)
                        pad_sum += (double)G[IDX2(u, v, KW)];
                }
                acc += (double)cval * pad_sum;
            }
            /* PAD_ZERO & PAD_NONE: no extra contribution */

            Y[IDX2(i, j, W)] = (float)acc;
        }
    }
}

/* ------------------------------ Timing & misc ------------------------------ */

/**
 * @brief Elapsed time in seconds between two timespecs.
 * @param a Start time.
 * @param b End time.
 * @return Elapsed time in seconds.
 */
double elapsed_seconds(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}
/**
 * @brief Ensure directory exists (mkdir if needed).
 * @param path Directory path.
 */
void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        (void)mkdir(path, 0775);
}

/* ------------------------------- Metrics CSV ------------------------------- */

/**
 * @brief Log run metrics (CSV), including GFLOPS and OpenMP config.
 * Columns: RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,time_s,gflops,threads,schedule,chunk
 */
void log_metrics(int H, int W, int KH, int KW, int outH, int outW,
                 op_kind op, conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs, double gflops,
                 int threads, const char *sched_str, int chunk)
{
    ensure_dir("metrics/omp/o3");

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
    snprintf(fname, sizeof(fname), "metrics/omp/o3/metrics_%s.csv", runid);
    FILE *csv = fopen(fname, "w");
    if (!csv)
    {
        perror(fname);
        return;
    }

    fprintf(csv, "RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,time_s,gflops,threads,schedule,chunk\n");
    fprintf(csv, "%s,%d,%d,%d,%d,%d,%d,%s,%s,%s,%.6g,%.9f,%.6f,%d,%s,%d\n",
            runid, H, W, KH, KW, outH, outW,
            (op == OP_CONV ? "conv" : "corr"),
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            elapsed_secs, gflops, threads, (sched_str ? sched_str : "none"), chunk);
    fclose(csv);
}

/* --------------------------------- Main --------------------------------- */
int main(int argc, char **argv)
{
    const char *img_path = NULL, *ker_path = NULL, *out_path = NULL;
    long H_req = -1, W_req = -1, KH_req = -1, KW_req = -1;
    unsigned long seed = (unsigned long)time(NULL);
    int have_seed = 0;
    op_kind op = OP_CONV;
    conv_mode cmode = MODE_SAME;
    pad_mode pmode = PAD_ZERO;
    float cval = 0.0f;
    int have_cval = 0;

    int threads = 1, have_threads = 0;
    omp_sched_t sched_kind;
#ifdef _OPENMP
    sched_kind = omp_sched_static;
#else
    sched_kind = 1;
#endif
    int have_sched = 0;
    int chunk = -1, have_chunk = 0;

    if (!parse_args(argc, argv,
                    &img_path, &ker_path, &out_path,
                    &H_req, &W_req, &KH_req, &KW_req,
                    &seed, &have_seed,
                    &op, &cmode, &pmode, &cval, &have_cval,
                    &threads, &have_threads, &sched_kind, &have_sched,
                    &chunk, &have_chunk))
        return EXIT_FAILURE;

    srand((unsigned)seed);

    /* Prepare inputs: file or RNG */
    int H = 0, W = 0;
    float *F = img_path ? read_matrix_2d(img_path, &H, &W)
                        : (H = (int)H_req, W = (int)W_req, gen_matrix_2d(H, W));
    int KH = 0, KW = 0;
    float *G = ker_path ? read_matrix_2d(ker_path, &KH, &KW)
                        : (KH = (int)KH_req, KW = (int)KW_req, gen_matrix_2d(KH, KW));

    /* Convolution uses flipped kernel; flip outside timing */
    float *G_use = G, *G_flip = NULL;
    if (op == OP_CONV)
    {
        G_flip = flip_kernel_2d(G, KH, KW);
        G_use = G_flip;
    }

    /* Output allocation */
    const int outH = (cmode == MODE_FULL) ? (H + KH - 1) : H;
    const int outW = (cmode == MODE_FULL) ? (W + KW - 1) : W;
    float *Y = (float *)malloc((size_t)outH * (size_t)outW * sizeof(float));
    if (!Y)
    {
        fprintf(stderr, "oom allocating output (%dx%d)\n", outH, outW);
        free(F);
        free(G);
        if (G_flip)
            free(G_flip);
        return EXIT_FAILURE;
    }

    /* ---- OpenMP defaults (deterministic) ---- */
#ifdef _OPENMP
    if (!have_threads)
        threads = 1;
    omp_set_dynamic(0);
    omp_set_nested(0);
    omp_set_num_threads(threads);

    if (!have_chunk)
    {
        long work = (long)outH * (long)outW; /* linearized collapsed loops */
        long tgt = (long)threads * 4;
        long ch = (work + tgt - 1) / tgt;
        if (ch < 1)
            ch = 1;
        chunk = (int)ch;
    }
    if (!have_sched)
        sched_kind = omp_sched_static;
    omp_set_schedule(sched_kind, chunk);

    omp_sched_t qk;
    int qc;
    omp_get_schedule(&qk, &qc);
    const char *sched_str = schedule_to_string(qk);
#else
    threads = 1;
    if (chunk < 1)
        chunk = 1;
    const char *sched_str = "none";
    if (have_threads || have_sched || have_chunk)
        fprintf(stderr, "note: compiled without OpenMP; ignoring --threads/--schedule/--chunk\n");
#endif

    /* First-touch init (outside timing) */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < outH * outW; i++)
        Y[i] = 0.0f;

    /* ---- Time ONLY the correlation kernel ---- */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (cmode == MODE_FULL)
        corr2d_full(F, H, W, G_use, KH, KW, Y);
    else
        corr2d_same(F, H, W, G_use, KH, KW, Y, pmode, cval);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    const double secs = elapsed_seconds(t0, t1);

    /* Simple FLOP model (approx; SAME ignores trimmed edges) */
    double flops = 2.0 * (double)H * (double)W * (double)KH * (double)KW;
    double gflops = (secs > 0.0) ? (flops / secs / 1e9) : 0.0;

    /* Human-readable timing */
    fprintf(stderr,
            "op=%s H=%d W=%d KH=%d KW=%d outH=%d outW=%d mode=%s pad=%s cval=%.6g | "
            "conv_time=%.9f s | %.3f GFLOP/s | OMP{threads=%d sched=%s chunk=%d}\n",
            (op == OP_CONV ? "conv" : "corr"),
            H, W, KH, KW, outH, outW,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? cval : 0.0),
            secs, gflops, threads, sched_str, chunk);

    /* Metrics CSV */
    log_metrics(H, W, KH, KW, outH, outW, op, cmode, pmode, cval, secs, gflops, threads, sched_str, chunk);

    /* Write output (assignment format) */
    write_matrix_2d(out_path, Y, outH, outW);

    free(F);
    free(G);
    free(Y);
    if (G_flip)
        free(G_flip);
    return EXIT_SUCCESS;
}