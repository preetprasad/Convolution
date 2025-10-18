/**
 * @file conv2d.c
 * @brief 2-D Correlation/Convolution (sequential) with SAME/FULL modes, STRIDE, padding strategies,
 *        reproducible RNG, assignment-compliant I/O, and per-run CSV metrics.
 *
 * This program implements:
 *   • Mathematical 2-D cross-correlation  (Y = F ⊗ G)  — implemented by corr2d_*() functions.
 *   • Mathematical 2-D convolution        (Y = F * G)  — realized as corr2d_*(F, flip(G)).
 *
 * ### Operation selection (one of):
 *   - default (no flag) or --conv : perform convolution ⇒ internally flip kernel once, then call corr2d_*.
 *   - --corr                      : perform correlation ⇒ no flip, call corr2d_* directly.
 *
 * ### Output size modes (-m/--mode)
 *   - same (default) : out size = ceil(H/sH) × ceil(W/sW)
 *   - full           : out size = ceil((H+KH-1)/sH) × ceil((W+KW-1)/sW)
 *
 * ### Stride control (-sH/-sW or --stride-h/--stride-w):
 *   - sH (default 1): vertical stride (row sampling)
 *   - sW (default 1): horizontal stride (column sampling)
 *   - stride > 1 reduces output size and compute cost proportionally
 *   - Example: sH=2, sW=2 → compute every 2nd row and 2nd column
 *
 * ### Padding policies (-p/--padding), SAME mode only
 *   - zero  (default): out-of-range reads contribute 0.0
 *   - none           : out-of-range reads are skipped (no contribution)
 *   - const          : out-of-range reads contribute cval × G[u,v]  (use -c/--cval)
 *
 * ### Input options
 *   - Read matrices (assignment format: header "H W", then H×W floats, row-major):
 *       -f <img.txt> | --file <img.txt>     : image F
 *       -g <ker.txt> | --kernel <ker.txt>   : kernel G
 *   - Generate random matrices U([-1,1]):
 *       -H <H> | --rows <H>                 : image height
 *       -W <W> | --cols <W>                 : image width
 *       -kH <KH> | --krows <KH>             : kernel height
 *       -kW <KW> | --kcols <KW>             : kernel width
 *   - RNG reproducibility:
 *       -s <seed> | -se <seed> | --seed <seed> : deterministic seed (defaults to time if omitted)
 *
 * ### Output
 *   - Assignment format:
 *       Line 1: "outH outW"
 *       Next   : outH lines × outW floats (3 decimals, space-separated)
 *   - Flag:
 *       -o <out.txt> | --out <out.txt>
 *
 * ### CSV metrics (one row per run)
 *   - Stored under metrics/o3/:
 *       • SLURM: metrics/o3/metrics_SLURM_<JOBID>.csv
 *       • Local: metrics/o3/metrics_LOCAL_YYYYMMDD_HHMMSS_<PID>.csv
 *   - Fields: RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,sH,sW,time
 *
 * ### Numerical policy
 *   - Arrays in float32
 *   - Accumulation in double, then cast to float
 *
 * ### Timing policy
 *   - Only the core corr2d_* kernel is timed
 *   - Preprocessing (e.g., kernel flip for convolution), I/O, RNG are excluded
 *
 * ### Build
 *   cc -std=c11 -O2 -Wall -Wextra -Werror -o conv2d conv2d.c
 *
 * ### Examples
 *   # Convolution (default), SAME, ZERO pad, random 1024×1024 with 5×5 kernel, stride 1×1
 *   ./conv2d -H 1024 -W 1024 -kH 5 -kW 5 -o Y.txt -s 42
 *
 *   # Correlation, FULL, file inputs, stride 2×2 (downsampling)
 *   ./conv2d --corr -f img.txt -g ker.txt -m full -sH 2 -sW 2 -o Y.txt
 *
 *   # SAME mode with constant padding, stride 3×1
 *   ./conv2d -H 512 -W 512 -kH 7 -kW 7 -p const -c 1.5 -sH 3 -sW 1 -o Y.txt
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
#include <math.h>

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

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void usage(const char *prog);
int parse_args(int argc, char **argv,
               const char **img_path, const char **ker_path, const char **out_path,
               long *H_req, long *W_req, long *KH_req, long *KW_req,
               unsigned long *seed, int *have_seed,
               op_kind *op, conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *sH, int *sW);

float *read_matrix_2d(const char *path, int *H_out, int *W_out);
void write_matrix_2d(const char *path, const float *A, int H, int W);
float *gen_matrix_2d(int H, int W);

float *flip_kernel_2d(const float *G, int KH, int KW);

/* Core stride implementations */
void corr2d_full_stride(const float *F, int H, int W,
                        const float *G, int KH, int KW,
                        int sH, int sW,
                        float *Y);

void corr2d_same_stride(const float *F, int H, int W,
                        const float *G, int KH, int KW,
                        int sH, int sW,
                        float *Y,
                        pad_mode pmod, float cval,
                        int use_conv_anchor);

/* Back-compat wrappers (no duplication) */
void corr2d_full(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y);

void corr2d_same(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y,
                 pad_mode pmod, float cval);

double elapsed_seconds(struct timespec a, struct timespec b);
void ensure_dir(const char *path);
void log_metrics(int H, int W, int KH, int KW, int outH, int outW,
                 op_kind op, conv_mode cmode, pad_mode pmode, float cval,
                 int sH, int sW,
                 double elapsed_secs);

/* ------------------------------ Utilities ------------------------------ */

#define IDX2(i, j, ldW) ((size_t)(i) * (size_t)(ldW) + (size_t)(j))

/**
 * @brief Print usage message to stderr.
 */
void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--conv|--corr] "
            "[-f img.txt | -H H --rows H -W W --cols W] "
            "[-g ker.txt | -kH KH --krows KH -kW KW --kcols KW] "
            "-o out.txt|--out out.txt "
            "[-s seed|-se seed|--seed seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value] "
            "[-sH stride_h|--stride-h stride_h] "
            "[-sW stride_w|--stride-w stride_w]\n",
            prog);
}

/**
 * @brief Parse CLI, supporting file/RNG inputs, op (conv/corr), mode/padding/cval, stride, output path.
 * @return 1 on success; 0 on usage/validation failure.
 */
int parse_args(int argc, char **argv,
               const char **img_path, const char **ker_path, const char **out_path,
               long *H_req, long *W_req, long *KH_req, long *KW_req,
               unsigned long *seed, int *have_seed,
               op_kind *op, conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *sH, int *sW)
{
    *img_path = *ker_path = *out_path = NULL;
    *H_req = *W_req = *KH_req = *KW_req = -1;
    *seed = (unsigned long)time(NULL);
    *have_seed = 0;
    *op = OP_CONV; /* default = convolution */
    *cmode = MODE_SAME;
    *pmode = PAD_ZERO;
    *cval = 0.0f;
    *have_cval = 0;
    *sH = 1;
    *sW = 1;

    /* ---- Pre-scan & filter: accept -kH/-kW and -sH/-sW ---- */
    int filtered_argc = 1; /* keep argv[0] */
    char **filtered_argv = (char **)malloc((size_t)argc * sizeof(char *));
    if (!filtered_argv)
    {
        perror("malloc filtered_argv");
        return 0;
    }
    filtered_argv[0] = argv[0];

    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];

        /* Handle -se (seed) before -sH/-sW to avoid ambiguity */
        if (strcmp(a, "-se") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-se requires an argument\n");
                free(filtered_argv);
                return 0;
            }
            *seed = strtoul(argv[++i], NULL, 10);
            *have_seed = 1;
            continue;
        }
        if (strncmp(a, "-se=", 4) == 0)
        {
            *seed = strtoul(a + 4, NULL, 10);
            *have_seed = 1;
            continue;
        }

        if (strcmp(a, "-kH") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-kH requires an argument\n");
                free(filtered_argv);
                return 0;
            }
            *KH_req = strtol(argv[++i], NULL, 10);
            continue;
        }
        if (strncmp(a, "-kH=", 4) == 0)
        {
            *KH_req = strtol(a + 4, NULL, 10);
            continue;
        }

        if (strcmp(a, "-kW") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-kW requires an argument\n");
                free(filtered_argv);
                return 0;
            }
            *KW_req = strtol(argv[++i], NULL, 10);
            continue;
        }
        if (strncmp(a, "-kW=", 4) == 0)
        {
            *KW_req = strtol(a + 4, NULL, 10);
            continue;
        }

        /* Stride height and width */
        if (strcmp(a, "-sH") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-sH requires an argument\n");
                free(filtered_argv);
                return 0;
            }
            *sH = (int)strtol(argv[++i], NULL, 10);
            continue;
        }
        if (strncmp(a, "-sH=", 4) == 0)
        {
            *sH = (int)strtol(a + 4, NULL, 10);
            continue;
        }

        if (strcmp(a, "-sW") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "-sW requires an argument\n");
                free(filtered_argv);
                return 0;
            }
            *sW = (int)strtol(argv[++i], NULL, 10);
            continue;
        }
        if (strncmp(a, "-sW=", 4) == 0)
        {
            *sW = (int)strtol(a + 4, NULL, 10);
            continue;
        }

        filtered_argv[filtered_argc++] = argv[i];
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
    while ((opt = getopt_long(filtered_argc, filtered_argv, "f:g:o:s:m:p:c:H:W:", long_opts, &idx)) != -1)
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
                usage(filtered_argv[0]);
                free(filtered_argv);
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
                usage(filtered_argv[0]);
                free(filtered_argv);
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
            break;
        case 2:
            *KW_req = strtol(optarg, NULL, 10);
            break;

        case 3:
            *sH = (int)strtol(optarg, NULL, 10);
            break;
        case 4:
            *sW = (int)strtol(optarg, NULL, 10);
            break;

        default:
            break;
        }
    }

    free(filtered_argv);

    /* Validation */
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
    if (*sH < 1)
    {
        fprintf(stderr, "sH (stride height) must be >= 1\n");
        return 0;
    }
    if (*sW < 1)
    {
        fprintf(stderr, "sW (stride width) must be >= 1\n");
        return 0;
    }

    return 1;
}

/**
 * @brief Read H×W matrix from assignment-style file: "H W" then H×W floats (row-major).
 * @return newly malloc'd float array; caller frees.
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
 * @brief Write H×W matrix to assignment-style file (header + values with 3 decimals).
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
 * @brief Generate H×W matrix with U([-1,1]) floats.
 * @return newly malloc'd float array; caller frees.
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

/**
 * @brief Produce a flipped copy of kernel G (both axes) for convolution.
 *        Gf[u,v] = G[KH-1-u, KW-1-v].
 * @return newly malloc'd flipped kernel; caller frees.
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

/* ======================== Core correlation kernels with STRIDE ======================== */

/**
 * @brief 2-D FULL cross-correlation with stride (output size: ceil((H+KH-1)/sH) × ceil((W+KW-1)/sW)).
 *
 * Mathematical form (no kernel flip):
 *   Y[oi_out, oj_out] = sum_i sum_j F[i,j] * G[oi - i, oj - j], over valid overlaps only.
 *   where oi = oi_out * sH, oj = oj_out * sW
 *
 * Accumulates in double, stores to float.
 */
void corr2d_full_stride(const float *F, int H, int W,
                        const float *G, int KH, int KW,
                        int sH, int sW,
                        float *Y)
{
    const int fullH = H + KH - 1;
    const int fullW = W + KW - 1;
    const int outH = ceil_div(fullH, sH);
    const int outW = ceil_div(fullW, sW);

    for (int oi_out = 0; oi_out < outH; oi_out++)
    {
        const int oi = oi_out * sH;
        const int i0 = (oi - (KH - 1) > 0) ? (oi - (KH - 1)) : 0;
        const int i1 = (oi < H) ? oi : (H - 1);

        for (int oj_out = 0; oj_out < outW; oj_out++)
        {
            const int oj = oj_out * sW;
            const int j0 = (oj - (KW - 1) > 0) ? (oj - (KW - 1)) : 0;
            const int j1 = (oj < W) ? oj : (W - 1);

            double acc = 0.0;
            for (int i = i0; i <= i1; i++)
            {
                const int u = oi - i;
                for (int j = j0; j <= j1; j++)
                {
                    const int v = oj - j;
                    acc += (double)F[IDX2(i, j, W)] * (double)G[IDX2(u, v, KW)];
                }
            }
            Y[IDX2(oi_out, oj_out, outW)] = (float)acc;
        }
    }
}

/**
 * @brief 2-D SAME cross-correlation with stride (output size: ceil(H/sH) × ceil(W/sW)) with optional padding.
 *
 * Mathematical form (no kernel flip):
 *   Y[i_out, j_out] = sum_{u=0..KH-1} sum_{v=0..KW-1} F[i + (u - cH), j + (v - cW)] * G[u,v]
 *   where i = i_out * sH, j = j_out * sW, cH = floor(KH/2), cW = floor(KW/2).
 *
 * NOTE: use_conv_anchor parameter controls anchor point calculation:
 *   - use_conv_anchor=0 (correlation): cH = (KH-1)/2  (SciPy correlate2d behavior)
 *   - use_conv_anchor=1 (convolution): cH = KH/2      (SciPy convolve2d behavior)
 *
 * Accumulates in double, stores to float.
 */
void corr2d_same_stride(const float *F, int H, int W,
                        const float *G, int KH, int KW,
                        int sH, int sW,
                        float *Y,
                        pad_mode pmod, float cval,
                        int use_conv_anchor)
{
    const int cH = use_conv_anchor ? (KH / 2) : ((KH - 1) / 2);
    const int cW = use_conv_anchor ? (KW / 2) : ((KW - 1) / 2);
    const int outH = ceil_div(H, sH);
    const int outW = ceil_div(W, sW);

    for (int i_out = 0; i_out < outH; i_out++)
    {
        const int i = i_out * sH;
        if (i >= H)
            break;

        int u0 = cH - i;
        if (u0 < 0)
            u0 = 0;
        int u1 = (H - 1 + cH) - i;
        if (u1 > KH - 1)
            u1 = KH - 1;

        for (int j_out = 0; j_out < outW; j_out++)
        {
            const int j = j_out * sW;
            if (j >= W)
                break;

            int v0 = cW - j;
            if (v0 < 0)
                v0 = 0;
            int v1 = (W - 1 + cW) - j;
            if (v1 > KW - 1)
                v1 = KW - 1;

            double acc = 0.0;

            /* In-bounds rectangle */
            for (int u = u0; u <= u1; u++)
            {
                const int ii = i + (u - cH);
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

            Y[IDX2(i_out, j_out, outW)] = (float)acc;
        }
    }
}

/* ======================== Back-compat wrappers (no duplication) ======================== */

/**
 * @brief 2-D FULL cross-correlation wrapper (default stride sH=1, sW=1).
 */
void corr2d_full(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y)
{
    corr2d_full_stride(F, H, W, G, KH, KW, /*sH=*/1, /*sW=*/1, Y);
}

/**
 * @brief 2-D SAME cross-correlation wrapper (default stride sH=1, sW=1, correlation anchor).
 */
void corr2d_same(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y,
                 pad_mode pmod, float cval)
{
    corr2d_same_stride(F, H, W, G, KH, KW, /*sH=*/1, /*sW=*/1, Y, pmod, cval, /*use_conv_anchor=*/0);
}

/* ------------------------------ Timing & I/O ------------------------------ */

/**
 * @brief Compute (b - a) wall-clock seconds from two timespecs.
 */
double elapsed_seconds(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

/**
 * @brief Ensure a directory exists (mkdir if missing).
 */
void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        (void)mkdir(path, 0775);
}

/**
 * @brief Log run metrics (CSV) into metrics/o3/ with a unique RunID.
 * Fields: RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,sH,sW,time,gflops
 */
void log_metrics(int H, int W, int KH, int KW, int outH, int outW,
                 op_kind op, conv_mode cmode, pad_mode pmode, float cval,
                 int sH, int sW,
                 double elapsed_secs)
{
    ensure_dir("metrics/");
    ensure_dir("metrics/o3/");

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
    snprintf(fname, sizeof(fname), "metrics/o3/metrics_%s.csv", runid);
    FILE *csv = fopen(fname, "w");
    if (!csv)
    {
        perror(fname);
        return;
    }

    /* Calculate GFLOPS: 2 ops (multiply + add) per kernel element per output element */
    double flops = 2.0 * (double)KH * (double)KW * (double)outH * (double)outW;
    double gflops = (elapsed_secs > 0.0) ? (flops / elapsed_secs / 1e9) : 0.0;

    fprintf(csv, "RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,sH,sW,time,gflops\n");
    fprintf(csv, "%s,%d,%d,%d,%d,%d,%d,%s,%s,%s,%.9g,%d,%d,%.9f,%.6f\n",
            runid, H, W, KH, KW, outH, outW,
            (op == OP_CONV ? "conv" : "corr"),
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            sH, sW,
            elapsed_secs, gflops);
    fclose(csv);
}

/* --------------------------------- Main --------------------------------- */

/**
 * @brief Orchestrates I/O, optional kernel flip (for conv), timing, kernel call, and logging.
 * @return EXIT_SUCCESS on success; EXIT_FAILURE otherwise.
 */
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
    int sH = 1, sW = 1;

    if (!parse_args(argc, argv,
                    &img_path, &ker_path, &out_path,
                    &H_req, &W_req, &KH_req, &KW_req,
                    &seed, &have_seed,
                    &op, &cmode, &pmode, &cval, &have_cval,
                    &sH, &sW))
    {
        return EXIT_FAILURE;
    }

    srand((unsigned)seed);

    /* Prepare F (image) */
    int H = 0, W = 0;
    float *F = img_path ? read_matrix_2d(img_path, &H, &W)
                        : (H = (int)H_req, W = (int)W_req, gen_matrix_2d(H, W));

    /* Prepare G (kernel) */
    int KH = 0, KW = 0;
    float *G = ker_path ? read_matrix_2d(ker_path, &KH, &KW)
                        : (KH = (int)KH_req, KW = (int)KW_req, gen_matrix_2d(KH, KW));

    /* Select kernel pointer for timed kernel: flip if doing convolution */
    float *G_use = G;
    float *G_flip = NULL;
    if (op == OP_CONV)
    {
        /* Flip happens outside timing by design */
        G_flip = flip_kernel_2d(G, KH, KW);
        G_use = G_flip;
    }

    /* Allocate output with stride-aware dimensions */
    const int outH = (cmode == MODE_FULL) ? ceil_div(H + KH - 1, sH) : ceil_div(H, sH);
    const int outW = (cmode == MODE_FULL) ? ceil_div(W + KW - 1, sW) : ceil_div(W, sW);
    float *Y = (float *)calloc((size_t)outH * (size_t)outW, sizeof(float));
    if (!Y)
    {
        fprintf(stderr, "oom allocating output (%dx%d)\n", outH, outW);
        free(F);
        free(G);
        if (G_flip)
            free(G_flip);
        return EXIT_FAILURE;
    }

    /* Time ONLY the correlation kernel (G may be flipped already for conv) */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (cmode == MODE_FULL)
    {
        corr2d_full_stride(F, H, W, G_use, KH, KW, sH, sW, Y);
    }
    else
    {
        /* Use different anchor conventions for conv vs corr to match SciPy */
        int use_conv_anchor = (op == OP_CONV) ? 1 : 0;
        corr2d_same_stride(F, H, W, G_use, KH, KW, sH, sW, Y, pmode, cval, use_conv_anchor);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double secs = elapsed_seconds(t0, t1);

    /* Calculate GFLOPS */
    double flops = 2.0 * (double)KH * (double)KW * (double)outH * (double)outW;
    double gflops = (secs > 0.0) ? (flops / secs / 1e9) : 0.0;

    /* Human-readable timing (stderr) */
    fprintf(stderr,
            "op=%s H=%d W=%d KH=%d KW=%d outH=%d outW=%d mode=%s pad=%s cval=%.6g stride=%dx%d | conv_time=%.9f s | %.3f GFLOP/s\n",
            (op == OP_CONV ? "conv" : "corr"),
            H, W, KH, KW, outH, outW,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? cval : 0.0),
            sH, sW, secs, gflops);

    /* Persist per-run metrics */
    log_metrics(H, W, KH, KW, outH, outW, op, cmode, pmode, cval, sH, sW, secs);

    /* Pre-round output to 3 decimal places (to match format precision) */
    for (int i = 0; i < outH * outW; i++)
    {
        double scaled = Y[i] * 1000.0;
        double rounded = nearbyintf(scaled);
        Y[i] = (float)(rounded * 0.001);
    }

    /* Write output */
    write_matrix_2d(out_path, Y, outH, outW);

    /* Cleanup */
    free(F);
    free(G);
    free(Y);
    if (G_flip)
        free(G_flip);

    return EXIT_SUCCESS;
}