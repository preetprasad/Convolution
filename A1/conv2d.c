/**
 * @file conv2d.c
 * @brief 2-D Correlation/Convolution (sequential) with SAME/FULL modes, padding strategies,
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
 *   - same (default) : out size = H × W
 *   - full           : out size = (H + KH − 1) × (W + KW − 1)
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
 *       -s <seed> | --seed <seed>           : deterministic seed (defaults to time if omitted)
 *
 * ### Output
 *   - Assignment format:
 *       Line 1: "outH outW"
 *       Next   : outH lines × outW floats (3 decimals, space-separated)
 *   - Flag:
 *       -o <out.txt> | --out <out.txt>
 *
 * ### CSV metrics (one row per run)
 *   - Stored under metrics/:
 *       • SLURM: metrics/metrics_SLURM_<JOBID>.csv
 *       • Local: metrics/metrics_LOCAL_YYYYMMDD_HHMMSS_<PID>.csv
 *   - Fields: RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,time
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
 *   # Convolution (default), SAME, ZERO pad, random 1024×1024 with 5×5 kernel
 *   ./conv2d -H 1024 -W 1024 -kH 5 -kW 5 -o Y.txt -s 42
 *
 *   # Correlation, FULL, file inputs
 *   ./conv2d --corr -f img.txt -g ker.txt -m full -o Y.txt
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
               op_kind *op, conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval);

float *read_matrix_2d(const char *path, int *H_out, int *W_out);
void write_matrix_2d(const char *path, const float *A, int H, int W);
float *gen_matrix_2d(int H, int W);

float *flip_kernel_2d(const float *G, int KH, int KW); /* convolution helper (allocates flipped) */

void corr2d_full(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y /* outH = H+KH-1, outW = W+KW-1 */);

void corr2d_same(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y /* H×W */,
                 pad_mode pmod, float cval);

double elapsed_seconds(struct timespec a, struct timespec b);
void ensure_dir(const char *path);
void log_metrics(int H, int W, int KH, int KW, int outH, int outW,
                 op_kind op, conv_mode cmode, pad_mode pmode, float cval,
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
            "[-s seed|--seed seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value]\n",
            prog);
}

/**
 * @brief Parse CLI, supporting file/RNG inputs, op (conv/corr), mode/padding/cval, output path.
 * @return 1 on success; 0 on usage/validation failure.
 */
int parse_args(int argc, char **argv,
               const char **img_path, const char **ker_path, const char **out_path,
               long *H_req, long *W_req, long *KH_req, long *KW_req,
               unsigned long *seed, int *have_seed,
               op_kind *op, conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval)
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

    /* ---- Pre-scan & filter: accept -kH/-kW and -kH=/-kW= ---- */
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
        {"rows", required_argument, 0, 'H'}, /* -H */
        {"cols", required_argument, 0, 'W'}, /* -W */
        {"krows", required_argument, 0, 1},  /* --krows */
        {"kcols", required_argument, 0, 2},  /* --kcols */
        {0, 0, 0, 0}};

    int opt, idx = 0;
    opterr = 0;
    while ((opt = getopt_long(filtered_argc, filtered_argv, "f:g:o:s:m:p:c:H:W:", long_opts, &idx)) != -1)
    {
        switch (opt)
        {
        case 10:
            *op = OP_CONV;
            break; /* --conv */
        case 11:
            *op = OP_CORR;
            break; /* --corr */
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
            break; /* --krows */
        case 2:
            *KW_req = strtol(optarg, NULL, 10);
            break; /* --kcols */

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

/* ------------------------- Core correlation kernels ------------------------- */

/**
 * @brief 2-D FULL cross-correlation (output size: (H+KH−1)×(W+KW−1)).
 *
 * Mathematical form (no kernel flip here):
 *   Y[oi,oj] = sum_i sum_j F[i,j] * G[oi - i, oj - j], over valid overlaps only.
 *
 * Implemented as an output-driven gather with tight index bounds so that
 * (u,v) = (oi - i, oj - j) always lies in [0..KH-1]×[0..KW-1].
 * Accumulates in double, stores to float.
 */
void corr2d_full(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y)
{
    const int outH = H + KH - 1;
    const int outW = W + KW - 1;

    for (int oi = 0; oi < outH; oi++)
        for (int oj = 0; oj < outW; oj++)
            Y[IDX2(oi, oj, outW)] = 0.0f;

    for (int oi = 0; oi < outH; oi++)
    {
        const int i0 = (oi - (KH - 1) > 0) ? (oi - (KH - 1)) : 0;
        const int i1 = (oi < (H - 1)) ? oi : (H - 1);
        for (int oj = 0; oj < outW; oj++)
        {
            const int j0 = (oj - (KW - 1) > 0) ? (oj - (KW - 1)) : 0;
            const int j1 = (oj < (W - 1)) ? oj : (W - 1);
            double acc = 0.0;
            for (int i = i0; i <= i1; i++)
            {
                const int u = oi - i; /* 0..KH-1 */
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
 * @brief 2-D SAME cross-correlation (output size: H×W) with optional padding.
 *
 * Mathematical form (no kernel flip here):
 *   Y[i,j] = sum_{u=0..KH-1} sum_{v=0..KW-1} F[i + (u - cH), j + (v - cW)] * G[u,v]
 * where cH = floor(KH/2), cW = floor(KW/2).
 *
 * Implementation trims (u,v) ranges so (ii,jj) = (i + u - cH, j + v - cW) stays in bounds.
 * For PAD_CONST, adds cval times the sum of G over the out-of-bounds area.
 * Accumulates in double, stores to float.
 */
void corr2d_same(const float *F, int H, int W,
                 const float *G, int KH, int KW,
                 float *Y,
                 pad_mode pmod, float cval)
{
    const int cH = KH / 2, cW = KW / 2;

    for (int i = 0; i < H; i++)
    {
        int u0 = cH - i;
        if (u0 < 0)
            u0 = 0;
        int u1 = (H - 1 + cH) - i;
        if (u1 > KH - 1)
            u1 = KH - 1;
        for (int j = 0; j < W; j++)
        {
            int v0 = cW - j;
            if (v0 < 0)
                v0 = 0;
            int v1 = (W - 1 + cW) - j;
            if (v1 > KW - 1)
                v1 = KW - 1;

            double acc = 0.0;

            /* In-bounds rectangle (no branches in inner loop) */
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
                /* Add cval * sum of G outside the in-bounds rectangle */
                double pad_sum = 0.0;

                for (int u = 0; u < u0; u++) /* rows fully above */
                    for (int v = 0; v < KW; v++)
                        pad_sum += (double)G[IDX2(u, v, KW)];
                for (int u = u1 + 1; u < KH; u++) /* rows fully below */
                    for (int v = 0; v < KW; v++)
                        pad_sum += (double)G[IDX2(u, v, KW)];

                for (int u = u0; u <= u1; u++)
                { /* side margins on middle rows */
                    for (int v = 0; v < v0; v++)
                        pad_sum += (double)G[IDX2(u, v, KW)];
                    for (int v = v1 + 1; v < KW; v++)
                        pad_sum += (double)G[IDX2(u, v, KW)];
                }

                acc += (double)cval * pad_sum;
            }

            /* PAD_ZERO & PAD_NONE: no extra work */
            Y[IDX2(i, j, W)] = (float)acc;
        }
    }
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
 * @brief Log run metrics (CSV) into metrics/ with a unique RunID.
 * Fields: RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,time
 */
void log_metrics(int H, int W, int KH, int KW, int outH, int outW,
                 op_kind op, conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs)
{
    ensure_dir("metrics/o3");

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

    fprintf(csv, "RunID,H,W,KH,KW,outH,outW,op,mode,padding,cval,time\n");
    fprintf(csv, "%s,%d,%d,%d,%d,%d,%d,%s,%s,%s,%.9g,%.9f\n",
            runid, H, W, KH, KW, outH, outW,
            (op == OP_CONV ? "conv" : "corr"),
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            elapsed_secs);
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

    if (!parse_args(argc, argv,
                    &img_path, &ker_path, &out_path,
                    &H_req, &W_req, &KH_req, &KW_req,
                    &seed, &have_seed,
                    &op, &cmode, &pmode, &cval, &have_cval))
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

    /* Allocate output */
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

    /* Time ONLY the correlation kernel (G may be flipped already for conv) */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (cmode == MODE_FULL)
    {
        corr2d_full(F, H, W, G_use, KH, KW, Y);
    }
    else
    {
        corr2d_same(F, H, W, G_use, KH, KW, Y, pmode, cval);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double secs = elapsed_seconds(t0, t1);

    /* Human-readable timing (stderr) */
    fprintf(stderr,
            "op=%s H=%d W=%d KH=%d KW=%d outH=%d outW=%d mode=%s pad=%s cval=%.6g | conv_time=%.9f s\n",
            (op == OP_CONV ? "conv" : "corr"),
            H, W, KH, KW, outH, outW,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? cval : 0.0), secs);

    /* Persist per-run metrics */
    log_metrics(H, W, KH, KW, outH, outW, op, cmode, pmode, cval, secs);

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