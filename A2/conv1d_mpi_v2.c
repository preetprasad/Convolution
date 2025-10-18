/**
 * @file conv1d_mpi.c
 * @brief Pure-MPI 1-D Convolution with SAME/FULL modes, padding, STRIDE, reproducible RNG,
 *        assignment-compliant text I/O (collective), and per-run CSV metrics (GFLOP/s).
 *
 * Feature parity with conv1d.c:
 *   - Modes: -m same|full
 *   - Padding for SAME: -p zero|none|const  (with -c/--cval)
 *   - Stride: -st/--stride
 *   - Inputs: files (-f/-g) or RNG (-L/--len, -kL/--klen) with seed -se/--seed (legacy -s kept)
 *   - Output: assignment-format text (line 1: length; line 2: L floats, 3 decimals, space-separated)
 *   - Metrics CSV per run under metrics/ (rank 0)
 *   - Deterministic RNG and kernel-only timing
 *
 * MPI specifics:
 *   - Rank 0 reads/generates f and g; broadcasts sizes and arrays.
 *   - Output indices [0..ceil(outLen/stride)-1] block-partitioned across ranks.
 *   - Each rank computes its local block and collectively writes text output using
 *     MPI_File_write_at_all with rank-wise byte offsets computed via MPI_Exscan.
 *   - Rank 0 writes the text header ("<L>\n") at offset 0.
 *
 * Build:
 *   mpicc -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c
 *   # optional debug:
 *   mpicc -std=c11 -O2 -Wall -Wextra -Werror -DDEBUG_MPI=1 -o conv1d_mpi conv1d_mpi.c
 */

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

/* ------------------------------ Utilities ------------------------------ */
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

/* mkdir -p */
static void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        (void)mkdir(path, 0775);
}

/* Per-run CSV metrics (rank 0) */
static void log_metrics(int N, int K, int outLen,
                        conv_mode cmode, pad_mode pmode, float cval,
                        double elapsed_secs, double gflops)
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

    fprintf(csv, "RunID,N,K,outLen,mode,padding,cval,time,gflops\n");
    fprintf(csv, "%s,%d,%d,%d,%s,%s,%.9g,%.9f,%.6f\n",
            runid, N, K, outLen,
            (cmode == MODE_FULL ? "full" : "same"),
            (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
            (pmode == PAD_CONST ? (double)cval : 0.0),
            elapsed_secs, gflops);
    fclose(csv);
}

/* ------------------------------ I/O helpers ------------------------------ */

static float *read_array_1d(const char *path, int *len_out)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        perror(path);
        return NULL;
    }
    int L = 0;
    if (fscanf(fp, "%d", &L) != 1 || L <= 0)
    {
        fprintf(stderr, "bad header in %s\n", path);
        fclose(fp);
        return NULL;
    }
    float *arr = (float *)malloc((size_t)L * sizeof(float));
    if (!arr)
    {
        fprintf(stderr, "OOM reading %s (L=%d)\n", path, L);
        fclose(fp);
        return NULL;
    }
    for (int i = 0; i < L; i++)
    {
        if (fscanf(fp, "%f", &arr[i]) != 1)
        {
            fprintf(stderr, "bad body in %s at index %d\n", path, i);
            free(arr);
            fclose(fp);
            return NULL;
        }
    }
    fclose(fp);
    *len_out = L;
    return arr;
}

static float *gen_array_1d(int n)
{
    if (n <= 0)
    {
        fprintf(stderr, "invalid length n=%d\n", n);
        return NULL;
    }
    float *a = (float *)malloc((size_t)n * sizeof(float));
    if (!a)
    {
        fprintf(stderr, "OOM generating array (n=%d)\n", n);
        return NULL;
    }
    for (int i = 0; i < n; i++)
    {
        float u01 = (float)rand() / (float)RAND_MAX; /* [0,1] */
        a[i] = -1.0f + 2.0f * u01;                   /* [-1,1] */
    }
    return a;
}

/* ------------------------------ Kernel ------------------------------ */

/* Compute output block [outStart..outEnd) for stride-sampled FULL or SAME (with padding).
 * We index outputs in the "stride" domain:
 *   - For SAME: logical output length = N, take indices n = n_out * stride in [0..N-1]
 *   - For FULL: logical length = N+K-1, take n in [0..N+K-2]
 * This function assumes we've already decided cmode and pmod. */
static void conv1d_block(const float *f, int N,
                         const float *g, int K,
                         float *out, int outStart, int outEnd,
                         conv_mode cmode, pad_mode pmod, float cval, int stride)
{
    if (outEnd <= outStart)
        return;

    if (cmode == MODE_FULL)
    {
        for (int n_out = outStart; n_out < outEnd; n_out++)
        {
            const int n = n_out * stride;
            double acc = 0.0;
            /* valid i: max(0, n-(K-1))..min(n, N-1), m=n-i */
            int i_lo = n - (K - 1);
            if (i_lo < 0)
                i_lo = 0;
            int i_hi = (n < (N - 1)) ? n : (N - 1);
            for (int i = i_lo; i <= i_hi; i++)
            {
                int m = n - i;
                acc += (double)f[i] * (double)g[m];
            }
            out[n_out - outStart] = (float)acc;
        }
    }
    else
    {                        /* MODE_SAME */
        const int c = K / 2; /* centered kernel index */
        for (int n_out = outStart; n_out < outEnd; n_out++)
        {
            const int n = n_out * stride; /* logical SAME output index (0..N-1) */
            double acc = 0.0;
            for (int m = 0; m < K; m++)
            {
                int idx = n - (m - c); /* input index */
                if (idx >= 0 && idx < N)
                {
                    acc += (double)f[idx] * (double)g[m];
                }
                else if (pmod == PAD_CONST)
                {
                    acc += (double)cval * (double)g[m];
                }
                else
                {
                    /* PAD_ZERO adds nothing, PAD_NONE => skip */
                }
            }
            out[n_out - outStart] = (float)acc;
        }
    }
}

/* ------------------------------ CLI parsing ------------------------------ */

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-f f.txt | -L N | --len N] "
            "[-g g.txt | -kL K | --klen K] "
            "-o out.txt|--out out.txt "
            "[-se seed|--seed seed|-s seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value] "
            "[-st stride|--stride stride]\n",
            prog);
}

static int parse_args(int argc, char **argv,
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

    /* Pre-filter for -L/-kL and multi-letter -se/-st (like your seq program) */
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
        {"seed", required_argument, 0, 's'},
        {"len", required_argument, 0, 1},
        {"klen", required_argument, 0, 2},
        {"mode", required_argument, 0, 3},
        {"padding", required_argument, 0, 4},
        {"cval", required_argument, 0, 5},
        {"stride", required_argument, 0, 6},
        {0, 0, 0, 0}};

    int opt, idx = 0;
    opterr = 0;
    while ((opt = getopt_long(fargc, fargv, "f:g:o:s:m:p:c:", long_opts, &idx)) != -1)
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
        case 1:
            *N_req = strtol(optarg, NULL, 10);
            break;
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break;
        case 6:
            *stride = (int)strtol(optarg, NULL, 10);
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

/* ------------------------------ Formatting for collective text I/O ------------------------------ */

/* Format local block of outputs as text with:
 *  - "v1 v2 v3 " for all ranks except the global last element which ends with "\n"
 *  - 3 decimals per value
 * Returns malloc'd char* and its length in *len_out.
 * 'globalLastIndex' is the index (0-based) of the last output element in the whole array.
 * We are formatting outputs for global indices [myStart..myEnd).
 */
static char *format_local_text(const float *local_out, int myStart, int myEnd,
                               int globalLastIndex, size_t *len_out)
{
    int count = myEnd - myStart;
    if (count <= 0)
    {
        *len_out = 0;
        return (char *)malloc(1); /* non-NULL */
    }

    /* Worst-case: each float "[-]xxx.xxx" (say up to ~16 chars incl. space), be generous */
    size_t cap = (size_t)count * 32 + 2;
    char *buf = (char *)malloc(cap);
    if (!buf)
    {
        *len_out = 0;
        return NULL;
    }

    size_t off = 0;
    for (int i = 0; i < count; i++)
    {
        int globalIdx = myStart + i;
        int isLast = (globalIdx == globalLastIndex);
        /* Last element gets newline, others get space */
        int n = snprintf(buf + off, cap - off, isLast ? "%.3f\n" : "%.3f ", local_out[i]);
        if (n < 0)
        {
            free(buf);
            *len_out = 0;
            return NULL;
        }
        off += (size_t)n;
        if (off + 32 > cap)
        {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb)
            {
                free(buf);
                *len_out = 0;
                return NULL;
            }
            buf = nb;
        }
    }
    *len_out = off;
    return buf;
}

/* ------------------------------ Main ------------------------------ */

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

    /* Parse CLI on rank 0, broadcast config */
    int ok = 1;
    if (rank == 0)
    {
        ok = parse_args(argc, argv,
                        &f_path, &g_path, &o_path,
                        &N_req, &K_req,
                        &seed, &have_seed,
                        &cmode, &pmode, &cval, &have_cval,
                        &stride);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok)
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Broadcast scalar config values and output path length + string */
    int o_len = 0;
    if (rank == 0)
        o_len = (int)strlen(o_path);
    MPI_Bcast(&o_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *opath_buf = (char *)malloc((size_t)o_len + 1);
    if (rank == 0)
    {
        memcpy(opath_buf, o_path, (size_t)o_len);
        opath_buf[o_len] = '\0';
    }
    MPI_Bcast(opath_buf, o_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0)
        o_path = opath_buf;

    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cmode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pmode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cval, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stride, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Prepare f and g on rank 0; broadcast sizes and data */
    int N = 0, K = 0;
    float *f = NULL, *g = NULL;

    if (rank == 0)
    {
        srand((unsigned)seed);
        if (f_path)
        {
            f = read_array_1d(f_path, &N);
            if (!f)
            {
                fprintf(stderr, "Failed to read f\n");
                ok = 0;
            }
        }
        else
        {
            N = (int)N_req;
            f = gen_array_1d(N);
            if (!f)
            {
                fprintf(stderr, "Failed to gen f\n");
                ok = 0;
            }
        }
        if (g_path)
        {
            g = read_array_1d(g_path, &K);
            if (!g)
            {
                fprintf(stderr, "Failed to read g\n");
                ok = 0;
            }
        }
        else
        {
            K = (int)K_req;
            g = gen_array_1d(K);
            if (!g)
            {
                fprintf(stderr, "Failed to gen g\n");
                ok = 0;
            }
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok)
    {
        if (rank == 0)
        {
            free(f);
            free(g);
        }
        free(opath_buf);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        f = (float *)malloc((size_t)N * sizeof(float));
        g = (float *)malloc((size_t)K * sizeof(float));
        if (!f || !g)
        {
            fprintf(stderr, "[rank %d] OOM\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(f, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g, K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Global stride-aware output length */
    int logicalLen = (cmode == MODE_FULL) ? (N + K - 1) : N;
    int global_outLen = ceil_div(logicalLen, stride);

    /* Block partitioning of [0..global_outLen) */
    int base = global_outLen / size;
    int rem = global_outLen % size;
    int myStart = rank * base + (rank < rem ? rank : rem);
    int myCount = base + (rank < rem ? 1 : 0);
    int myEnd = myStart + myCount;

#ifdef DEBUG_MPI
    fprintf(stderr, "[rank %d/%d] outLen=%d stride=%d | range=[%d..%d) count=%d\n",
            rank, size, global_outLen, stride, myStart, myEnd, myCount);
#endif

    /* Compute local block */
    float *local_out = (myCount > 0) ? (float *)malloc((size_t)myCount * sizeof(float)) : NULL;
    if (myCount > 0 && !local_out)
    {
        fprintf(stderr, "[rank %d] OOM local_out\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    conv1d_block(f, N, g, K, local_out, myStart, myEnd, cmode, pmode, cval, stride);
    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ------------------------------ Collective text output ------------------------------ */
    /* Header: "<global_outLen>\n" written by rank 0 at offset 0 */
    char header[64];
    int header_len = snprintf(header, sizeof(header), "%d\n", global_outLen);
    if (header_len < 0 || header_len >= (int)sizeof(header))
    {
        header_len = 0;
    }

    /* Format my chunk into text */
    size_t my_text_len = 0;
    char *my_text = format_local_text(local_out, myStart, myEnd,
                                      /*globalLastIndex=*/global_outLen - 1, &my_text_len);
    if (!my_text)
    {
        my_text_len = 0;
    }

    /* Compute byte offset for each rank's chunk via exclusive scan on lengths */
    long my_len_long = (long)my_text_len;
    long prefix_bytes = 0;
    MPI_Exscan(&my_len_long, &prefix_bytes, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        prefix_bytes = 0; /* Exscan is undefined for rank 0 input */

    MPI_File fh;
    int ferr = MPI_File_open(MPI_COMM_WORLD, (char *)o_path,
                             MPI_MODE_CREATE | MPI_MODE_WRONLY,
                             MPI_INFO_NULL, &fh);
    if (ferr != MPI_SUCCESS)
    {
        if (rank == 0)
            fprintf(stderr, "MPI_File_open failed\n");
        /* Fallback: rank 0 write (gather) if needed — omitted for brevity */
        /* Clean up and exit */
        free(local_out);
        free(f);
        free(g);
        free(my_text);
        free(opath_buf);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Rank 0 writes header */
    if (rank == 0 && header_len > 0)
    {
        MPI_Status st;
        MPI_File_write_at(fh, /*offset=*/0, header, header_len, MPI_CHAR, &st);
    }

    /* All ranks collectively write their chunk right after header */
    MPI_Offset my_off = (MPI_Offset)header_len + (MPI_Offset)prefix_bytes;
    MPI_Status st_all;
    MPI_File_write_at_all(fh, my_off, my_text, (int)my_text_len, MPI_CHAR, &st_all);

    MPI_File_close(&fh);

    /* ------------------------------ Report + Metrics ------------------------------ */
    if (rank == 0)
    {
        double flops = 2.0 * (double)global_outLen * (double)K;
        double gflops = (max_time > 0.0) ? (flops / max_time / 1e9) : 0.0;
        fprintf(stderr,
                "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g stride=%d | ranks=%d | conv_time=%.9f s | %.3f GFLOP/s\n",
                N, K, global_outLen,
                (cmode == MODE_FULL ? "full" : "same"),
                (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                (pmode == PAD_CONST ? cval : 0.0),
                stride, size, max_time, gflops);

        log_metrics(N, K, global_outLen, cmode, pmode, cval, max_time, gflops);
    }

    /* Cleanup */
    free(local_out);
    free(my_text);
    free(f);
    free(g);
    free(opath_buf);

    MPI_Finalize();
    return EXIT_SUCCESS;
}