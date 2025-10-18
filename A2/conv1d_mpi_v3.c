/**
 * @file conv1d_mpi.c
 * @brief TRUE halo-exchange MPI 1-D Convolution with modes, padding, STRIDE, RNG/file I/O,
 *        kernel-only timing (MPI_Wtime), GFLOP/s, per-run CSV metrics, and tidy DEBUG output.
 *
 * This is a pure-MPI implementation of 1-D convolution (y = f * g) that:
 *   • Supports SAME (default) and FULL output length modes.
 *   • Supports padding policies (SAME mode): zero|none|const (-p, -c).
 *   • Implements STRIDE (-st/--stride).
 *   • Reads inputs from files (-f/-g) or generates with RNG (-L/-kL, -se/--seed, legacy -s).
 *   • Writes assignment-style output (length on first line, then values).
 *   • Times ONLY the convolution compute loop using MPI_Wtime (excludes file I/O & halo exchange).
 *   • Reports GFLOP/s (global) using Σ(2*K*local_out_count) / max_kernel_time.
 *   • Logs metrics to a unique CSV file under metrics/ (one row per run).
 *
 * Parallel strategy (Lecture 07 “Halo exchanges” style):
 *   1) Root reads/generates f and g; broadcasts N,K,mode,stride,padding,cval and g[].
 *   2) Contiguous partition of f via MPI_Scatterv (each rank owns f[istart .. iend]).
 *   3) True halo exchange with neighbors using fixed halo width H = K-1:
 *        - receive left halo from rank-1 and right halo from rank+1 via MPI_Sendrecv
 *        - halos at global ends are implicitly out-of-range; padding handled in compute
 *   4) Each rank computes its assigned output indices (even block over n_out) with STRIDE.
 *   5) Root gathers y via MPI_Gatherv and writes to file.
 *
 * SAME mode (centered kernel, c = K/2):
 *   y[n] = sum_{m=0..K-1} f[n - (m - c)] * g[m], with padding behavior for out-of-range.
 *   Output indices n ∈ [0..N-1], sampled every 'stride'.
 *
 * FULL mode:
 *   y[n] = sum_{i=max(0,n-(K-1))..min(n,N-1)} f[i]*g[n-i]
 *   Output indices n ∈ [0..N+K-2], sampled every 'stride'.
 *
 * Notes:
 *   • We choose a safe halo width H = K-1 for both modes, so every rank can compute its
 *     local outputs using only “local + ghost” data. Padding for global out-of-range is
 *     applied in the inner loop (not by fabricating halos).
 *   • The kernel timing starts AFTER halo exchange, measuring only the arithmetic.
 *
 * Build:
 *   mpicc -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c
 *   # Debug prints:
 *   mpicc -DDEBUG_MPI=1 -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c
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

/* ----------------------------- Prototypes ----------------------------- */
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
double elapsed_seconds(struct timespec a, struct timespec b); /* kept for completeness */
void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs, double gflops);

/* ----------------------------- Utilities ----------------------------- */
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

    /* Pre-filter -L/-kL and multi-letter -se/-st so order doesn't matter */
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
        {"seed", required_argument, 0, 's'}, /* legacy -s */
        {"len", required_argument, 0, 1},
        {"klen", required_argument, 0, 2},
        {"mode", required_argument, 0, 3},
        {"padding", required_argument, 0, 4},
        {"cval", required_argument, 0, 5},
        {"stride", required_argument, 0, 't'}, /* legacy -t removed; keep long for completeness */
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
            break; /* legacy */
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
            /* legacy short stride was removed intentionally; prefer -st */
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
        fprintf(stderr, "bad header in %s\n", path);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    float *a = (float *)malloc((size_t)L * sizeof(float));
    if (!a)
    {
        fprintf(stderr, "OOM reading %s\n", path);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < L; i++)
        if (fscanf(fp, "%f", &a[i]) != 1)
        {
            fprintf(stderr, "bad body in %s at %d\n", path, i);
            free(a);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
    fclose(fp);
    *len_out = L;
    return a;
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
        fprintf(stderr, "invalid n=%d\n", n);
        exit(EXIT_FAILURE);
    }
    float *a = (float *)malloc((size_t)n * sizeof(float));
    if (!a)
    {
        fprintf(stderr, "OOM gen n=%d\n", n);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++)
    {
        float u = (float)rand() / (float)RAND_MAX;
        a[i] = -1.0f + 2.0f * u;
    }
    return a;
}
void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        (void)mkdir(path, 0775);
}
double elapsed_seconds(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}
void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs, double gflops)
{
    ensure_dir("metrics/");
    const char *slurm = getenv("SLURM_JOB_ID");
    char runid[128];
    if (slurm && slurm[0])
        snprintf(runid, sizeof(runid), "SLURM_%s", slurm);
    else
    {
        time_t t = time(NULL);
        struct tm tm;
        localtime_r(&t, &tm);
        pid_t pid = getpid();
        strftime(runid, sizeof(runid), "LOCAL_%Y%m%d_%H%M%S", &tm);
        size_t L = strlen(runid);
        snprintf(runid + L, sizeof(runid) - L, "_%d", (int)pid);
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
            (pmode == PAD_CONST ? (double)cval : 0.0), elapsed_secs, gflops);
    fclose(csv);
}

/* ----------------------------- Main ----------------------------- */
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

    /* Parse everywhere (safe); only rank 0 will do file I/O */
    if (!parse_args(argc, argv, &f_path, &g_path, &o_path,
                    &N_req, &K_req, &seed, &have_seed,
                    &cmode, &pmode, &cval, &have_cval, &stride))
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Root prepares inputs */
    int N = 0, K = 0;
    float *f_root = NULL, *g_root = NULL;
    if (rank == 0)
    {
        if (f_path)
            f_root = read_array_1d(f_path, &N);
        else
        {
            N = (int)N_req;
            srand((unsigned)seed);
            f_root = gen_array_1d(N);
        }

        if (g_path)
            g_root = read_array_1d(g_path, &K);
        else
        {
            K = (int)K_req;
            if (!have_seed)
                srand((unsigned)seed);
            g_root = gen_array_1d(K);
        }
    }

    /* Broadcast sizes and scalar config */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cmode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pmode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cval, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stride, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Broadcast g[] */
    float *g = (float *)malloc((size_t)K * sizeof(float));
    if (!g)
    {
        fprintf(stderr, "[%d] OOM g\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0)
        memcpy(g, g_root, (size_t)K * sizeof(float));
    MPI_Bcast(g, K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Partition f and scatter */
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc((size_t)size * sizeof(int));
        displs = (int *)malloc((size_t)size * sizeof(int));
        int base = N / size, rem = N % size, off = 0;
        for (int r = 0; r < size; r++)
        {
            int cnt = base + (r < rem ? 1 : 0);
            sendcounts[r] = cnt;
            displs[r] = off;
            off += cnt;
        }
    }
    int local_count = 0, local_start = 0;
    if (rank == 0)
    {
        local_count = sendcounts[0];
        local_start = displs[0];
    }
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(displs, 1, MPI_INT, &local_start, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float *f_local = (float *)malloc((size_t)local_count * sizeof(float));
    if (!f_local)
    {
        fprintf(stderr, "[%d] OOM f_local\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Scatterv(f_root, sendcounts, displs, MPI_FLOAT,
                 f_local, local_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(f_root);
        free(g_root);
        free(sendcounts);
        free(displs);
    }

    /* Build ghosted buffer with true halo exchange */
    const int H = (K > 0 ? K - 1 : 0); /* safe halo width */
    const int buf_len = local_count + 2 * H;
    float *buf = (float *)malloc((size_t)buf_len * sizeof(float));
    if (!buf)
    {
        fprintf(stderr, "[%d] OOM buf\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < buf_len; i++)
        buf[i] = 0.0f; /* halos init to zero */
    memcpy(buf + H, f_local, (size_t)local_count * sizeof(float));

    /* Neighbor ranks */
    int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Left halo: recv from left, send our left boundary */
    MPI_Sendrecv(
        f_local, (H <= local_count ? H : local_count), MPI_FLOAT, left, 101,
        buf, H, MPI_FLOAT, left, 102,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Right halo: recv from right, send our right boundary */
    MPI_Sendrecv(
        (local_count >= H ? f_local + (local_count - H) : f_local),
        (H <= local_count ? H : local_count), MPI_FLOAT, right, 102,
        buf + (H + local_count), H, MPI_FLOAT, right, 101,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Output layout */
    const int fullLen = N + K - 1;
    const int outLen = (cmode == MODE_FULL) ? ceil_div(fullLen, stride) : ceil_div(N, stride);

    /* Even block decomposition over outLen (in terms of y index j_out = 0..outLen-1) */
    int base_out = outLen / size, rem_out = outLen % size;
    int my_out_start = rank * base_out + (rank < rem_out ? rank : rem_out);
    int my_out_count = base_out + (rank < rem_out ? 1 : 0);
    int my_out_end = my_out_start + my_out_count; /* half-open */

#ifdef DEBUG_MPI
    /* Ordered and detailed debug output incl. output counts */
    for (int r = 0; r < size; r++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r)
        {
            fprintf(stderr,
                    "[rank %d/%d] %s: N=%d K=%d stride=%d | "
                    "n_range=[%d..%d) (out_cnt=%d) | "
                    "f_core=[%d..%d] (core_cnt=%d) | H=%d\n",
                    rank, size,
                    (cmode == MODE_FULL ? "FULL" : "SAME"),
                    N, K, stride,
                    my_out_start * stride,
                    (my_out_end > 0 ? (my_out_end - 1) * stride : -1) + 1,
                    my_out_count,
                    local_start, local_start + local_count - 1, local_count,
                    H);
            fflush(stderr);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* Local output buffer */
    float *y_local = (float *)malloc((size_t)my_out_count * sizeof(float));
    if (!y_local)
    {
        fprintf(stderr, "[%d] OOM y_local\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Kernel timing: ONLY the math loops (use MPI_Wtime) */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (cmode == MODE_SAME)
    {
        const int c = K / 2;
        for (int j = 0; j < my_out_count; j++)
        {
            const int n = (my_out_start + j) * stride; /* global n in [0..N-1] */
            double acc = 0.0;
            for (int m = 0; m < K; m++)
            {
                int idx = n - (m - c); /* global index into f */
                if (idx >= 0 && idx < N)
                {
                    int in_local = idx - local_start + H; /* map to ghosted buffer */
                    if (in_local >= 0 && in_local < buf_len)
                    {
                        acc += (double)buf[in_local] * (double)g[m];
                    }
                }
                else if (pmode == PAD_CONST)
                {
                    acc += (double)cval * (double)g[m];
                }
                /* PAD_ZERO or PAD_NONE => no-op */
            }
            y_local[j] = (float)acc;
        }
    }
    else
    { /* MODE_FULL */
        for (int j = 0; j < my_out_count; j++)
        {
            const int n = (my_out_start + j) * stride; /* 0..N+K-2 */
            double acc = 0.0;
            int i_lo = n - (K - 1);
            if (i_lo < 0)
                i_lo = 0;
            int i_hi = (n < N - 1) ? n : (N - 1);
            for (int i = i_lo; i <= i_hi; i++)
            {
                int in_local = i - local_start + H;
                if (in_local >= 0 && in_local < buf_len)
                {
                    int m = n - i; /* 0..K-1 */
                    acc += (double)buf[in_local] * (double)g[m];
                }
            }
            y_local[j] = (float)acc;
        }
    }

    double t1 = MPI_Wtime();
    double local_secs = t1 - t0;

#ifdef DEBUG_MPI
    /* per-rank timing report */
    for (int r = 0; r < size; r++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r)
        {
            fprintf(stderr, "[rank %d/%d] kernel_time=%.9f s (out_cnt=%d)\n",
                    rank, size, local_secs, my_out_count);
            fflush(stderr);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* Global performance metrics: sum flops, max/min/avg time */
    double local_flops = 2.0 * (double)K * (double)my_out_count;
    double sum_flops = 0.0, max_secs = 0.0, min_secs = 0.0, sum_secs = 0.0;
    MPI_Reduce(&local_flops, &sum_flops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_secs, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_secs, &min_secs, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_secs, &sum_secs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Gather output */
    int *recvcounts = NULL, *rdispls = NULL;
    if (rank == 0)
    {
        recvcounts = (int *)malloc((size_t)size * sizeof(int));
        rdispls = (int *)malloc((size_t)size * sizeof(int));
    }
    MPI_Gather(&my_out_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        int off = 0;
        for (int r = 0; r < size; r++)
        {
            rdispls[r] = off;
            off += recvcounts[r];
        }
    }
    float *y_root = NULL;
    if (rank == 0)
    {
        y_root = (float *)malloc((size_t)outLen * sizeof(float));
        if (!y_root)
        {
            fprintf(stderr, "[root] OOM y_root\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Gatherv(y_local, my_out_count, MPI_FLOAT,
                y_root, recvcounts, rdispls, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /* Root: report and write */
    if (rank == 0)
    {
        double gflops = (max_secs > 0.0) ? (sum_flops / (max_secs * 1e9)) : 0.0;
        double avg_secs = sum_secs / (double)size;
        fprintf(stderr,
                "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g stride=%d | ranks=%d | "
                "conv_time=%.9f s | %.3f GFLOP/s | perRank(min/avg/max)=%.9f/%.9f/%.9f s \n",
                N, K, outLen,
                (cmode == MODE_FULL ? "full" : "same"),
                (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                (pmode == PAD_CONST ? cval : 0.0),
                stride, size, max_secs, gflops,
                min_secs, avg_secs, max_secs);

        write_array_1d(o_path, y_root, outLen);
        log_metrics(N, K, outLen, cmode, pmode, cval, max_secs, gflops);
    }

    free(f_local);
    free(buf);
    free(g);
    free(y_local);
    if (rank == 0)
    {
        free(y_root);
        free(recvcounts);
        free(rdispls);
    }

    MPI_Finalize();
    return 0;
}