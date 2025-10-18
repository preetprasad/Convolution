/**
 * @file conv1d_mpi.c
 * @brief TRUE halo-exchange MPI 1-D Convolution with modes, padding, STRIDE, RNG/file I/O,
 *        kernel-only timing (MPI_Wtime), GFLOP/s, per-rank timing summary, and MPI-IO binary output.
 *
 * Features:
 *   • SAME / FULL modes, padding (zero|none|const), STRIDE (-st/--stride)
 *   • Inputs: files (-f/-g) or RNG (-L/-kL, -se/--seed, legacy -s)
 *   • Non-blocking halo exchange (MPI_Irecv/Isend + MPI_Waitall)
 *   • Times ONLY compute (excludes I/O + halo), reports GFLOP/s
 *   • Per-rank timing (min/avg/max) rolled into the root summary line
 *   • MPI-IO binary output (default):
 *       - int32 header = outLen, followed by outLen float32 values
 *       - each rank writes its segment at correct byte offset with MPI_File_write_at_all
 *       - uses MPI_Type_contiguous(my_out_count, MPI_FLOAT, &segtype)
 *   • Optional human-readable text output via --text (otherwise binary default)
 *   • Per-run CSV metrics (RunID, N, K, outLen, mode, padding, cval, time, gflops)
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
#include <math.h>
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
typedef enum
{
    OUT_BIN = 0,
    OUT_TEXT = 1
} out_mode;

/* ----------------------------- Prototypes ----------------------------- */
void usage(const char *prog);
int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *stride, out_mode *omode);

float *read_array_1d(const char *path, int *len_out);
void write_array_1d_text(const char *path, const float *arr, int len);
float *gen_array_1d(int n);

void ensure_dir(const char *path);
double elapsed_seconds(struct timespec a, struct timespec b); /* kept for parity */
void log_metrics(int N, int K, int outLen,
                 conv_mode cmode, pad_mode pmode, float cval,
                 double elapsed_secs, double gflops);

/* ----------------------------- Utilities ----------------------------- */
void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-f f.txt | -L N | --len N] "
            "[-g g.txt | -kL K | --klen K] "
            "-o out|--out out "
            "[-se seed|--seed seed|-s seed] "
            "[-m same|full|--mode same|full] "
            "[-p zero|none|const|--padding zero|none|const] "
            "[-c value|--cval value] "
            "[-st stride|--stride stride] "
            "[--text|--binary]\n",
            prog);
}

int parse_args(int argc, char **argv,
               const char **f_path, const char **g_path, const char **o_path,
               long *N_req, long *K_req,
               unsigned long *seed, int *have_seed,
               conv_mode *cmode, pad_mode *pmode, float *cval, int *have_cval,
               int *stride, out_mode *omode)
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
    *omode = OUT_BIN; /* default switched to binary per request */

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

        if (!strcmp(a, "--text"))
        {
            *omode = OUT_TEXT;
            continue;
        }
        if (!strcmp(a, "--binary"))
        {
            *omode = OUT_BIN;
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
        {"stride", required_argument, 0, 't'}, /* legacy -t removed; long kept */
        {"text", no_argument, 0, 6},
        {"binary", no_argument, 0, 7},
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
        case 't': /* legacy short stride was removed intentionally; prefer -st */
            *stride = (int)strtol(optarg, NULL, 10);
            break;
        case 1:
            *N_req = strtol(optarg, NULL, 10);
            break; /* --len */
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break; /* --klen */
        case 6:
            *omode = OUT_TEXT;
            break; /* --text */
        case 7:
            *omode = OUT_BIN;
            break; /* --binary */
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
void write_array_1d_text(const char *path, const float *arr, int len)
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
            (pmode == PAD_CONST ? (double)cval : 0.0),
            elapsed_secs, gflops);
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
    out_mode omode = OUT_BIN;

    if (!parse_args(argc, argv, &f_path, &g_path, &o_path,
                    &N_req, &K_req, &seed, &have_seed,
                    &cmode, &pmode, &cval, &have_cval,
                    &stride, &omode))
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

    /* Build ghosted buffer with true halo exchange (non-blocking) */
    const int H = (K > 0 ? K - 1 : 0); /* safe halo width */
    const int buf_len = local_count + 2 * H;
    float *buf = (float *)malloc((size_t)buf_len * sizeof(float));
    if (!buf)
    {
        fprintf(stderr, "[%d] OOM buf\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < buf_len; i++)
        buf[i] = 0.0f; /* halos default 0; const pad applied in compute */
    memcpy(buf + H, f_local, (size_t)local_count * sizeof(float));

    int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Request reqs[4];
    int nreq = 0;
    int nL = (H <= local_count) ? H : local_count;
    int nR = (H <= local_count) ? H : local_count;

    /* Post Irecvs first, then Isends */
    /* recv LEFT halo into buf[0..H-1] (tag 102) */
    MPI_Irecv(buf, H, MPI_FLOAT, left, 102, MPI_COMM_WORLD, &reqs[nreq++]);
    /* recv RIGHT halo into buf[H+local_count .. H+local_count+H-1] (tag 101) */
    MPI_Irecv(buf + (H + local_count), H, MPI_FLOAT, right, 101, MPI_COMM_WORLD, &reqs[nreq++]);

    /* send LEFT boundary (our first nL) to left (tag 101) */
    MPI_Isend(f_local, nL, MPI_FLOAT, left, 101, MPI_COMM_WORLD, &reqs[nreq++]);
    /* send RIGHT boundary (our last nR) to right (tag 102) */
    MPI_Isend((local_count >= nR ? f_local + (local_count - nR) : f_local),
              nR, MPI_FLOAT, right, 102, MPI_COMM_WORLD, &reqs[nreq++]);

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

#ifdef DEBUG_MPI
    /* One clean, ordered line per rank describing work domain */
    int fullLen = N + K - 1;
    int outLen = (cmode == MODE_FULL) ? ceil_div(fullLen, stride) : ceil_div(N, stride);
    int base_out = outLen / size, rem_out = outLen % size;
    int my_out_start = rank * base_out + (rank < rem_out ? rank : rem_out);
    int my_out_count = base_out + (rank < rem_out ? 1 : 0);
    int my_out_end = my_out_start + my_out_count;
    for (int r = 0; r < size; r++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r)
        {
            fprintf(stderr,
                    "[rank %d/%d] %s: N=%d K=%d stride=%d | "
                    "n_range=[%d..%d) (out_cnt=%d) | "
                    "f_core=[%d..%d] (core_cnt=%d) | H=%d\n",
                    rank, size, (cmode == MODE_FULL ? "FULL" : "SAME"),
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

    /* Output layout and my slice (recompute here for non-DEBUG too) */
    int fullLen2 = N + K - 1;
    int outLen2 = (cmode == MODE_FULL) ? ceil_div(fullLen2, stride) : ceil_div(N, stride);
    int base_out2 = outLen2 / size, rem_out2 = outLen2 % size;
    int my_out_start2 = rank * base_out2 + (rank < rem_out2 ? rank : rem_out2);
    int my_out_count2 = base_out2 + (rank < rem_out2 ? 1 : 0);

    float *y_local = (float *)malloc((size_t)my_out_count2 * sizeof(float));
    if (!y_local)
    {
        fprintf(stderr, "[%d] OOM y_local\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Kernel timing: ONLY the math using MPI_Wtime() */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (cmode == MODE_SAME)
    {
        const int c = K / 2;
        for (int j = 0; j < my_out_count2; j++)
        {
            const int n = (my_out_start2 + j) * stride; /* global n in [0..N-1] */
            double acc = 0.0;
            for (int m = 0; m < K; m++)
            {
                int idx = n - (m - c); /* global index into f */
                if (idx >= 0 && idx < N)
                {
                    int in_local = idx - local_start + H;
                    if (in_local >= 0 && in_local < buf_len)
                        acc += (double)buf[in_local] * (double)g[m];
                }
                else if (pmode == PAD_CONST)
                {
                    acc += (double)cval * (double)g[m];
                }
            }
            y_local[j] = (float)acc;
        }
    }
    else /* MODE_FULL */
    {
        for (int j = 0; j < my_out_count2; j++)
        {
            const int n = (my_out_start2 + j) * stride; /* 0..N+K-2 */
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
    fprintf(stderr, "[rank %d/%d] kernel_time=%.9f s (out_cnt=%d)\n",
            rank, size, local_secs, my_out_count2);
#endif

    /* Global performance metrics: sum flops, min/avg/max time */
    double local_flops = 2.0 * (double)K * (double)my_out_count2;
    double sum_flops = 0.0, max_secs = 0.0, min_secs = 0.0, sum_secs = 0.0;
    MPI_Reduce(&local_flops, &sum_flops, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_secs, &max_secs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_secs, &min_secs, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_secs, &sum_secs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Gather output counts to build displacements for both gather and MPI-IO */
    int *recvcounts = NULL, *rdispls = NULL;
    if (rank == 0)
    {
        recvcounts = (int *)malloc((size_t)size * sizeof(int));
        rdispls = (int *)malloc((size_t)size * sizeof(int));
    }
    MPI_Gather(&my_out_count2, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        int off = 0;
        for (int r = 0; r < size; r++)
        {
            rdispls[r] = off;
            off += recvcounts[r];
        }
    }

    /* Optional text output path for compatibility */
    if (omode == OUT_TEXT)
    {
        float *y_root = NULL;
        if (rank == 0)
        {
            y_root = (float *)malloc((size_t)outLen2 * sizeof(float));
            if (!y_root)
            {
                fprintf(stderr, "[root] OOM y_root\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        MPI_Gatherv(y_local, my_out_count2, MPI_FLOAT,
                    y_root, recvcounts, rdispls, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            double gflops = (max_secs > 0.0) ? (sum_flops / (max_secs * 1e9)) : 0.0;
            double avg_secs = sum_secs / (double)size;
            fprintf(stderr,
                    "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g stride=%d | ranks=%d | "
                    "conv_time=%.9f s | %.3f GFLOP/s | perRank(min/avg/max)=%.9f/%.9f/%.9f s\n",
                    N, K, outLen2,
                    (cmode == MODE_FULL ? "full" : "same"),
                    (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                    (pmode == PAD_CONST ? cval : 0.0),
                    stride, size, max_secs, gflops, min_secs, avg_secs, max_secs);

            write_array_1d_text(o_path, y_root, outLen2);
            log_metrics(N, K, outLen2, cmode, pmode, cval, max_secs, gflops);
        }
        if (rank == 0)
        {
            free(y_root);
            free(recvcounts);
            free(rdispls);
        }
        free(f_local);
        free(buf);
        free(g);
        free(y_local);
        MPI_Finalize();
        return 0;
    }

    /* ---- Binary MPI-IO path (default) ----
       File layout: [int32 outLen] then outLen float32 values (ranked segments in order 0..P-1). */
    MPI_File fh;
    MPI_Status st;
    int mpierr = MPI_File_open(MPI_COMM_WORLD, (char *)o_path,
                               MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &fh);
    if (mpierr != MPI_SUCCESS)
    {
        if (rank == 0)
            fprintf(stderr, "MPI_File_open failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* --- Round y_local values to 3 decimal places (match sequential text output) --- */
    for (int j = 0; j < my_out_count2; ++j)
    {
        // Scale, round, and rescale to 3 decimal places
        float scaled = y_local[j] * 1000.0f;
        float rounded = nearbyintf(scaled); // ties-to-even, same as printf("%.3f")
        y_local[j] = rounded * 0.001f;
    }

    /* Rank 0 writes the header (outLen) at offset 0 */
    if (rank == 0)
    {
        int outLen_hdr = outLen2;
        MPI_File_write_at(fh, /*offset*/ 0, &outLen_hdr, 1, MPI_INT, &st);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Compute file offset directly from my global output start index */
    MPI_Offset header_bytes = (MPI_Offset)sizeof(int);
    MPI_Offset my_byte_offset = header_bytes + (MPI_Offset)my_out_start2 * (MPI_Offset)sizeof(float);

    /* Build a contiguous datatype for my segment and write with count=1 */
    MPI_Datatype segtype;
    MPI_Type_contiguous(my_out_count2, MPI_FLOAT, &segtype);
    MPI_Type_commit(&segtype);

    /* Collective write (each rank at its own offset) */
    MPI_File_write_at_all(fh, my_byte_offset, y_local, 1, segtype, &st);

    MPI_Type_free(&segtype);
    MPI_File_close(&fh);

    /* Root report and metrics */
    if (rank == 0)
    {
        double gflops = (max_secs > 0.0) ? (sum_flops / (max_secs * 1e9)) : 0.0;
        double avg_secs = sum_secs / (double)size;
        fprintf(stderr,
                "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g stride=%d | ranks=%d | "
                "conv_time=%.9f s | %.3f GFLOP/s | perRank(min/avg/max)=%.9f/%.9f/%.9f s\n",
                N, K, outLen2,
                (cmode == MODE_FULL ? "full" : "same"),
                (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                (pmode == PAD_CONST ? cval : 0.0),
                stride, size, max_secs, gflops, min_secs, avg_secs, max_secs);

        log_metrics(N, K, outLen2, cmode, pmode, cval, max_secs, gflops);

        /* cleanup recv arrays */
        free(recvcounts);
        free(rdispls);
    }

    free(f_local);
    free(buf);
    free(g);
    free(y_local);

    MPI_Finalize();
    return 0;
}