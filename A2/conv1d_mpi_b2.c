/**
 * @file conv1d_mpi.c
 * @brief Pure-MPI 1-D Convolution with SAME/FULL modes, padding, STRIDE, RNG, assignment I/O,
 *        and per-run CSV metrics (incl. GFLOP/s). Uses B2-style output-range + per-rank f-halo.
 *
 * Features (parity with conv1d.c):
 *  • TRUE convolution (y = f * g), not correlation.
 *  • Modes (-m): same (default), full.
 *  • Padding (-p) for SAME: zero (default), none, const (-c/--cval).
 *  • Stride (-st/--stride) downsampling; output lengths:
 *      SAME -> ceil(N / stride), FULL -> ceil((N + K - 1) / stride).
 *  • Inputs: files (-f/-g) or RNG (-L/--len, -kL/--klen) with seed (-se/--seed, legacy -s).
 *  • Output: assignment format (length line, then values with 3 decimals).
 *  • Timing: ONLY the convolution kernel (distribution/gather excluded via barriers around compute).
 *  • Metrics: CSV per run under metrics/ with RunID, N, K, outLen, mode, padding, cval, time, gflops.
 *
 * MPI design (B2-style):
 *  • Split outputs (n) evenly across ranks:
 *       SAME: n in [0..N-1],      FULL: n in [0..N+K-2]
 *  • Each rank computes only stride-aligned outputs in its n-range.
 *  • Each rank receives exactly the contiguous f-slice (halo) needed to cover all its n:
 *       For a block [n0..n1), required f ∈ [max(0,n0-(K-1)) .. min(N-1,n1-1)] (FULL)
 *       SAME also uses a single contiguous f-slice; out-of-range is handled per padding policy.
 *  • g is small ⇒ broadcast to all ranks.
 *  • Gather stride outputs with MPI_Gatherv to form the final y on rank 0.
 *
 * Build:
 *   mpicc -std=c11 -O2 -Wall -Wextra -Werror -o conv1d_mpi conv1d_mpi.c
 *   # Add -DDEBUG_MPI for per-rank debug prints.
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
static inline int min_i(int a, int b) { return (a < b) ? a : b; }
static inline int max_i(int a, int b) { return (a > b) ? a : b; }

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

void ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        (void)mkdir(path, 0775);
}

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

/* ------------------------------ I/O helpers ------------------------------ */
float *read_array_1d(const char *path, int *len_out)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        perror(path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int L = 0;
    if (fscanf(fp, "%d", &L) != 1 || L <= 0)
    {
        fprintf(stderr, "bad header in %s (expected positive integer length)\n", path);
        fclose(fp);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    float *arr = (float *)malloc((size_t)L * sizeof(float));
    if (!arr)
    {
        fprintf(stderr, "OOM reading %s\n", path);
        fclose(fp);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < L; i++)
    {
        if (fscanf(fp, "%f", &arr[i]) != 1)
        {
            fprintf(stderr, "bad body in %s at index %d\n", path, i);
            free(arr);
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
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
        MPI_Abort(MPI_COMM_WORLD, 1);
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
        fprintf(stderr, "invalid length %d\n", n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    float *a = (float *)malloc((size_t)n * sizeof(float));
    if (!a)
    {
        fprintf(stderr, "OOM gen_array_1d n=%d\n", n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < n; i++)
    {
        float u01 = (float)rand() / (float)RAND_MAX; /* [0,1] */
        a[i] = -1.0f + 2.0f * u01;                   /* [-1,1] */
    }
    return a;
}

/* ------------------------------ Metrics ------------------------------ */
void log_metrics(int N, int K, int outLen,
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

/* ------------------------------ CLI parsing ------------------------------ */
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

    /* Pre-filter -L/-kL and multi-letter -se/-st for convenience. */
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
                fprintf(stderr, "-L requires arg\n");
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
                fprintf(stderr, "-kL requires arg\n");
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
                fprintf(stderr, "-se requires arg\n");
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
                fprintf(stderr, "-st requires arg\n");
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
        {"seed", required_argument, 0, 's'}, /* legacy short -s accepted */
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
        case 1:
            *N_req = strtol(optarg, NULL, 10);
            break; /* --len */
        case 2:
            *K_req = strtol(optarg, NULL, 10);
            break; /* --klen */
        case 6:
            *stride = (int)strtol(optarg, NULL, 10);
            break; /* --stride */
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
        fprintf(stderr, "Missing -L/--len\n");
        return 0;
    }
    if (!*g_path && *K_req <= 0)
    {
        fprintf(stderr, "Missing -kL/--klen\n");
        return 0;
    }
    if (*pmode == PAD_CONST && !*have_cval)
        fprintf(stderr, "warning: -p const without -c; using cval=0.0\n");
    if (*stride <= 0)
    {
        fprintf(stderr, "stride must be >= 1\n");
        return 0;
    }
    return 1;
}

/* ------------------------------ SAME (B2: output-range + f-halo) ------------------------------ */
static void same_mode_output_halo(int N, const float *f_global,
                                  int K, const float *g,
                                  pad_mode pmod, float cval,
                                  int stride,
                                  int rank, int P,
                                  float **y_global_out, int *yLen_out)
{
    const int outLen = ceil_div(N, stride);
    const int c = K / 2;

    /* Partition SAME outputs n ∈ [0..N-1] across ranks (by raw n). */
    int base = N / P, rem = N % P;
    int n0 = rank * base + (rank < rem ? rank : rem);
    int n1 = n0 + base + (rank < rem ? 1 : 0);

    /* f slice covering [n0..n1) for SAME: all valid idx = n + (m - c) falling in [0..N-1].
       A safe single contiguous slice is simply [i_lo, i_hi] = [max(0,n0-(K-1)), min(N-1,n1-1)].
       If no outputs, empty slice. */
    int i_lo = 0, i_hi = -1;
    if (n0 < n1)
    {
        i_lo = max_i(0, n0 - (K - 1));
        i_hi = min_i(N - 1, n1 - 1);
    }
    int n_local_f = (i_lo <= i_hi) ? (i_hi - i_lo + 1) : 0;

    /* Ship per-rank f slice point-to-point. */
    float *f_local = NULL;
    if (rank == 0)
    {
        for (int r = 0; r < P; r++)
        {
            int r_base = N / P, r_rem = N % P;
            int rn0 = r * r_base + (r < r_rem ? r : r_rem);
            int rn1 = rn0 + r_base + (r < r_rem ? 1 : 0);
            int ri_lo = 0, ri_hi = -1;
            if (rn0 < rn1)
            {
                ri_lo = max_i(0, rn0 - (K - 1));
                ri_hi = min_i(N - 1, rn1 - 1);
            }
            int r_cnt = (ri_lo <= ri_hi) ? (ri_hi - ri_lo + 1) : 0;

            if (r == 0)
            {
                if (r_cnt > 0)
                {
                    f_local = (float *)malloc((size_t)r_cnt * sizeof(float));
                    memcpy(f_local, f_global + ri_lo, (size_t)r_cnt * sizeof(float));
                }
            }
            else
            {
                MPI_Send(&r_cnt, 1, MPI_INT, r, 100, MPI_COMM_WORLD);
                if (r_cnt > 0)
                    MPI_Send(f_global + ri_lo, r_cnt, MPI_FLOAT, r, 101, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        MPI_Recv(&n_local_f, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n_local_f > 0)
        {
            f_local = (float *)malloc((size_t)n_local_f * sizeof(float));
            MPI_Recv(f_local, n_local_f, MPI_FLOAT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

#ifdef DEBUG_MPI
    fprintf(stderr,
            "[rank %d/%d] SAME: N=%d K=%d stride=%d | n_range=[%d..%d) | f_slice=[%d..%d] count=%d\n",
            rank, P, N, K, stride, n0, n1, i_lo, i_hi, n_local_f);
#endif

    /* Local stride outputs: n = t*stride ∈ [n0..n1) => t ∈ [t_lo..t_hi). */
    int t_lo = ceil_div(n0, stride);
    int t_hi = ceil_div(n1, stride);
    int my_out = (t_hi > t_lo) ? (t_hi - t_lo) : 0;

    float *y_local = NULL;
    if (my_out > 0)
    {
        y_local = (float *)malloc((size_t)my_out * sizeof(float));
        if (!y_local)
        {
            fprintf(stderr, "OOM y_local SAME\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    for (int t = 0; t < my_out; t++)
    {
        int n = (t_lo + t) * stride; /* global output index in [n0..n1) */
        double acc = 0.0;
        for (int m = 0; m < K; m++)
        {
            int idx = n + (m - c); /* input index */
            if (idx >= 0 && idx < N)
            {
                int li = idx - i_lo; /* map to local f slice */
                acc += (double)f_local[li] * (double)g[m];
            }
            else if (pmod == PAD_CONST)
            {
                acc += (double)cval * (double)g[m];
            } /* PAD_ZERO & PAD_NONE => skip */
        }
        y_local[t] = (float)acc;
    }

    /* Gather to y (length = ceil(N/stride)). */
    int my_two[2] = {t_lo, t_hi};
    int *all_two = NULL, *counts = NULL, *displs = NULL;
    float *y_global = NULL;

    if (rank == 0)
        all_two = (int *)malloc(2 * P * sizeof(int));
    MPI_Gather(my_two, 2, MPI_INT, all_two, 2, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        counts = (int *)malloc(P * sizeof(int));
        displs = (int *)malloc(P * sizeof(int));
        int disp = 0;
        for (int r = 0; r < P; r++)
        {
            int r_lo = all_two[2 * r + 0];
            int r_hi = all_two[2 * r + 1];
            int cnt = (r_hi > r_lo) ? (r_hi - r_lo) : 0;
            counts[r] = cnt;
            displs[r] = disp;
            disp += cnt;
        }
        y_global = (float *)malloc((size_t)outLen * sizeof(float));
    }

    MPI_Gatherv(y_local, my_out, MPI_FLOAT,
                y_global, counts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        *y_global_out = y_global;
        *yLen_out = outLen;
    }

    if (rank == 0)
    {
        free(all_two);
        free(counts);
        free(displs);
    }
    if (y_local)
        free(y_local);
    if (f_local)
        free(f_local);
}

/* ------------------------------ FULL (B2: output-range + f-halo) ------------------------------ */
static void full_mode_output_halo(int N, const float *f_global,
                                  int K, const float *g,
                                  int stride,
                                  int rank, int P,
                                  float **y_global_out, int *yLen_out)
{
    const int fullLen = N + K - 1; /* n in [0..fullLen-1] */
    const int outLen = ceil_div(fullLen, stride);

    /* Partition FULL outputs by raw n. */
    int base = fullLen / P, rem = fullLen % P;
    int n0 = rank * base + (rank < rem ? rank : rem);
    int n1 = n0 + base + (rank < rem ? 1 : 0);

    /* Per-rank f slice covering [n0..n1):
       For a given n: i ∈ [max(0, n-(K-1)) .. min(n, N-1)].
       Over the whole block:
         i_lo = max(0, n0-(K-1)), i_hi = min(N-1, n1-1).
    */
    int i_lo = 0, i_hi = -1;
    if (n0 < n1)
    {
        i_lo = max_i(0, n0 - (K - 1));
        i_hi = min_i(N - 1, n1 - 1);
    }
    int n_local_f = (i_lo <= i_hi) ? (i_hi - i_lo + 1) : 0;

    /* Ship per-rank f slice point-to-point. */
    float *f_local = NULL;
    if (rank == 0)
    {
        for (int r = 0; r < P; r++)
        {
            int r_base = fullLen / P, r_rem = fullLen % P;
            int rn0 = r * r_base + (r < r_rem ? r : r_rem);
            int rn1 = rn0 + r_base + (r < r_rem ? 1 : 0);
            int ri_lo = 0, ri_hi = -1;
            if (rn0 < rn1)
            {
                ri_lo = max_i(0, rn0 - (K - 1));
                ri_hi = min_i(N - 1, rn1 - 1);
            }
            int r_cnt = (ri_lo <= ri_hi) ? (ri_hi - ri_lo + 1) : 0;

            if (r == 0)
            {
                if (r_cnt > 0)
                {
                    f_local = (float *)malloc((size_t)r_cnt * sizeof(float));
                    memcpy(f_local, f_global + ri_lo, (size_t)r_cnt * sizeof(float));
                }
            }
            else
            {
                MPI_Send(&r_cnt, 1, MPI_INT, r, 300, MPI_COMM_WORLD);
                if (r_cnt > 0)
                    MPI_Send(f_global + ri_lo, r_cnt, MPI_FLOAT, r, 301, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        MPI_Recv(&n_local_f, 1, MPI_INT, 0, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n_local_f > 0)
        {
            f_local = (float *)malloc((size_t)n_local_f * sizeof(float));
            MPI_Recv(f_local, n_local_f, MPI_FLOAT, 0, 301, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

#ifdef DEBUG_MPI
    fprintf(stderr,
            "[rank %d/%d] FULL: fullLen=%d K=%d stride=%d | n_range=[%d..%d) | f_slice=[%d..%d] count=%d\n",
            rank, P, fullLen, K, stride, n0, n1, i_lo, i_hi, n_local_f);
#endif

    /* Local stride outputs: n = t*stride in [n0..n1) ⇒ t ∈ [t_lo..t_hi). */
    int t_lo = ceil_div(n0, stride);
    int t_hi = ceil_div(n1, stride);
    int my_out = (t_hi > t_lo) ? (t_hi - t_lo) : 0;

    float *y_local = NULL;
    if (my_out > 0)
    {
        y_local = (float *)malloc((size_t)my_out * sizeof(float));
        if (!y_local)
        {
            fprintf(stderr, "OOM y_local FULL\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    for (int t = 0; t < my_out; t++)
    {
        int n = (t_lo + t) * stride; /* global output index in [0..fullLen-1] */
        int i_start = max_i(0, n - (K - 1));
        int i_end = min_i(n, N - 1);
        double acc = 0.0;
        for (int i = i_start; i <= i_end; i++)
        {
            int j = n - i;     /* 0..K-1 */
            int li = i - i_lo; /* map to local f slice */
            acc += (double)f_local[li] * (double)g[j];
        }
        y_local[t] = (float)acc;
    }

    /* Gather to y (length = ceil(fullLen/stride)). */
    int my_two[2] = {t_lo, t_hi};
    int *all_two = NULL, *counts = NULL, *displs = NULL;
    float *y_global = NULL;

    if (rank == 0)
        all_two = (int *)malloc(2 * P * sizeof(int));
    MPI_Gather(my_two, 2, MPI_INT, all_two, 2, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        counts = (int *)malloc(P * sizeof(int));
        displs = (int *)malloc(P * sizeof(int));
        int disp = 0;
        for (int r = 0; r < P; r++)
        {
            int r_lo = all_two[2 * r + 0];
            int r_hi = all_two[2 * r + 1];
            int cnt = (r_hi > r_lo) ? (r_hi - r_lo) : 0;
            counts[r] = cnt;
            displs[r] = disp;
            disp += cnt;
        }
        y_global = (float *)malloc((size_t)outLen * sizeof(float));
    }

    MPI_Gatherv(y_local, my_out, MPI_FLOAT,
                y_global, counts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        *y_global_out = y_global;
        *yLen_out = outLen;
    }

    if (rank == 0)
    {
        free(all_two);
        free(counts);
        free(displs);
    }
    if (y_local)
        free(y_local);
    if (f_local)
        free(f_local);
}

/* ------------------------------ Main ------------------------------ */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    const char *f_path = NULL, *g_path = NULL, *o_path = NULL;
    long N_req = -1, K_req = -1;
    unsigned long seed = (unsigned long)time(NULL);
    int have_seed = 0;
    conv_mode cmode = MODE_SAME;
    pad_mode pmode = PAD_ZERO;
    float cval = 0.0f;
    int have_cval = 0;
    int stride = 1;

    /* Only rank 0 parses CLI; broadcast config. */
    int ok = 1;
    if (rank == 0)
    {
        ok = parse_args(argc, argv, &f_path, &g_path, &o_path,
                        &N_req, &K_req, &seed, &have_seed,
                        &cmode, &pmode, &cval, &have_cval,
                        &stride);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok)
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* Broadcast scalar config (sizes later). */
    int cm = (int)cmode, pm = (int)pmode;
    MPI_Bcast(&cm, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cmode = (conv_mode)cm;
    MPI_Bcast(&pm, 1, MPI_INT, 0, MPI_COMM_WORLD);
    pmode = (pad_mode)pm;
    MPI_Bcast(&cval, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stride, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    srand((unsigned)seed); /* for RNG gen on rank 0, but harmless everywhere */

    int N = 0, K = 0;
    float *f_global = NULL, *g_global = NULL;

    if (rank == 0)
    {
        /* Prepare f and g */
        if (f_path)
            f_global = read_array_1d(f_path, &N);
        else
        {
            N = (int)N_req;
            f_global = gen_array_1d(N);
        }

        if (g_path)
            g_global = read_array_1d(g_path, &K);
        else
        {
            K = (int)K_req;
            g_global = gen_array_1d(K);
        }

        if (pmode == PAD_CONST && !have_cval)
        {
            fprintf(stderr, "warning: -p const without -c; cval=0.0\n");
        }
    }

    /* Broadcast N,K and g. (f distributed in slices later.) */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        g_global = (float *)malloc((size_t)K * sizeof(float));
    MPI_Bcast(g_global, K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* ---- Kernel timing: exclude I/O & distribution (barrier before compute; barrier after) ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    float *y_global = NULL;
    int outLen = 0;
    if (cmode == MODE_SAME)
    {
        same_mode_output_halo(N, f_global, K, g_global, pmode, cval, stride, rank, P,
                              &y_global, &outLen);
    }
    else
    {
        full_mode_output_halo(N, f_global, K, g_global, stride, rank, P,
                              &y_global, &outLen);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /* Reduce to the maximum kernel time across ranks. */
    double local = t1 - t0, secs = 0.0;
    MPI_Reduce(&local, &secs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Report & metrics (rank 0) */
    if (rank == 0)
    {
        /* FLOP model ~ 2 * outLen * K (same as seq for comparability). */
        double flops = 2.0 * (double)outLen * (double)K;
        double gflops = (secs > 0.0) ? (flops / secs / 1e9) : 0.0;
        fprintf(stderr,
                "N=%d K=%d outLen=%d mode=%s pad=%s cval=%.6g stride=%d | ranks=%d | conv_time=%.9f s | %.3f GFLOP/s\n",
                N, K, outLen,
                (cmode == MODE_FULL ? "full" : "same"),
                (pmode == PAD_ZERO ? "zero" : (pmode == PAD_NONE ? "none" : "const")),
                (pmode == PAD_CONST ? cval : 0.0f),
                stride, P, secs, gflops);

        /* Write output & metrics */
        write_array_1d(o_path, y_global, outLen);
        log_metrics(N, K, outLen, cmode, pmode, cval, secs, gflops);
    }

    /* Debug range prints (after compute to avoid interleaving) */
#ifdef DEBUG_MPI
    if (cmode == MODE_SAME)
    {
        int base = N / P, rem = N % P;
        int n0 = rank * base + (rank < rem ? rank : rem);
        int n1 = n0 + base + (rank < rem ? 1 : 0);
        fprintf(stderr, "[rank %d/%d] SAME done: n_range=[%d..%d)\n", rank, P, n0, n1);
    }
    else
    {
        int fullLen = N + K - 1;
        int base = fullLen / P, rem = fullLen % P;
        int n0 = rank * base + (rank < rem ? rank : rem);
        int n1 = n0 + base + (rank < rem ? 1 : 0);
        fprintf(stderr, "[rank %d/%d] FULL done: n_range=[%d..%d)\n", rank, P, n0, n1);
    }
#endif

    /* Cleanup */
    if (rank == 0)
    {
        free(f_global);
        free(g_global);
        free(y_global);
    }
    else
    {
        free(g_global);
        /* f_local/y_local already freed in helpers */
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}