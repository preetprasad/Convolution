#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

float *gen_matrix_2d(int H, int W) {
    float *A = (float *)malloc((size_t)H * (size_t)W * sizeof(float));
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            float u01 = (float)rand() / (float)RAND_MAX;
            A[i * W + j] = -1.0f + 2.0f * u01;
        }
    return A;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        srand(42);
        float *F = gen_matrix_2d(6, 6);
        printf("F (6x6) from rank 0:\n");
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                printf("%.3f ", F[i * 6 + j]);
            }
            printf("\n");
        }
        float *G = gen_matrix_2d(3, 3);
        printf("\nG (3x3) from rank 0:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%.3f ", G[i * 3 + j]);
            }
            printf("\n");
        }
        free(F);
        free(G);
    }
    
    MPI_Finalize();
    return 0;
}
