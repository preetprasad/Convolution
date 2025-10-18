#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        srand(42);
        printf("First 10 random numbers from rank 0:\n");
        for (int i = 0; i < 10; i++) {
            float u01 = (float)rand() / (float)RAND_MAX;
            float val = -1.0f + 2.0f * u01;
            printf("%.6f ", val);
        }
        printf("\n");
    }
    
    MPI_Finalize();
    return 0;
}
