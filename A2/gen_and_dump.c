#include <stdio.h>
#include <stdlib.h>

#define IDX2(i, j, ldW) ((size_t)(i) * (size_t)(ldW) + (size_t)(j))

float *gen_matrix_2d(int H, int W) {
    float *A = (float *)malloc((size_t)H * (size_t)W * sizeof(float));
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            float u01 = (float)rand() / (float)RAND_MAX;
            A[IDX2(i, j, W)] = -1.0f + 2.0f * u01;
        }
    return A;
}

void write_matrix(const char *path, float *A, int H, int W) {
    FILE *fp = fopen(path, "w");
    fprintf(fp, "%d %d\n", H, W);
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++)
            fprintf(fp, (j + 1 == W) ? "%.6f\n" : "%.6f ", A[IDX2(i, j, W)]);
    }
    fclose(fp);
}

int main() {
    srand(42);
    float *F = gen_matrix_2d(3, 3);
    float *G = gen_matrix_2d(2, 2);
    write_matrix("F_gen.txt", F, 3, 3);
    write_matrix("G_gen.txt", G, 2, 2);
    printf("Generated F and G with seed 42\n");
    free(F); free(G);
    return 0;
}
