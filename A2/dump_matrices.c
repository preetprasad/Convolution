#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

float *flip_kernel_2d(const float *G, int KH, int KW) {
    float *Gf = (float *)malloc((size_t)KH * (size_t)KW * sizeof(float));
    for (int u = 0; u < KH; u++)
        for (int v = 0; v < KW; v++)
            Gf[IDX2(u, v, KW)] = G[IDX2(KH - 1 - u, KW - 1 - v, KW)];
    return Gf;
}

int main() {
    srand(42);
    float *F = gen_matrix_2d(3, 3);
    float *G = gen_matrix_2d(2, 2);
    float *Gf = flip_kernel_2d(G, 2, 2);
    
    printf("F (3x3):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            printf("%.6f ", F[IDX2(i, j, 3)]);
        printf("\n");
    }
    
    printf("\nG (2x2):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++)
            printf("%.6f ", G[IDX2(i, j, 2)]);
        printf("\n");
    }
    
    printf("\nGf (flipped 2x2):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++)
            printf("%.6f ", Gf[IDX2(i, j, 2)]);
        printf("\n");
    }
    
    free(F); free(G); free(Gf);
    return 0;
}
