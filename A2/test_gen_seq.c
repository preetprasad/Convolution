#include <stdio.h>
#include <stdlib.h>

int main() {
    srand(42);
    printf("F (6x6):\n");
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float u01 = (float)rand() / (float)RAND_MAX;
            float val = -1.0f + 2.0f * u01;
            printf("%.3f ", val);
        }
        printf("\n");
    }
    printf("\nG (3x3):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float u01 = (float)rand() / (float)RAND_MAX;
            float val = -1.0f + 2.0f * u01;
            printf("%.3f ", val);
        }
        printf("\n");
    }
    return 0;
}
