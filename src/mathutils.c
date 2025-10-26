#include <stdlib.h>
#include <math.h>
#include "mathutils.h"

double **mat_new(int r, int c, int rnd){
    double **m = malloc(r * sizeof(double *));
    for (int i = 0; i < r; i++){
        m[i] = malloc(c * sizeof(double));
        for (int j = 0; j < c; j++){
            m[i][j] = rnd ? ((double)rand() / RAND_MAX * 2 - 1) : 0.0;
        }
    }
    return m;
}

void mat_free(double **m, int r){
    for (int i = 0; i < r; i++){
        free(m[i]);
    }
    free(m);
}

double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

double dsigmoid(double y){
    return y * (1 - y);
}