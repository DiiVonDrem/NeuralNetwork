#include <stdio.h>
#include <stdlib.h>
#include "network.h"

int main(){
    srand(42);
    Net *n = net_create(2, 2, 1, 0.5);
    double X[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[4][1] = {{0}, {1}, {1}, {0}};
    double *in[4], *out[4];
    for (int i = 0; i < 4; i++) {
        in[i] = X[i];
        out[i] = Y[i];
    }
    net_train(n, in, out, 4, 10000);
    printf("\nResults:\n");
    for (int i = 0; i < 4; i++)
        net_show(n, X[i]);

    net_free(n);
    return 0;
}