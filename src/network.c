#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "network.h"
#include "mathutils.h"

Net *net_create(int in, int hid, int out, double lr){
    Net *n = malloc(sizeof(Net));
    n->inputs = in;
    n->hidden = hid;
    n->outputs = out;
    n->lr = lr;

    n->ih = mat_new(in, hid, 1);
    n->ho = mat_new(hid, out, 1);
    n->h = calloc(hid, sizeof(double));
    n->o = calloc(out, sizeof(double));

    return n;
}

void net_free(Net *n){
    mat_free(n->ih, n->inputs);
    mat_free(n->ho, n->hidden);
    free(n->h);
    free(n->o);
    free(n);
}

void net_forward(Net *n, double *input){
    for (int j = 0; j < n->hidden; j++){
        double sum = 0;
        for (int i = 0; i < n->inputs; i++){
            sum += input[i] * n->ih[i][j];
        }
        n->h[j] = sigmoid(sum);
    }

    for (int k = 0; k < n->outputs; k++){
        double sum = 0;
        for (int j = 0; j < n->hidden; j++){
            sum += n->h[j] * n->ho[j][k];
        }
        n->o[k] = sigmoid(sum);
    }
}

void net_train(Net *n, double **data_in, double **data_out, int samples, int epochs){
    for (int e = 0; e < epochs; e++) {
        double err = 0;

        for (int s = 0; s < samples; s++){
            double *x = data_in[s];
            double *y = data_out[s];

            net_forward(n, x);

            double *outErr = calloc(n->outputs, sizeof(double));
            for (int k = 0; k < n->outputs; k++){
                double diff = y[k] - n->o[k];
                outErr[k] = diff * dsigmoid(n->o[k]);
                err += diff * diff;
            }

            double *hidErr = calloc(n->hidden, sizeof(double));
            for (int j = 0; j < n->hidden; j++){
                double sum = 0;
                for (int k = 0; k < n->outputs; k++){
                    sum += outErr[k] * n->ho[j][k];
                }
                hidErr[j] = sum * dsigmoid(n->h[j]);
            }

            for (int j = 0; j < n->hidden; j++){
                for (int k = 0; k < n->outputs; k++){
                    n->ho[j][k] += n->lr * outErr[k] * n->h[j];
                }
            }

            for (int i = 0; i < n->inputs; i++){
                for (int j = 0; j < n->hidden; j++){
                    n->ih[i][j] += n->lr * hidErr[j] * x[i];
                }
            }
            free(outErr);
            free(hidErr);
        }

        if (e % 1000 == 0){
            printf("epoch %d  err: %.6f\n", e, err / samples);
        }
    }
}

void net_show(Net *n, double *input){
    net_forward(n, input);
    printf("[%.0f %.0f] -> %.3f\n", input[0], input[1], n->o[0]);
}
