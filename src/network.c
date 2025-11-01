#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "network.h"
#include "mathutils.h"

double result;

Net *net_create(int in, int hid, int out, double lr) {
    Net *n = malloc(sizeof(Net));
    n->inputs = in;
    n->hidden = hid;
    n->outputs = out;
    n->lr = lr;

    n->ih = mat_new(in, hid, 1);
    n->ho = mat_new(hid, out, 1);
    n->h = calloc(hid, sizeof(double));
    n->o = calloc(out, sizeof(double));

    // Bias
    n->bh = malloc(hid * sizeof(double));
    n->bo = malloc(out * sizeof(double));
    for (int j = 0; j < hid; j++) n->bh[j] = ((double)rand() / RAND_MAX * 2 - 1);
    for (int k = 0; k < out; k++) n->bo[k] = ((double)rand() / RAND_MAX * 2 - 1);

    return n;
}

void net_free(Net *n) {
    mat_free(n->ih, n->inputs);
    mat_free(n->ho, n->hidden);
    free(n->h);
    free(n->o);
    free(n->bh);
    free(n->bo);
    free(n);
}

void net_forward(Net *n, double *input) {
    // Hidden layer
    for (int j = 0; j < n->hidden; j++) {
        double sum = n->bh[j];
        for (int i = 0; i < n->inputs; i++)
            sum += input[i] * n->ih[i][j];
        n->h[j] = sigmoid(sum);
    }

    // Output layer
    for (int k = 0; k < n->outputs; k++) {
        double sum = n->bo[k];
        for (int j = 0; j < n->hidden; j++)
            sum += n->h[j] * n->ho[j][k];
        n->o[k] = sigmoid(sum);
    }
}

void net_train(Net *n, double **data_in, double **data_out, int samples, int epochs) {
    for (int e = 0; e < epochs; e++) {
        double err = 0;

        for (int s = 0; s < samples; s++) {
            double *x = data_in[s];
            double *y = data_out[s];

            net_forward(n, x);

            double *outErr = calloc(n->outputs, sizeof(double));
            for (int k = 0; k < n->outputs; k++) {
                double diff = y[k] - n->o[k];
                outErr[k] = diff * dsigmoid(n->o[k]);
                err += diff * diff;
            }

            double *hidErr = calloc(n->hidden, sizeof(double));
            for (int j = 0; j < n->hidden; j++) {
                double sum = 0;
                for (int k = 0; k < n->outputs; k++)
                    sum += outErr[k] * n->ho[j][k];
                hidErr[j] = sum * dsigmoid(n->h[j]);
            }

            // Aggiornamento pesi output
            for (int j = 0; j < n->hidden; j++)
                for (int k = 0; k < n->outputs; k++)
                    n->ho[j][k] += n->lr * outErr[k] * n->h[j];

            // Aggiornamento bias output
            for (int k = 0; k < n->outputs; k++)
                n->bo[k] += n->lr * outErr[k];

            // Aggiornamento pesi hidden
            for (int i = 0; i < n->inputs; i++)
                for (int j = 0; j < n->hidden; j++)
                    n->ih[i][j] += n->lr * hidErr[j] * x[i];

            // Aggiornamento bias hidden
            for (int j = 0; j < n->hidden; j++)
                n->bh[j] += n->lr * hidErr[j];

            free(outErr);
            free(hidErr);
        }

        if (e % 1000 == 0)
            printf("epoch %d  err: %.6f\n", e, err / samples);
    }
}

void net_show(Net *n, double *input, int rule) {
    net_forward(n, input);
    result = (n->o[0]);
    if (rule == 1){
        printf("[%.0f %.0f] -> %.3f\n", input[0], input[1], n->o[0]);
    }
}

double answer(){
    return result;
}
