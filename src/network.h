#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    int inputs, hidden, outputs;
    double lr;

    double **ih; // weight input → hidden
    double **ho; // weight hidden → output
    double *h;   // activ hidden
    double *o;   // activ output

    double *bh;  // bias hidden
    double *bo;  // bias output
} Net;

Net *net_create(int in, int hid, int out, double lr);
void net_free(Net *n);
void net_forward(Net *n, double *input);
void net_train(Net *n, double **data_in, double **data_out, int samples, int epochs);
void net_show(Net *n, double *input);

#endif
