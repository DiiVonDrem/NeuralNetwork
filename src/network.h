#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    int inputs;
    int hidden;
    int outputs;
    double **ih; // input -> hidden
    double **ho; // hidden -> output
    double *h;   // hidden activations
    double *o;   // output activations
    double lr;   // learning rate
} Net;

Net *net_create(int in, int hid, int out, double lr);
void net_free(Net *n);
void net_forward(Net *n, double *input);
void net_train(Net *n, double **data_in, double **data_out, int samples, int epochs);
void net_show(Net *n, double *input);

#endif
