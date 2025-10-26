#ifndef MATHUTILS_H
#define MATHUTILS_H

double **mat_new(int r, int c, int rnd);
void mat_free(double **m, int r);
double sigmoid(double x);
double dsigmoid(double y);

#endif
