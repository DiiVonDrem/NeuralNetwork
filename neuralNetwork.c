#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define numInputs 2
#define numHidden 2
#define numOutputs 1
#define numSets 4
#define lr 0.1
#define epochs 10000

static inline double init_weight();
static inline double sigmoid(double x);
static inline double dSigmoid(double y);
static inline void shuffle(int *a, size_t n);
static inline void mat_vec_mul(const double *m, const double *v, double *o, int r, int c);
static inline void vec_add(double *a, const double *b, int n);
static inline void vec_sigmoid(double *v, int n);

int main() {
    srand(time(NULL));

    const double X[numSets][numInputs] = {{0,0},{1,0},{0,1},{1,1}};
    const double Y[numSets][numOutputs] = {{0},{1},{1},{0}};

    double W1[numHidden * numInputs], B1[numHidden];
    double W2[numOutputs * numHidden], B2[numOutputs];
    for (int i = 0; i < numHidden * numInputs; ++i) W1[i] = init_weight();
    for (int i = 0; i < numOutputs * numHidden; ++i) W2[i] = init_weight();
    for (int i = 0; i < numHidden; ++i) B1[i] = init_weight();
    for (int i = 0; i < numOutputs; ++i) B2[i] = init_weight();

    double H[numHidden], O[numOutputs], dH[numHidden], dO[numOutputs];
    int order[numSets] = {0,1,2,3};

    for (int e = 0; e < epochs; ++e)
    {
        shuffle(order, numSets);
        double loss = 0.0;
        for (int s = 0; s < numSets; ++s)
        {
            int i = order[s];
            mat_vec_mul(W1, X[i], H, numHidden, numInputs);
            vec_add(H, B1, numHidden);
            vec_sigmoid(H, numHidden);
            mat_vec_mul(W2, H, O, numOutputs, numHidden);
            vec_add(O, B2, numOutputs);
            vec_sigmoid(O, numOutputs);
            for (int j = 0; j < numOutputs; ++j)
            {
                double err = Y[i][j] - O[j]; dO[j] = err * dSigmoid(O[j]); loss += err * err;
            }
            for (int j = 0; j < numHidden; ++j)
            {
                double err = 0.0; for (int k = 0; k < numOutputs; ++k) err += dO[k] * W2[k * numHidden + j]; dH[j] = err * dSigmoid(H[j]);
            }
            for (int j = 0; j < numOutputs; ++j)
            {
                B2[j] += dO[j] * lr; for (int k = 0; k < numHidden; ++k) W2[j * numHidden + k] += H[k] * dO[j] * lr;
            }
            for (int j = 0; j < numHidden; ++j)
            {
                B1[j] += dH[j] * lr; for (int k = 0; k < numInputs; ++k) W1[j * numInputs + k] += X[i][k] * dH[j] * lr;
            }
        }
        if (e % 100 == 0)
        {
            printf("Epoch %d | Loss: %.8f\n", e, loss / numSets);
        }
        if (loss / numSets < 1e-5)
        {
            break;
        }
    }

    printf("\n--- TEST FINALE ---\n");
    for (int i = 0; i < numSets; ++i)
    {
        mat_vec_mul(W1, X[i], H, numHidden, numInputs);
        vec_add(H, B1, numHidden);
        vec_sigmoid(H, numHidden);
        mat_vec_mul(W2, H, O, numOutputs, numHidden);
        vec_add(O, B2, numOutputs);
        vec_sigmoid(O, numOutputs);
        printf("Input: %.1f %.1f → Output previsto: %.6f\n", X[i][0], X[i][1], O[0]);
    }

    getchar();
    return 0;
}

static inline double init_weight() 
{ 
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0; 
}

static inline double sigmoid(double x)
{
    if (x < -60.0)
    {
        return 0.0;
    }if (x > 60.0)
    {
     return 1.0;
    }
    return 1.0 / (1.0 + exp(-x)); 
}

static inline double dSigmoid(double y)
{
    return y * (1.0 - y);
}

static inline void shuffle(int *a, size_t n)
{
    for (size_t i = n - 1; i > 0; --i)
    {
        size_t j = rand() % (i + 1);
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }
}
static inline void mat_vec_mul(const double *m, const double *v, double *o, int r, int c)
{
    for (int i = 0; i < r; ++i)
    {
        double s = 0.0;
        for (int j = 0; j < c; ++j)
        {
            s += m[i * c + j] * v[j]; o[i] = s;
        }
    }
}

static inline void vec_add(double *a, const double *b, int n)
{
    for (int i = 0; i < n; ++i)
    {
        a[i] += b[i];
    }
}

static inline void vec_sigmoid(double *v, int n)
{
    for (int i = 0; i < n; ++i)
    {
        v[i] = sigmoid(v[i]);
    }
}