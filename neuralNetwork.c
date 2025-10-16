#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// xor NeuralNetwork

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

double init_weights();
double sigmoid(double x);
double dSigmoid(double x);
void shuffle(int *array, size_t n);
int run();

int main()
{
    srand(time(NULL));
    run();
    system("pause");
    return 0;
}

int run()
{
    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeight[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f}
    };

    double training_outputs[numTrainingSets][numOutputs] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    for (int i = 0; i < numInputs; i++)
        for (int j = 0; j < numHiddenNodes; j++)
            hiddenWeights[i][j] = init_weights();

    for (int i = 0; i < numHiddenNodes; i++)
        for (int j = 0; j < numOutputs; j++)
            outputWeight[i][j] = init_weights();

    for (int i = 0; i < numHiddenNodes; i++)
        hiddenLayerBias[i] = init_weights();

    for (int i = 0; i < numOutputs; i++)
        outputLayerBias[i] = init_weights();

    int trainingSetOrder[] = {0,1,2,3};
    int numberOfEpochs = 10000;

    for(int epoch = 0; epoch < numberOfEpochs; epoch++)
    {
        double epochLoss = 0.0;
        shuffle(trainingSetOrder, numTrainingSets);

        for(int x = 0; x < numTrainingSets; x++)
        {
            int i = trainingSetOrder[x];

            for(int j = 0; j < numHiddenNodes; j++)
            {
                double activation = hiddenLayerBias[j];
                for(int k = 0; k < numInputs; k++)
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                hiddenLayer[j] = sigmoid(activation);
            }

            for(int j = 0; j < numOutputs; j++)
            {
                double activation = outputLayerBias[j];
                for(int k = 0; k < numHiddenNodes; k++)
                    activation += hiddenLayer[k] * outputWeight[k][j];
                outputLayer[j] = sigmoid(activation);
            }

            double deltaOutput[numOutputs];
            for(int j = 0; j < numOutputs; j++)
            {
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
                epochLoss += error * error;
            }

            double deltaHidden[numHiddenNodes];
            for(int j = 0; j < numHiddenNodes; j++)
            {
                double error = 0.0f;
                for(int k = 0; k < numOutputs; k++)
                    error += deltaOutput[k] * outputWeight[j][k];
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            for(int j = 0; j < numOutputs; j++)
            {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for(int k = 0; k < numHiddenNodes; k++)
                    outputWeight[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
            }

            for(int j = 0; j < numHiddenNodes; j++)
            {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for(int k = 0; k < numInputs; k++)
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
            }
        }

        if (epoch % 100 == 0)
            printf("Epoch %d loss: %g\n", epoch, epochLoss / numTrainingSets);

        if ((epochLoss / numTrainingSets) < 1e-5)
            break;
    }

    fputs("Final Hidden Weights\n[", stdout);
    for(int j = 0; j < numHiddenNodes; j++)
    {
        fputs("[  ", stdout);
        for(int k = 0; k < numInputs; k++)
            printf("%f ", hiddenWeights[k][j]);
        fputs("] ", stdout);
    }

    fputs("]\nFinal Hidden Biases\n[ ", stdout);
    for(int j = 0; j < numHiddenNodes; j++)
        printf("%f ", hiddenLayerBias[j]);

    fputs("]\nFinal Output Weights\n[", stdout);
    for(int j = 0; j < numOutputs; j++)
    {
        fputs("[  ", stdout);
        for(int k = 0; k < numHiddenNodes; k++)
            printf("%f ", outputWeight[k][j]);
        fputs("] \n", stdout);
    }

    fputs("]\nFinal Output Biases\n[ ", stdout);
    for(int j = 0; j < numOutputs; j++)
        printf("%f ", outputLayerBias[j]);
    fputs("] \n", stdout);

    printf("\n--- TEST FINALE ---\n");
    for (int i = 0; i < numTrainingSets; i++)
    {
        for (int j = 0; j < numHiddenNodes; j++)
        {
            double activation = hiddenLayerBias[j];
            for (int k = 0; k < numInputs; k++)
            {
                activation += training_inputs[i][k] * hiddenWeights[k][j];
                hiddenLayer[j] = sigmoid(activation);
            }
        }

        for (int j = 0; j < numOutputs; j++)
        {
            double activation = outputLayerBias[j];
            for (int k = 0; k < numHiddenNodes; k++)
            {
                activation += hiddenLayer[k] * outputWeight[k][j];
            }
            outputLayer[j] = sigmoid(activation);
        }

        printf("Input: %g, %g → Output previsto: %g\n",training_inputs[i][0], training_inputs[i][1], outputLayer[0]);
    }
    
    return 0;
}

double init_weights()
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x)
{
    return x * (1.0 - x);
}

void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        for (size_t i = n - 1; i > 0; i--)
        {
            size_t j = rand() % (i + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}
