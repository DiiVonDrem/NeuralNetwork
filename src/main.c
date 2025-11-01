#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "json_io.h"

int main(void){
    srand(42);
    int rule = 0;
    double X[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    Net *n = net_create(2, 2, 1, 0.5);

    if(load_network_json("data/trained_network.json", n)){
        printf("Rete caricata da file JSON.\n");
    }
    else{
        double X[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
        double Y[4][1] = {{0}, {1}, {1}, {0}};
        double *in[4], *out[4];
        for (int i = 0; i < 4; i++) {
            in[i] = X[i];
            out[i] = Y[i];
        }

        net_train(n, in, out, 4, 10000);
        save_network_json("data/trained_network.json", n);
    }
    while(1){
        printf("XOR Neural Network\n");
        printf("Press 1 to start ; Press 2 to see weights ; Press 0 to exit \n");
        int choice = 0;
        int input1, input2;
        int inputEff = 0;
        scanf("%d", &choice);
        switch (choice){
        case 1:
            printf("Enter two binary inputs (0 or 1): ");
            scanf("%d %d", &input1, &input2);
            if(input1 == 0 && input2 == 0){
                inputEff = 0;
            } else if(input1 == 0 && input2 == 1){
                inputEff = 1;
            } else if(input1 == 1 && input2 == 0){
                inputEff = 2;
            } else if(input1 == 1 && input2 == 1){
                inputEff = 3;
            } else {
                printf("Invalid inputs. Please enter binary values (0 or 1).\n");
                break;
            }
            switch(inputEff)
            {
            case (0):
                net_show(n, X[0], rule);
                if(answer() >= 0.5){
                    printf("Output: 1\n");
                } else {
                    printf("Output: 0\n");
                }
                break;
            case (1):
                net_show(n, X[1], rule);
                if(answer() >= 0.5){
                    printf("Output: 1\n");
                } else {
                    printf("Output: 0\n");
                }
                break;
            case (2):
                net_show(n, X[2], rule);
                if(answer() >= 0.5){
                    printf("Output: 1\n");
                } else {
                    printf("Output: 0\n");
                }
                break;
            case (3):
                net_show(n, X[3], rule);
                if(answer() >= 0.5){
                    printf("Output: 1\n");
                } else {
                    printf("Output: 0\n");
                }
            }
            break;
        case 0:
            return 0;
        case 2:
            rule = 1;
            printf("Enter two binary inputs (0 or 1): ");
            scanf("%d %d", &input1, &input2);
            if(input1 == 0 && input2 == 0){
                inputEff = 0;
            } else if(input1 == 0 && input2 == 1){
                inputEff = 1;
            } else if(input1 == 1 && input2 == 0){
                inputEff = 2;
            } else if(input1 == 1 && input2 == 1){
                inputEff = 3;
            } else {
                printf("Invalid inputs. Please enter binary values (0 or 1).\n");
                break;
            }
            switch(inputEff)
            {
            case (0):
                net_show(n, X[0], rule);
                if(answer() >= 0.5){
                    printf("Output: 1\n");
                } else {
                    printf("Output: 0\n");
                }
                break;
            case (1):
                net_show(n, X[1], rule);
                if(answer() >= 0.5){
                    printf("Output: 1\n");
                } else {
                    printf("Output: 0\n");
                }
                break;
            case (2):
                net_show(n, X[2], rule);
                if(answer() >= 0.5){
                    printf("Output: 1\n");
                } else {
                    printf("Output: 0\n");
                }
                break;
            case (3):
                net_show(n, X[3], rule);
                if(answer() >= 0.5){
                    printf("Output: 1\n");
                } else {
                    printf("Output: 0\n");
                }
            }
        break;
        default:
            break;
        }
    }
    
}
