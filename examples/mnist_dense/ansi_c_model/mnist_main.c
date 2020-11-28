
#include <stdio.h>
#include <stdlib.h>
#include "mnist_model.h"

int main(int argc, char **argv[]) {

    // allocate tensor arena
    void *tensorArena = malloc(MNIST_TENSOR_ARENA_SIZE);
    if (tensorArena == NULL) {
        fprintf(stderr, "Failed to allocate tensor arena of %d bytes\n", MNIST_TENSOR_ARENA_SIZE);
        return 1;
    }

    // input and output tensor arrays
    float input[784], output[10];

    // Populate input data here.
    
    // execute inference model on test
    mnist_model(tensorArena, input, output);
    
    // Do something with the output tensor
    
    // free memory and exit successfully
    free(tensorArena);
    exit(0);
}
    