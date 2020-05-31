#include <stdio.h>
#include <stdlib.h>
#include "squeeze_net_model.h"

int main(int argc, char **argv[]) {

    // allocate tensor arena
    void *tensor_arena = malloc(NAMESPACE_TENSOR_ARENA_SIZE);
    if (tensor_arena == NULL) {
        printf("Failed to allocate tensor arena of %d bytes\n", NAMESPACE_TENSOR_ARENA_SIZE);
        return 1;
    }

    // define input and output tensors
    float input_tensor[150528];
    float output_tensor[1001];

    printf("About to start inference model.\n");
    namespace_model(tensor_arena, input_tensor, output_tensor);
    printf("Completed inference model okay.\n");

    free(tensor_arena);
}