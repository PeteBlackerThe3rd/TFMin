/*
    TFMin Minimal TensorFlow to C++ exporter
    ------------------------------------------


*/
#include <iostream>
#include "test_source_matmul.h"

int main()
{
    // Instantiate mnist inference model object
    MatMulTest test_model;
    
    // Create single threaded executionEigen device
    Eigen::DefaultDevice device;

    bool res = test_model.validate(device);

    std::cout << "Result was" << !res << std::endl;

    // return model inverse validation result (0 == sucess)
    return !res;
}