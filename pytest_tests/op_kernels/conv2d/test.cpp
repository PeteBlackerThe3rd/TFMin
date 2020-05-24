/*
    TFMin Minimal TensorFlow to C++ exporter
    ------------------------------------------


*/
#include <iostream>
#include "tmp/test_source_conv2d.h"

int main()
{
    // Instantiate mnist inference model object
    Conv2dTest test_model;
    
    // Create single threaded executionEigen device
    Eigen::DefaultDevice device;

    bool res = test_model.validate(device);

    std::cout << "Result was" << !res << std::endl;

    // return model inverse validation result (0 == sucess)
    return !res;
}