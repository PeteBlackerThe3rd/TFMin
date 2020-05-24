/*
    TFMin v1.0 Minimal TensorFlow to C++ exporter
    ------------------------------------------

    Copyright (C) 2019 Pete Blacker, Surrey Space Centre & Airbus Defence and Space Ltd.
    Pete.Blacker@Surrey.ac.uk
    https://www.surrey.ac.uk/surrey-space-centre/research-groups/on-board-data-handling

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    in the LICENCE file of this software.  If not, see
    <http://www.gnu.org/licenses/>.

    ---------------------------------------------------------------------

    Simple example integrating the c++ MNIST classification inference model
    generated by TFMin with an application
*/
#include <stdlib.h>
#include <stdio.h>

#include "mnist_model.h"

int main()
{
    // Create non-zero input vector
	float input[784];
    for (int i=0; i<784; ++i)
        input[i] = i / 200.0;

	float output[10];

	MNISTModel mnist;

#ifdef EIGEN_USE_THREADS
	Eigen::ThreadPool pool(2);
	Eigen::ThreadPoolDevice device(&pool, 2);
#else
	Eigen::DefaultDevice device;
#endif

    std::cout << "Running model plain." << std::endl;
    mnist.eval(device, input, output);
    std::cout << "Done." << std::endl;

    std::cout << "about to verify model." << std::endl;
	if (mnist.validate(device))
		std::cout << "Verification Passed." << std::endl;
	else
	{
		std::cout << "Error: Verification Failed." << std::endl;
		return 1;
    }

    std::cout << "get time performance of model." << std::endl;
    TFMin::TimingResult times = mnist.timing(device, input, output, false);
    mnist.printTiming(times);
	std::cout << "Completed running model." << std::endl;

    // test completed okay.
	return 0;
}
