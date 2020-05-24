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

    ModelBase - Virtual base class from which all model implementation objects
    are derived from.
*/
#ifndef __TF_MIN_H__
#define __TF_MIN_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <vector>
#include <initializer_list>
#include <limits>
#include <time.h>

#include <unsupported/Eigen/CXX11/Tensor>
#ifdef EIGEN_USE_THREADS
#include <mutex>
#include <unsupported/Eigen/CXX11/ThreadPool>
#endif

// include conv2d and depthwise_conv reference algorithms
#include "conv.h"
#include "depthwise_conv.h"

namespace TFMin {

/** Object to store the timing results of a single operation */
class OperationTime
{
public:
    OperationTime(std::string name, float duration)
    {
        this->name = name;
        this->duration = duration;
    };

    std::string name;
    float duration;
};
typedef std::vector<OperationTime> TimingResult;

/** Object to store the representation of a memory trace event. */
class MemoryTraceEvent {
public:
    MemoryTraceEvent(std::string name, int* addr) {
        this->name = name;
        this->addr = addr;
    }

    volatile int* addr;
    std::string name;
};
typedef std::vector<MemoryTraceEvent> MemoryTraceEvents;

class MemoryTraceArea {
public:
    MemoryTraceArea(int offset,
                     int size,
                     std::string startEvent,
                     std::string endEvent) {
        this->offset = offset;
        this->size = size;
        this->startEvent = startEvent;
        this->endEvent = endEvent;
    }

    int offset, size;
    std::string startEvent, endEvent;
};
typedef std::vector<MemoryTraceArea> MemoryTraceAreas;

class ModelBase
{
public:

    ModelBase()
    {
        // Construct dimensions used to emulate 2D matrix multiplication using tensor contraction
        matMulDims = {Eigen::IndexPair < int > (1, 0)};

        // define number of elements of large tensors to print
        maxPrintElements = 20;
        halfPrintElements = 10;

        modelName = "";
    };

    /** getTime() returns time in seconds since the epoc as a floating point
     * This function is safe on both desktop systems and barebones LEON systems
     */
    float getTime()
    {
        clock_t time = clock();
#ifdef LEON
        return (double)time / 1000000.0;
#else
        return (double)time / CLOCKS_PER_SEC;
#endif
    };

    /// Method to convert the given Dimensions object into a string representation
    template <typename T>
    std::string shapeString(T tensorShape)
    {
        std::string shape = "[ ";
        for (int d=0; d<tensorShape.size(); ++d)
        {
            if (d != 0)
                shape += ", ";

            shape += std::to_string(tensorShape[d]);
        }
        shape += " ]";

        return shape;
    };

    template <class scalar, int rank, int layout>
    bool tensorsApproximatelyEqual(Eigen::TensorMap<Eigen::Tensor<scalar, rank, layout> > A,
                                   Eigen::TensorMap<Eigen::Tensor<scalar, rank, layout> > B,
                                   bool print = false)
    {
        // check tensor shapes match
        for (int d=0; d<A.NumDimensions; ++d)
            if (A.dimensions()[d] != B.dimensions()[d])
            {
                if (print)
                {
                    std::cout << "Tensor shapes do not match: A[ ";
                    std::cout << shapeString(A.dimensions()) << " ] B[";
                    std::cout << shapeString(B.dimensions()) << " ]\n";
                }
                return false;
            }

        // map both tensors to rank 1 to ease comparison of elements
        int flatSize = 1;
        for (int d=0; d<A.NumDimensions; ++d)
            flatSize *= A.dimensions()[d];

        Eigen::TensorMap<Eigen::Tensor<scalar, 1, layout> > AFlat(A.data(), flatSize);
        Eigen::TensorMap<Eigen::Tensor<scalar, 1, layout> > BFlat(B.data(), flatSize);

        // check NAN maps match
        for (int i=0; i<flatSize; ++i)
            if (std::isfinite(AFlat(i)) != std::isfinite(BFlat(i)))
            {
                if (print)
                    std::cout << "Tensor NAN maps do not match.\n";
                return false;
            }

        // check finite values are within given tolerance
        scalar tol = getTolerance<scalar>();

        for (int i=0; i<flatSize; ++i)
            if (fabs(AFlat(i) - BFlat(i)) > tol)
            {
                if (print)
                {
                    std::string location = "]";
                    int rIndex = i;
                    for (int d=0; d<A.NumDimensions; ++d)
                    {
                        int dIndex = rIndex % A.dimensions()[d];
                        rIndex -= dIndex;
                        location = std::to_string(dIndex) + ", " + location;
                    }
                    location = "[ " + location;

                    std::cout << "Tensor difference greater than tolerance found. ";
                    std::cout << fabs(AFlat(i) - BFlat(i)) << " greater than threshold of " << tol << "\n";
                    std::cout << "At location " << location << "\n";
                }
                return false;
            }

        return true;
    };

    /** Overloaded versions of the tensorsApproximatelyEqual method
      *
      * These are used so that it will work with any combination of Tensors or TensorMaps
      */
    template <class scalar, int rank, int layout>
    bool tensorsApproximatelyEqual(Eigen::Tensor<scalar, rank, layout> A,
                                   Eigen::Tensor<scalar, rank, layout> B,
                                   bool print = false)
    {
        return tensorsApproximatelyEqual(Eigen::TensorMap<Eigen::Tensor<scalar, rank, layout> >(A),
                                         Eigen::TensorMap<Eigen::Tensor<scalar, rank, layout> >(B),
                                         print);
    }

    template <class scalar, int rank, int layout>
    bool tensorsApproximatelyEqual(Eigen::Tensor<scalar, rank, layout> A,
                                   Eigen::TensorMap<Eigen::Tensor<scalar, rank, layout> > B,
                                   bool print = false)
    {
        return tensorsApproximatelyEqual(Eigen::TensorMap<Eigen::Tensor<scalar, rank, layout> >(A),
                                         B,
                                         print);
    }

    template <class scalar, int rank, int layout>
    bool tensorsApproximatelyEqual(Eigen::TensorMap<Eigen::Tensor<scalar, rank, layout> > A,
                                   Eigen::Tensor<scalar, rank, layout> B,
                                   bool print = false)
    {
        return tensorsApproximatelyEqual(A,
                                         Eigen::TensorMap<Eigen::Tensor<scalar, rank, layout> >(B),
                                         print);
    }

    template <class T, int rank, int layout>
    void printTensor(Eigen::TensorMap<Eigen::Tensor<T, rank, layout> > tensor)
    {
        std::cout << "[D" << rank << "]------\n";
        int reduceDimension = tensor.NumDimensions-1;

        for (int d=0; d<tensor.dimensions()[reduceDimension]; ++d)
        {
            std::cout << d << "--------\n";
            Eigen::Tensor<T, rank - 1, layout> tensorChip = tensor.chip(d, reduceDimension);
            printTensor(tensorChip);
            if (tensor.dimensions()[reduceDimension] > maxPrintElements && d == halfPrintElements)
            {
                std::cout << ". . . ." << std::endl;
                d = tensor.dimensions()[reduceDimension] - halfPrintElements;
            }
        }
    }

    template <class T, int layout>
    void printTensor(Eigen::TensorMap<Eigen::Tensor<T, 1, layout> > tensor)
    {
        std::cout << "[ ";

        if (tensor.dimensions()[0] > maxPrintElements)
        {
            for (int d0=0; d0<halfPrintElements; ++d0)
                std::cout << tensor(d0) << ", ";
            std::cout << "... ";
            for (int d0=tensor.dimensions()[0]-halfPrintElements; d0<tensor.dimensions()[0]; ++d0)
                std::cout << tensor(d0) << ", ";
        }
        else
        {
            for (int d0=0; d0 < tensor.dimensions()[0]; ++d0)
                std::cout << tensor(d0) << ", ";
        }
        std::cout << " ]\n";
    }

    template <class T, int rank, int layout>
    void printTensor(Eigen::Tensor<T, rank, layout> tensor)
    {
        printTensor(Eigen::TensorMap<Eigen::Tensor<T, rank, layout> >(tensor));
    }

    void printTiming(TimingResult times)
    {
        std::cout << "Operating timing results for " << modelName << std::endl;

        float total = 0.0f;

        for (int t=0; t<times.size(); ++t)
        {
            std::cout << times[t].duration << ", seconds, " << times[t].name << std::endl;
            total += times[t].duration;
        }

        std::cout << total << ", seconds, Total duration" << std::endl;
    }

    // Useful constants used by model operations
    Eigen::array < Eigen::IndexPair < int >, 1 > matMulDims;

private:

    int maxPrintElements;
    int halfPrintElements;

    std::string modelName;

    template <class scalar>
    inline scalar getTolerance()
    {
        // return the machine epsilon for the given type scaled up a bit to account for
        // cumulative errors. TODO will need to formalise this a bit more in the end!
        return std::numeric_limits<scalar>::epsilon() * 200.0;
    }
};

}  // namespace TFMin

#endif // __TF_MIN_H__
