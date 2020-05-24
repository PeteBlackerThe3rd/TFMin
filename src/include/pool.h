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

     TFMin runtime support library.

     2D Pooling operations.
*/
#ifndef __POOL_H__
#define __POOL_H__

#include <iostream>
#include <string>
#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>
#include "tfl_pool.h"
#include "tfl_utils.h"

namespace TFMin {

using namespace tflite;

class PoolTFL : TFLUtils {
public:
    /*template <class T>
    static void conv(Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> input,
                Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> filter,
                Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> output,
                Eigen::PaddingType paddingType,
                int strideWidth, int strideHeight,
                int dilationWidthFactor, int dilationHeightFactor) {

        ConvParams op_params;

        op_params.padding_values.width =
            computePadding(strideWidth, dilationWidthFactor,
                           input.dimensions()[1], filter.dimensions()[1],
                           output.dimensions()[1]);

        op_params.padding_values.height =
            computePadding(strideHeight, dilationHeightFactor,
                           input.dimensions()[0], filter.dimensions()[0],
                           output.dimensions()[0]);

        op_params.stride_width = strideWidth;
        op_params.stride_height = strideHeight;
        op_params.dilation_width_factor = dilationWidthFactor;
        op_params.dilation_height_factor = dilationHeightFactor;
        //op_params.float_activation_min = output_activation_min;
        //op_params.float_activation_max = output_activation_max;

        tflite::Conv(op_params,
                     shapeFromEigenPlus1(input), input.data(),
                     shapeFromEigen(filter), filter.data(),
                     tflite::RuntimeShape({}), nullptr,
                     shapeFromEigenPlus1(output), output.data());
    }*/

    template <class T>
    static int poolOverlap(Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> input,
                           //Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> filter,
                           Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> output,
                           Eigen::PaddingType paddingType,
                           int poolWidth, int poolHeight,
                           int strideWidth, int strideHeight) {

        PoolParams op_params;

        switch (paddingType) {
            case Eigen::PADDING_VALID: op_params.padding_type = kValid; break;
            case Eigen::PADDING_SAME: op_params.padding_type = kSame; break;
            default: op_params.padding_type = kNone;
        }

        op_params.padding_values.width =
            computePadding(strideWidth, 1,
                           input.dimensions()[1], poolWidth,
                           output.dimensions()[1]);

        op_params.padding_values.height =
            computePadding(strideHeight, 1,
                           input.dimensions()[0], poolHeight,
                           output.dimensions()[0]);

        op_params.filter_height = poolHeight;
        op_params.filter_width = poolWidth;
        op_params.stride_width = strideWidth;
        op_params.stride_height = strideHeight;
        //op_params.dilation_width_factor = 1;
        //op_params.dilation_height_factor = 1;
        //op_params.float_activation_min = output_activation_min;
        //op_params.float_activation_max = output_activation_max;

        return tflite::PoolOverlap(op_params,
                     shapeFromEigenPlus1(input), input.data(),
                     shapeFromEigenPlus1(output), output.data());
    }
};

}  // namespace TFMin


#endif  // __POOL_H__