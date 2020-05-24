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

     Depthwise convolution operation.
*/
#ifndef __DEPTHWISE_CONV_H__
#define __DEPTHWISE_CONV_H__

#include <iostream>
#include <string>
#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>
#include "tfl_conv.h"
#include "tfl_depthwiseconv_float.h"
#include "tfl_utils.h"

namespace TFMin {

using namespace tflite;

class DepthwiseConvFloatTFL : TFLUtils {
public:
    template <class T>
    static void depthwiseConv(Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> input,
                Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> filter,
                Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> output,
                int strideWidth, int strideHeight) {

    /*TfLiteContext* context, TfLiteNode* node,
                   TfLiteDepthwiseConvParams* params, OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* output) {
      float output_activation_min, output_activation_max;
      CalculateActivationRange(params->activation, &output_activation_min,
                               &output_activation_max);*/

      DepthwiseParams op_params;
      // Padding type is ignored, but still set.
      op_params.padding_type = PaddingType::kSame;
      op_params.padding_values.width =
            computePadding(strideWidth, 1,
                           input.dimensions()[1], filter.dimensions()[1],
                           output.dimensions()[1]);
      op_params.padding_values.height =
            computePadding(strideHeight, 1,
                           input.dimensions()[0], filter.dimensions()[0],
                           output.dimensions()[0]);
      op_params.stride_width = strideWidth;
      op_params.stride_height = strideHeight;
      op_params.dilation_width_factor = 1;
      op_params.dilation_height_factor = 1;
      op_params.depth_multiplier = filter.dimensions()[3];

      tflite::DepthwiseConv(op_params,
                            shapeFromEigenPlus1(input), input.data(),
                            shapeFromEigen(filter), filter.data(),
                            tflite::RuntimeShape({}), nullptr,
                            shapeFromEigenPlus1(output), output.data());
    }

    template <class T>
    static int depthwiseConvOverlap(Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> input,
                Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> filter,
                Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> output,
                int strideWidth, int strideHeight,
                Eigen::MatrixXi *inTrace = nullptr,
                Eigen::MatrixXi *outTrace = nullptr,
                Eigen::VectorXi *inMin = nullptr,
                Eigen::VectorXi *outMax = nullptr,
                Eigen::VectorXi *inMinEst = nullptr,
                Eigen::VectorXi *outMaxEst = nullptr) {

    /*TfLiteContext* context, TfLiteNode* node,
                   TfLiteDepthwiseConvParams* params, OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* filter,
                   const TfLiteTensor* bias, TfLiteTensor* output) {
      float output_activation_min, output_activation_max;
      CalculateActivationRange(params->activation, &output_activation_min,
                               &output_activation_max);*/

      DepthwiseParams op_params;
      // Padding type is ignored, but still set.
      op_params.padding_type = PaddingType::kSame;

      std::cerr << "Depthwise conv." << std::endl;
      std::cerr << "computing padding with [s " << strideWidth;
      std::cerr << ", 1, " << input.dimensions()[1];
      std::cerr << ", " << filter.dimensions()[1];
      std::cerr << ", " << output.dimensions()[1] << std::endl;

      op_params.padding_values.width =
            computePadding(strideWidth, 1,
                           input.dimensions()[1], filter.dimensions()[1],
                           output.dimensions()[1]);
      op_params.padding_values.height =
            computePadding(strideHeight, 1,
                           input.dimensions()[0], filter.dimensions()[0],
                           output.dimensions()[0]);
      op_params.stride_width = strideWidth;
      op_params.stride_height = strideHeight;
      op_params.dilation_width_factor = 1;
      op_params.dilation_height_factor = 1;
      op_params.depth_multiplier = filter.dimensions()[3];

      return tflite::DepthwiseConvOverlap(op_params,
                            shapeFromEigenPlus1(input), input.data(),
                            shapeFromEigen(filter), filter.data(),
                            tflite::RuntimeShape({}), nullptr,
                            shapeFromEigenPlus1(output), output.data(),
                            inTrace,
                            outTrace,
                            inMin,
                            outMax,
                            inMinEst,
                            outMaxEst);
    }
};

}  // namespace TFMin


#endif  // __DEPTHWISE_CONV_H__