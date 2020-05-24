/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_FLOAT_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_FLOAT_H_

//#include "tensorflow/lite/kernels/internal/common.h"
//#include "tensorflow/lite/kernels/internal/compatibility.h"
//#include "tensorflow/lite/kernels/internal/types.h"
//#include <stdlib.h>
#include "tfl_utils.h"

namespace TFMin {
namespace tflite {

struct DepthwiseParams {
  PaddingType padding_type;
  PaddingValues padding_values;
  int16 stride_width;
  int16 stride_height;
  int16 dilation_width_factor;
  int16 dilation_height_factor;
  int16 depth_multiplier;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32 input_offset;
  int32 weights_offset;
  int32 output_offset;
  int32 output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32 quantized_activation_min;
  int32 quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  //TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  //TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  //TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  //std::cout << "input [ " << input_shape.Dims(0) << ", " << input_shape.Dims(1) << ", " << input_shape.Dims(2) << ", " << input_shape.Dims(3) << " ]\n";
  //std::cout << "filter [ " << filter_shape.Dims(0) << ", " << filter_shape.Dims(1) << ", " << filter_shape.Dims(2) << ", " << filter_shape.Dims(3) << " ]\n";
  //std::cout << "output [ " << output_shape.Dims(0) << ", " << output_shape.Dims(1) << ", " << output_shape.Dims(2) << ", " << output_shape.Dims(3) << " ]\n";

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 2, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  //const int filter_height = filter_shape.Dims(1);
  //const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(0); // <--- modified
  const int filter_width = filter_shape.Dims(1); // <-- modified
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  //TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  //TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            float total = 0.f;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value =
                      input_data[Offset(input_shape, b, in_y, in_x, ic)];
                  float filter_value = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, oc)];
                  total += (input_value * filter_value);
                }
              }
            }
            float bias_value = 0.0f;
            if (bias_data) {
              bias_value = bias_data[oc];
            }
            output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                total + bias_value;
            /*output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                ActivationFunctionWithMinMax(total + bias_value,
                                             output_activation_min,
                                             output_activation_max);*/
          }
        }
      }
    }
  }
}

inline int DepthwiseConvOverlap(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* output_data,
    Eigen::MatrixXi *inTrace = nullptr,
    Eigen::MatrixXi *outTrace = nullptr,
    Eigen::VectorXi *inMin = nullptr,
    Eigen::VectorXi *outMax = nullptr,
    Eigen::VectorXi *inMinEst = nullptr,
    Eigen::VectorXi *outMaxEst = nullptr) {

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  //TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  //TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  //TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  //std::cout << "input [ " << input_shape.Dims(0) << ", " << input_shape.Dims(1) << ", " << input_shape.Dims(2) << ", " << input_shape.Dims(3) << " ]\n";
  //std::cout << "filter [ " << filter_shape.Dims(0) << ", " << filter_shape.Dims(1) << ", " << filter_shape.Dims(2) << ", " << filter_shape.Dims(3) << " ]\n";
  //std::cout << "output [ " << output_shape.Dims(0) << ", " << output_shape.Dims(1) << ", " << output_shape.Dims(2) << ", " << output_shape.Dims(3) << " ]\n";

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 2, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  //const int filter_height = filter_shape.Dims(1);
  //const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(0); // <--- modified
  const int filter_width = filter_shape.Dims(1); // <-- modified
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  //TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  //TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

  int maxOffset = 0;
  int count=0;

  long inputBufSize = input_shape.Dims(1) * input_shape.Dims(2) * input_shape.Dims(3);
  long outputBufSize = output_shape.Dims(1) * output_shape.Dims(2) * output_shape.Dims(3);
  long iterationCount = batches * output_height * output_width * input_depth * depth_multiplier;
  long iteration = 0;

  float inputBufScale, outputBufScale, iterationScale;
  if (inTrace != nullptr) {
    inputBufScale = inTrace->cols() / (float)inputBufSize;
    outputBufScale = inTrace->cols() / (float)outputBufSize;
    iterationScale = inTrace->rows() / (float)iterationCount;

    std::cerr << "Setting up trace image input scale [" << inputBufScale << "] iteration scale [" << iterationScale << "]" << std::endl;
    std::cerr << "Input buf " << inputBufSize << " bytes. ";
    std::cerr << "Output buf " << outputBufSize << " bytes. ";
    std::cerr << iterationCount << " iterations." << std::endl;
  }

  float inOutRatioIgnoringPadding = (stride_width * stride_height) / (float)depth_multiplier;

  std::cerr << "input w h c : " << input_width << ", " << input_height << ", " << input_depth << std::endl;
  std::cerr << "kernel w h : " << filter_width << ", " << filter_height << std::endl;
  std::cerr << "padding w h :" << pad_width << ", " << pad_height << std::endl;

  std::cerr << "stride_height " << stride_height << std::endl;
  std::cerr << "stride_width " << stride_width << std::endl;
  std::cerr << "depth_multiplier " << depth_multiplier << std::endl;
  std::cerr << "input_depth " << input_depth << std::endl;
  std::cerr << "Calculated an in out ratio of " << inOutRatioIgnoringPadding << std::endl;

  long iterationStep = output_width * input_depth * depth_multiplier;
  long inputOffsetStep = Offset(input_shape, 0, 2 * stride_height, (output_width-1) * stride_width, input_depth-1) - Offset(input_shape, 0, 1 * stride_height, (output_width-1) * stride_width, input_depth-1);
  float slope = inputOffsetStep / (float)iterationStep;
  std::cerr << "intStep " << iterationStep << std::endl;
  std::cerr << "inputOffsetStep eqn Test " << (stride_height * input_depth * input_width) << std::endl;
  std::cerr << " offsetStep " << inputOffsetStep << std::endl;
  std::cerr << " slope " << slope << std::endl;
  std::cerr << "Slope Eqn Test = " << ((stride_height * input_width) / (float)(output_width * depth_multiplier)) << std::endl;
  std::cerr << "itStep test " << (iterationStep * output_height) << std::endl;

  long padding_offset = Offset(input_shape,
                               0,
                               2 * stride_height - pad_height,
                               ((output_width-1) * stride_width) - pad_width,
                               input_depth-1) - ((slope * (iterationStep*3))-1);

  std::cerr << "Calculated padding offset " << padding_offset << std::endl;

  //long test_padding_offset = (2*stride_height*input_width - pad_height*input_width + output_width*stride_width - stride_width - pad_width + 1)*input_depth - (3*slope*output_width*input_depth*depth_multiplier);
  long test_padding_offset = (/*2*stride_height*input_width*/0 - pad_height*input_width + output_width*stride_width - stride_width - pad_width - 1*stride_height*input_width + 1)*input_depth;
  std::cerr << "Testing padding offset Eqn : " << test_padding_offset << std::endl;

  //padding_offset = inputBufSize - (slope*iterationCount);
  //std::cerr << "Cheating padding offset " << padding_offset << std::endl;

  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            Eigen::Index r = iteration*iterationScale;

            int minReadOffset = -1;

            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                    int readOffset = Offset(input_shape, b, in_y, in_x, ic);

                    if (inTrace != nullptr) {
                        //Eigen::Index r = iteration*iterationScale;
                        Eigen::Index c = readOffset*inputBufScale;
                        (*inTrace)(r,c) = 1;
                    }

                    if (minReadOffset == -1 || readOffset < minReadOffset) {
                        minReadOffset = readOffset;
                        //std::cerr << "min found at [" << filter_x << "," << filter_y << "]\n";
                    }
                }
              }
            }

            //int minReadBase = Offset(input_shape, b, out_y, out_x, ic);
            long minReadBase = iteration * slope; /// inOutRatioIgnoringPadding;
            long estMinReadOffset = std::max((minReadBase + padding_offset), 0L);
            estMinReadOffset = std::min(estMinReadOffset, inputBufSize);
            estMinReadOffset = std::max(estMinReadOffset, 0L);

            //std::cerr << minReadOffset << ", " << estMinReadOffset << std::endl;

            int writeOffset = Offset(output_shape, b, out_y, out_x, oc);

            if (inMin != nullptr)
                (*inMin)(r) = minReadOffset*inputBufScale;
            if (outMax != nullptr)
                (*outMax)(r) = writeOffset*outputBufScale;
            if (inMinEst != nullptr)
                (*inMinEst)(r) = estMinReadOffset*inputBufScale;

            if (outTrace != nullptr) {
                //Eigen::Index r = iteration*iterationScale;
                Eigen::Index c = writeOffset*outputBufScale;
                (*outTrace)(r,c) = 1;
            }

            minReadOffset *= sizeof(float);
            writeOffset *= sizeof(float);

            int offset = writeOffset - minReadOffset;
            if (offset > maxOffset)
                maxOffset = offset;

            ++iteration;
          }
        }
      }
    }
  }

  return maxOffset;
}

}  // end namespace tflite
}  // end namespace TFMin

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEPTHWISECONV_FLOAT_H_
