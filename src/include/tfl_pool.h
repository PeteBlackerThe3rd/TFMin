/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef __TFL_POOL_H__
#define __TFL_POOL_H__

//#include "tensorflow/lite/kernels/internal/common.h"
//#include "tensorflow/lite/kernels/internal/quantization_util.h"
//#include "tensorflow/lite/kernels/internal/round.h"
//#include "tensorflow/lite/kernels/internal/types.h"

#include "tfl_utils.h"

namespace TFMin {
namespace tflite {

struct PoolParams {
  //FusedActivationFunctionType activation;
  PaddingType padding_type;
  PaddingValues padding_values;
  int stride_height;
  int stride_width;
  int filter_height;
  int filter_width;
  // uint8, etc, activation params.
  int32 quantized_activation_min;
  int32 quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

inline int PoolOverlap(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const float* input_data,
                        const RuntimeShape& output_shape,
                        float* output_data) {
  /*TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GE(params.quantized_activation_min, 0);
  TFLITE_DCHECK_LE(params.quantized_activation_max, 255);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);*/
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int maxOffset = 0;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          //uint8 max = 0;

          int minReadOffset = -1;

          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              int readOffset = Offset(input_shape, batch, in_y, in_x, channel);
                  if (minReadOffset == -1 || readOffset < minReadOffset)
                    minReadOffset = readOffset;
            }
          }
          int writeOffset = Offset(output_shape, batch, out_y, out_x, channel);

          minReadOffset *= sizeof(float);
          writeOffset *= sizeof(float);
          int offset = writeOffset - minReadOffset;
          if (offset > maxOffset)
            maxOffset = offset;
        }
      }
    }
  }

  return maxOffset;
}

}  // namespace tflite
}  // namespace TFMin

#endif  // __TFL_POOL_H__
