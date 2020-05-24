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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV_H_

// #define USE_THREADS

#ifdef USE_THREADS
#include <thread>
#endif
#include "tfl_utils.h"
//#include "tensorflow/lite/kernels/internal/types.h"
//#include "tensorflow/lite/kernels/internal/common.h"

namespace TFMin {
namespace tflite {

struct ConvParams {
  PaddingType padding_type;
  PaddingValues padding_values;
  // TODO(starka): This was just "stride", so check that width+height is OK.
  int16 stride_width;
  int16 stride_height;
  int16 dilation_width_factor;
  int16 dilation_height_factor;
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


#ifdef USE_THREADS
class ConvThread
{
public:
    ConvThread(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data,
                 int out_y_min, int out_y_max) : params(params),
                                                 input_shape(input_shape),
                                                 input_data(input_data),
                                                 filter_shape(filter_shape),
                                                 filter_data(filter_data),
                                                 bias_shape(bias_shape),
                                                 bias_data(bias_data),
                                                 output_shape(output_shape),
                                                 output_data(output_data),
                                                 out_y_min(out_y_min),
                                                 out_y_max(out_y_max) {};

    void operator()()
    {
      const int stride_width = params.stride_width;
      const int stride_height = params.stride_height;
      const int dilation_width_factor = params.dilation_width_factor;
      const int dilation_height_factor = params.dilation_height_factor;
      const int pad_width = params.padding_values.width;
      const int pad_height = params.padding_values.height;
      //const float output_activation_min = params.float_activation_min;
      //const float output_activation_max = params.float_activation_max;
      //TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
      //TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
      //TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

      //std::cout << "Starting TFL conv2d\n";
      //std::cout << "padding = [" << pad_width << ", " << pad_height << "]\n";

      //(void)im2col_data;   // only used in optimized code.
      //(void)im2col_shape;  // only used in optimized code.

      //std::cout << "input [ " << input_shape.Dims(0) << ", " << input_shape.Dims(1) << ", " << input_shape.Dims(2) << ", " << input_shape.Dims(3) << " ]\n";
      //std::cout << "filter [ " << filter_shape.Dims(0) << ", " << filter_shape.Dims(1) << ", " << filter_shape.Dims(2) << ", " << filter_shape.Dims(3) << " ]\n";
      //std::cout << "output [ " << output_shape.Dims(0) << ", " << output_shape.Dims(1) << ", " << output_shape.Dims(2) << ", " << output_shape.Dims(3) << " ]\n";

      const int batches = MatchingDim(input_shape, 0, output_shape, 0);
      const int input_depth = MatchingDim(input_shape, 3, filter_shape, 2);
      const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);

      //std::cout << "batches [" << batches << "] in depth [" << input_depth << "] out depth [" << output_depth << "]\n";


      /*if (bias_data) {
        TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
      }*/
      const int input_height = input_shape.Dims(1);
      const int input_width = input_shape.Dims(2);
      const int filter_height = filter_shape.Dims(0); // <---
      const int filter_width = filter_shape.Dims(1); // <--
      const int output_height = output_shape.Dims(1);
      const int output_width = output_shape.Dims(2);
      for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = out_y_min; out_y < out_y_max; ++out_y) {
          for (int out_x = 0; out_x < output_width; ++out_x) {
            for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
              const int in_x_origin = (out_x * stride_width) - pad_width;
              const int in_y_origin = (out_y * stride_height) - pad_height;
              float total = 0.f;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    const int in_x = in_x_origin + dilation_width_factor * filter_x;
                    const int in_y =
                        in_y_origin + dilation_height_factor * filter_y;
                    // If the location is outside the bounds of the input image,
                    // use zero as a default value.
                    if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                        (in_y < input_height)) {
                      float input_value = input_data[Offset(
                          input_shape, batch, in_y, in_x, in_channel)];
                      float filter_value =
                          filter_data[Offset(filter_shape, filter_y,
                                             filter_x, in_channel, out_channel)];
                          //filter_data[Offset(filter_shape, out_channel, filter_y,
                          //                   filter_x, in_channel)];
                      total += (input_value * filter_value);
                    }
                  }
                }
              }
              //std::cout << "total calculated.\n";
              float bias_value = 0.0f;
              if (bias_data) {
                bias_value = bias_data[out_channel];
              }
              output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
                  total + bias_value;

              //std::cout << "[" << out_x << ", " << out_y << "] (" << (total + bias_value) << ")\n";
            }
          }
        }

        //std::cout << "Conv2d completed" << std::endl;
      }
    }

    const ConvParams& params;
    const RuntimeShape& input_shape;
    const float* input_data;
    const RuntimeShape& filter_shape;
    const float* filter_data;
    const RuntimeShape& bias_shape;
    const float* bias_data;
    const RuntimeShape& output_shape;
    float* output_data;
    int out_y_min;
    int out_y_max;
};

inline void ConvThreaded(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, int threadCount = 4) {

  const int output_height = output_shape.Dims(1);

  std::vector<std::thread> threads;

  std::cout << "Setting up multithreaded conv with " << threadCount << " threads" << std::endl;
  std::cout << "And an input y size of " << output_height << std::endl;

  int out_y_min = 0;
  int y_per_thread = std::ceil(output_height / threadCount);
  for (int t=0; t<threadCount; t++) {
    int out_y_max = ((t+1) * output_height) / threadCount;
    if (out_y_max > output_height)
        out_y_max = output_height-1;

    std::cout << "Spawning thread with range [" << out_y_min << " to " << out_y_max << "]" << std::endl;

    threads.push_back(std::thread( ConvThread(params, input_shape,
                                              input_data, filter_shape,
                                              filter_data, bias_shape,
                                              bias_data, output_shape,
                                              output_data,
                                              out_y_min, out_y_max) ));

    out_y_min = out_y_max;
  }

  for (auto &t : threads)
    t.join();
};
#endif

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data/*, const RuntimeShape& im2col_shape,
                 float* im2col_data*/) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  //const float output_activation_min = params.float_activation_min;
  //const float output_activation_max = params.float_activation_max;
  //TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  //TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  //TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  //std::cout << "Starting TFL conv2d\n";
  //std::cout << "padding = [" << pad_width << ", " << pad_height << "]\n";

  //(void)im2col_data;   // only used in optimized code.
  //(void)im2col_shape;  // only used in optimized code.

  //std::cout << "input [ " << input_shape.Dims(0) << ", " << input_shape.Dims(1) << ", " << input_shape.Dims(2) << ", " << input_shape.Dims(3) << " ]\n";
  //std::cout << "filter [ " << filter_shape.Dims(0) << ", " << filter_shape.Dims(1) << ", " << filter_shape.Dims(2) << ", " << filter_shape.Dims(3) << " ]\n";
  //std::cout << "output [ " << output_shape.Dims(0) << ", " << output_shape.Dims(1) << ", " << output_shape.Dims(2) << ", " << output_shape.Dims(3) << " ]\n";

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 2);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);

  //std::cout << "batches [" << batches << "] in depth [" << input_depth << "] out depth [" << output_depth << "]\n";


  /*if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }*/
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(0); // <---
  const int filter_width = filter_shape.Dims(1); // <--
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  float filter_value =
                      filter_data[Offset(filter_shape, filter_y,
                                         filter_x, in_channel, out_channel)];
                      //filter_data[Offset(filter_shape, out_channel, filter_y,
                      //                   filter_x, in_channel)];
                  total += (input_value * filter_value);
                }
              }
            }
          }
          //std::cout << "total calculated.\n";
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[out_channel];
          }
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              total + bias_value;

          //std::cout << "[" << out_x << ", " << out_y << "] (" << (total + bias_value) << ")\n";
        }
      }
    }

    //std::cout << "Conv2d completed" << std::endl;
  }
};

inline int ConvOverlap(const ConvParams& params, const RuntimeShape& input_shape,
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


  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 2);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(0); // <--- modified
  const int filter_width = filter_shape.Dims(1); // <-- modified
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  long inputBufSize = input_shape.Dims(1) * input_shape.Dims(2) * input_shape.Dims(3);
  long outputBufSize = output_shape.Dims(1) * output_shape.Dims(2) * output_shape.Dims(3);
  long iterationCount = batches * output_height * output_width * output_depth;
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

  long iterationStep = output_width * output_depth;
  //long inputOffsetStep = Offset(input_shape, 0, 2 * stride_height, (output_width-1) * stride_width, 0) - Offset(input_shape, 0, 1 * stride_height, (output_width-1) * stride_width, 0);
  long inputOffsetStep = Offset(input_shape, 0, 2 * stride_height, 0, 0) - Offset(input_shape, 0, 1 * stride_height, 0, 0);

  long testInputOffsetStep = stride_height*input_width*input_depth;

  std::cerr << "Calculated inputOffsetStep " << inputOffsetStep << std::endl;
  std::cerr << "Test inputOffsetStep " << testInputOffsetStep << std::endl;

  float slope = (stride_height*input_width*input_depth) / (float)(output_width * output_depth);

  long padding_offset = Offset(input_shape,
                               0,
                               2 * stride_height - pad_height,
                               ((output_width-1) * stride_width) - pad_width,
                               0) - ((slope * (iterationStep*3))-1);

  long testPaddingOffset = (output_width*stride_width - pad_height*input_width - stride_height*input_width - stride_width - pad_width) * input_depth + 1;
  //testPaddingOffset -= (3*stride_height*input_width*input_depth);
  //testPaddingOffset -= ((slope * (iterationStep*3))-1);

  std::cerr << "Offset " << padding_offset << std::endl;
  std::cerr << "test Offset " << testPaddingOffset << std::endl;

  std::cerr << "iteration step " << iterationStep << "itCount " << iterationCount;

  int maxOffset = 0;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          Eigen::Index r = iteration*iterationScale;

          int minReadOffset = -1;

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int readOffset = Offset(input_shape, batch, in_y,
                                          in_x, in_channel);
                  if (minReadOffset == -1 || readOffset < minReadOffset)
                    minReadOffset = readOffset;

                  if (inTrace != nullptr) {
                        //Eigen::Index r = iteration*iterationScale;
                        Eigen::Index c = readOffset*inputBufScale;
                        (*inTrace)(r,c) = 1;
                    }
                }
              }
            }
          }

          long estMinReadOffset = std::max(((long)(iteration * slope) + padding_offset), 0L);
          estMinReadOffset = std::min(estMinReadOffset, inputBufSize);
          estMinReadOffset = std::max(estMinReadOffset, 0L);

          // I think the above can be simplified to the expression
          int altMinReadOffset = Offset(input_shape,
                                        batch,
                                        std::max(in_x_origin,0),
                                        std::max(in_y_origin,0),
                                        0);

          int writeOffset = Offset(output_shape, batch, out_y, out_x,
                                   out_channel);

          if (inMin != nullptr)
            (*inMin)(r) = minReadOffset*inputBufScale;
          if (outMax != nullptr)
            (*outMax)(r) = writeOffset*outputBufScale;
          if (inMinEst != nullptr)
            (*inMinEst)(r) = estMinReadOffset*inputBufScale;

          if (outTrace != nullptr) {
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

  return maxOffset;
};

}  // namespace tflite
}  // namespace TFMin


#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV_H_
