#ifndef __TFL_UTILS_H__
#define __TFL_UTILS_H__

#include <iostream>
#include <string>
#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>
#include "tfl_conv.h"

namespace TFMin {
namespace tflite {

typedef unsigned char uint8;
typedef short int int16;
typedef int int32;
typedef long int int64;

enum PaddingType : uint8 { kNone, kSame, kValid };

struct PaddingValues {
  int16 width;
  int16 height;
  // offset is used for calculating "remaining" padding, for example, `width`
  // is 1 and `width_offset` is 1, so padding_left is 1 while padding_right is
  // 1 + 1 = 2.
  int16 width_offset;
  // Same as width_offset except it's over the height dimension.
  int16 height_offset;
};

class RuntimeShape {
 public:
  // Shapes with dimensions up to 4 are stored directly in the structure, while
  // larger shapes are separately allocated.
  static constexpr int kMaxSmallSize = 4;

  RuntimeShape& operator=(RuntimeShape const&) = delete;

  RuntimeShape() : size_(0) {}

  explicit RuntimeShape(int dimensions_count) : size_(dimensions_count) {
    if (dimensions_count > kMaxSmallSize) {
#ifdef TF_LITE_STATIC_MEMORY
      TFLITE_CHECK(false && "No shape resizing supported on this platform");
#else   // TF_LITE_STATIC_MEMORY
      dims_pointer_ = new int32[dimensions_count];
#endif  // TF_LITE_STATIC_MEMORY
    }
  }

  RuntimeShape(int shape_size, int32 value) : size_(0) {
    Resize(shape_size);
    for (int i = 0; i < shape_size; ++i) {
      SetDim(i, value);
    }
  }

  RuntimeShape(int dimensions_count, const int32* dims_data) : size_(0) {
    ReplaceWith(dimensions_count, dims_data);
  }

  RuntimeShape(const std::initializer_list<int> init_list) : size_(0) {
    BuildFrom(init_list);
  }

  // Avoid using this constructor.  We should be able to delete it when C++17
  // rolls out.
  RuntimeShape(RuntimeShape const& other) : size_(other.DimensionsCount()) {
    if (size_ > kMaxSmallSize) {
      dims_pointer_ = new int32[size_];
    }
    std::memcpy(DimsData(), other.DimsData(), sizeof(int32) * size_);
  }

  bool operator==(const RuntimeShape& comp) const {
    return this->size_ == comp.size_ &&
           std::memcmp(DimsData(), comp.DimsData(), size_ * sizeof(int32)) == 0;
  }

  ~RuntimeShape() {
    if (size_ > kMaxSmallSize) {
#ifdef TF_LITE_STATIC_MEMORY
      TFLITE_CHECK(false && "No shape resizing supported on this platform");
#else   // TF_LITE_STATIC_MEMORY
      delete[] dims_pointer_;
#endif  // TF_LITE_STATIC_MEMORY
    }
  }

  inline int32 DimensionsCount() const { return size_; }
  inline int32 Dims(int i) const {
    //TFLITE_DCHECK_GE(i, 0);
    //TFLITE_DCHECK_LT(i, size_);
    return size_ > kMaxSmallSize ? dims_pointer_[i] : dims_[i];
  }
  inline void SetDim(int i, int32 val) {
    //TFLITE_DCHECK_GE(i, 0);
    //TFLITE_DCHECK_LT(i, size_);
    if (size_ > kMaxSmallSize) {
      dims_pointer_[i] = val;
    } else {
      dims_[i] = val;
    }
  }

  inline int32* DimsData() {
    return size_ > kMaxSmallSize ? dims_pointer_ : dims_;
  }
  inline const int32* DimsData() const {
    return size_ > kMaxSmallSize ? dims_pointer_ : dims_;
  }
  // The caller must ensure that the shape is no bigger than 4-D.
  inline const int32* DimsDataUpTo4D() const { return dims_; }

  inline void Resize(int dimensions_count) {
    if (size_ > kMaxSmallSize) {
#ifdef TF_LITE_STATIC_MEMORY
      TFLITE_CHECK(false && "No shape resizing supported on this platform");
#else   // TF_LITE_STATIC_MEMORY
      delete[] dims_pointer_;
#endif  // TF_LITE_STATIC_MEMORY
    }
    size_ = dimensions_count;
    if (dimensions_count > kMaxSmallSize) {
#ifdef TF_LITE_STATIC_MEMORY
      TFLITE_CHECK(false && "No shape resizing supported on this platform");
#else   // TF_LITE_STATIC_MEMORY
      dims_pointer_ = new int32[dimensions_count];
#endif  // TF_LITE_STATIC_MEMORY
    }
  }

  inline void ReplaceWith(int dimensions_count, const int32* dims_data) {
    Resize(dimensions_count);
    int32* dst_dims = DimsData();
    std::memcpy(dst_dims, dims_data, dimensions_count * sizeof(int32));
  }

  template <typename T>
  inline void BuildFrom(const T& src_iterable) {
    const int dimensions_count =
        std::distance(src_iterable.begin(), src_iterable.end());
    Resize(dimensions_count);
    int32* data = DimsData();
    for (auto it : src_iterable) {
      *data = it;
      ++data;
    }
  }

  // This will probably be factored out. Old code made substantial use of 4-D
  // shapes, and so this function is used to extend smaller shapes. Note that
  // (a) as Dims<4>-dependent code is eliminated, the reliance on this should be
  // reduced, and (b) some kernels are stricly 4-D, but then the shapes of their
  // inputs should already be 4-D, so this function should not be needed.
  inline static RuntimeShape ExtendedShape(int new_shape_size,
                                           const RuntimeShape& shape) {
    return RuntimeShape(new_shape_size, shape, 1);
  }

  inline void BuildFrom(const std::initializer_list<int> init_list) {
    BuildFrom<const std::initializer_list<int>>(init_list);
  }

  // Returns the total count of elements, that is the size when flattened into a
  // vector.
  inline int FlatSize() const {
    int buffer_size = 1;
    const int* dims_data = reinterpret_cast<const int*>(DimsData());
    for (int i = 0; i < size_; i++) {
      buffer_size *= dims_data[i];
    }
    return buffer_size;
  }

  bool operator!=(const RuntimeShape& comp) const { return !((*this) == comp); }

 private:
  // For use only by ExtendedShape(), written to guarantee (return-value) copy
  // elision in C++17.
  // This creates a shape padded to the desired size with the specified value.
  RuntimeShape(int new_shape_size, const RuntimeShape& shape, int pad_value)
      : size_(0) {
    // If the following check fails, it is likely because a 4D-only kernel is
    // being used with an array of larger dimension count.
    //TFLITE_CHECK_GE(new_shape_size, shape.DimensionsCount());
    Resize(new_shape_size);
    const int size_increase = new_shape_size - shape.DimensionsCount();
    for (int i = 0; i < size_increase; ++i) {
      SetDim(i, pad_value);
    }
    std::memcpy(DimsData() + size_increase, shape.DimsData(),
                sizeof(int32) * shape.DimensionsCount());
  }

  int32 size_;
  union {
    int32 dims_[kMaxSmallSize];
    int32* dims_pointer_;
  };
};

inline int Offset(const RuntimeShape& shape, int i0, int i1, int i2, int i3) {
  //TFLITE_DCHECK_EQ(shape.DimensionsCount(), 4);
  const int* dims_data = reinterpret_cast<const int*>(shape.DimsDataUpTo4D());

  /*assert(i0 >= 0 && "Dimension 0 is negative");
  assert(i1 >= 0 && "Dimension 1 is negative");
  assert(i2 >= 0 && "Dimension 2 is negative");
  assert(i3 >= 0 && "Dimension 3 is negative");

  assert(i0 < dims_data[0] && "Dimension 0 is out of range");
  assert(i1 < dims_data[1] && "Dimension 1 is out of range");
  assert(i2 < dims_data[2] && "Dimension 2 is out of range");
  assert(i3 < dims_data[3] && "Dimension 3 is out of range");*/

  //TFLITE_DCHECK(i0 >= 0 && i0 < dims_data[0]);
  //TFLITE_DCHECK(i1 >= 0 && i1 < dims_data[1]);
  //TFLITE_DCHECK(i2 >= 0 && i2 < dims_data[2]);
  //TFLITE_DCHECK(i3 >= 0 && i3 < dims_data[3]);
  return ((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3;
}

inline const int MatchingDim(const RuntimeShape& shape_a, int dim_a,
                                const RuntimeShape& shape_b, int dim_b) {

    assert(dim_a <= shape_a.DimensionsCount() &&
            "dimension index out of range, when getting matched dimension.");
    assert(dim_b <= shape_b.DimensionsCount() &&
            "dimension index out of range, when getting matched dimension.");

    char errorMsg[256];
    sprintf(errorMsg, "Error. dimension [%d & %d] don't match when getting matched dimension indices [%d & %d].",
            shape_a.Dims(dim_a),
            shape_b.Dims(dim_b),
            dim_a,
            dim_b);
    if (shape_a.Dims(dim_a) != shape_b.Dims(dim_b)) {
        std::cerr << errorMsg << std::endl;
    }
    assert(shape_a.Dims(dim_a) == shape_b.Dims(dim_b) &&
            "Error. dimensions don't match when getting matched dimension.");

    return shape_a.Dims(dim_a);
}

class TFLUtils {
public:

    static inline int computePadding(int stride, int dilation_rate, int in_size,
                              int filter_size, int out_size) {
        int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
        int padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
        return padding > 0 ? padding : 0;
    }

    template <class T, int rank, int storageOrder>
    static tflite::RuntimeShape shapeFromEigen(Eigen::TensorMap<Eigen::Tensor<T, rank, storageOrder>> tensor) {

        auto eDims = tensor.dimensions();
        tflite::int32 *dims = new tflite::int32[eDims.size()];

        for (int d=0; d<eDims.size(); ++d)
            dims[d] = eDims[d];

        return tflite::RuntimeShape(eDims.size(), dims);
    }

    template <class T, int rank, int storageOrder>
    static tflite::RuntimeShape shapeFromEigenPlus1(Eigen::TensorMap<Eigen::Tensor<T, rank, storageOrder>> tensor) {

        auto eDims = tensor.dimensions();
        tflite::int32 *dims = new tflite::int32[eDims.size()+1];

        dims[0] = 1;

        for (int d=0; d<eDims.size(); ++d)
            dims[d+1] = eDims[d];

        return tflite::RuntimeShape(eDims.size()+1, dims);
    }
};

} // end namespace tflite
} // end namespace TFMin

#endif // __TFL_UTILS_H__
