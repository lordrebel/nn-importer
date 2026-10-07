

#pragma once
#include "Support/Dnnl/common.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "llvm/ADT/ArrayRef.h"

using namespace dnnl;
namespace im {

class LRN {
  using tag = memory::format_tag;
  using dt = memory::data_type;

public:
  LRN();

  template <typename T>
  inline LRN &src(T *src, llvm::ArrayRef<int64_t> lhs_shape) {
    auto mds =
        memory::desc(lhs_shape, data_traits<T>::data_type, get_tag(lhs_shape));
    src_mem = memory(mds, eng, src);
    return *this;
  };

  template <typename T>
  inline LRN &dst(T *dst, llvm::ArrayRef<int64_t> ret_shape) {
    auto mds =
        memory::desc(ret_shape, data_traits<T>::data_type, get_tag(ret_shape));
    dst_mem = memory(mds, eng, dst);
    return *this;
  };

  inline LRN &algorithem(algorithm algorithm) {
    algorithm_ = algorithm;
    return *this;
  }
  inline LRN &size(int64_t size) {
    size_ = size;
    return *this;
  }

  inline LRN &param(float alpha, float beta, float bias) {
    alpha_ = alpha;
    beta_ = beta;
    bias_ = bias;
    return *this;
  }

  void setup();
  void run();

private:
  engine eng;
  float alpha_, beta_, bias_;
  algorithm algorithm_;
  int64_t size_;
  stream engine_stream;
  primitive lrn_prim;
  memory src_mem;
  memory dst_mem;
};
} // namespace im
