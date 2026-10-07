

#pragma once
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;
namespace im {

void post_relu(primitive_attr &attr, bool &do_relu, double &relu_limit);
} // namespace im
