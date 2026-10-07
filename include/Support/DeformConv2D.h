#pragma once
#include "Dialect/Front/IR/Front.h"
#include "Support/AttrStruct.h"
#include "Support/Dnnl/Dnnl.h"
#include "Support/Module.h"
#include <vector>

using namespace std;

namespace im {

void parseGatherParam(const deform_conv2d_attr_t &attr,
                      deform_gather_attr_t &gattr);

void parseConvParam(const deform_conv2d_attr_t &attr, conv_attr_t &cattr);

void processDeformGather(InferenceParameter &p,
                         const deform_gather_attr_t &attr, float *data_out,
                         bool top_flag);

void processDeformConv2D(InferenceParameter &p,
                         const deform_conv2d_attr_t &attr);
} // namespace im
