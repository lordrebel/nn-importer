//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::MeanStdScaleOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::MeanStdScaleOp::init(InferenceParameter &p) {
  return success();
}

void front::MeanStdScaleOp::deinit(InferenceParameter &p) {}

LogicalResult front::MeanStdScaleOp::inference(InferenceParameter &p) {
  // top meanstdscale op do not need inference.
  UNREACHABLE_THIS("Not Implemented");
  return failure();
}

void front::MeanStdScaleOp::shape_inference() {
  common_shape_inference(getOperation());
}
