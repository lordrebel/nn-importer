#pragma once

#include "mlir/IR/OpDefinition.h"

namespace im {
struct InferenceParameter {
  std::vector<float *> inputs;
  std::vector<float *> outputs;
  void *handle = nullptr;
};

} // namespace im

/// Include the ODS generated interface header files.
#include "Interface/InferenceInterface.h.inc"
