#pragma once

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/Pass.h.inc"

} // namespace mlir

