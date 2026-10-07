

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::ErfOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult front::ErfOp::init(InferenceParameter &p) { return success(); }
void front::ErfOp::deinit(InferenceParameter &p) {}

LogicalResult front::ErfOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    p.outputs[0][i] = std::erf(p.inputs[0][i]);
  }
  return success();
}

void front::ErfOp::shape_inference() { common_shape_inference(getOperation()); }
