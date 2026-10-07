

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::CoshOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult front::CoshOp::init(InferenceParameter &p) { return success(); }
void front::CoshOp::deinit(InferenceParameter &p) {}

LogicalResult front::CoshOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::cosh(val);
  }
  return success();
}

void front::CoshOp::shape_inference() {
  common_shape_inference(getOperation());
}
