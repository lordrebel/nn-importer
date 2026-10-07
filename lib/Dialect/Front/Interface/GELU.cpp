

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::GELUOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 5;
}

LogicalResult front::GELUOp::init(InferenceParameter &p) { return success(); }
void front::GELUOp::deinit(InferenceParameter &p) {}

LogicalResult front::GELUOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = 0.5 * val * (1.0 + std::erf(val / std::sqrt(2.0)));
  }
  return success();
}

void front::GELUOp::shape_inference() {
  common_shape_inference(getOperation());
}
