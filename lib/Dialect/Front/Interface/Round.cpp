

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::RoundOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::RoundOp::init(InferenceParameter &p) { return success(); }
void front::RoundOp::deinit(InferenceParameter &p) {}
LogicalResult front::RoundOp::inference(InferenceParameter &p) {
  int64_t num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int64_t i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::round(val);
  }
  return success();
}

void front::RoundOp::shape_inference() {
  common_shape_inference(getOperation());
}
