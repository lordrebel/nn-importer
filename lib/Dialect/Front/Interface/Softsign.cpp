

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::SoftsignOp::getFLOPs() {
  return module::getNumElements(getInput()) * 3;
}

LogicalResult front::SoftsignOp::init(InferenceParameter &p) {
  return success();
}
void front::SoftsignOp::deinit(InferenceParameter &p) {}

LogicalResult front::SoftsignOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = val / (1 + std::abs(val));
  }
  return success();
}

void front::SoftsignOp::shape_inference() {
  common_shape_inference(getOperation());
}
