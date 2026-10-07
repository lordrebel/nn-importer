

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::TanOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult front::TanOp::init(InferenceParameter &p) { return success(); }
void front::TanOp::deinit(InferenceParameter &p) {}

LogicalResult front::TanOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::tan(val);
  }
  return success();
}

void front::TanOp::shape_inference() { common_shape_inference(getOperation()); }
