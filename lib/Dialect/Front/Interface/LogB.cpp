

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::LogBOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult front::LogBOp::init(InferenceParameter &p) { return success(); }
void front::LogBOp::deinit(InferenceParameter &p) {}

LogicalResult front::LogBOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
  int base = getBase();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::log(val) / std::log(base);
  }
  return success();
}

void front::LogBOp::shape_inference() {
  common_shape_inference(getOperation());
}
