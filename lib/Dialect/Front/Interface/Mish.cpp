

#include "Support/GenericCpuFunc.h"
#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::MishOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult front::MishOp::init(InferenceParameter &p) { return success(); }
void front::MishOp::deinit(InferenceParameter &p) {}

LogicalResult front::MishOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = my_mish_activate(val);
  }
  return success();
}

void front::MishOp::shape_inference() {
  common_shape_inference(getOperation());
}
