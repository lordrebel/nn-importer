#include "Dialect/Front/IR/Front.h"
#include "Support/Dnnl/Dnnl.h"
#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::Pow3Op::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::Pow3Op::init(InferenceParameter &p) { return success(); }
void front::Pow3Op::deinit(InferenceParameter &p) {}

LogicalResult front::Pow3Op::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    auto ex = p.inputs[1][i];
    p.outputs[0][i] = std::pow(val, ex);
  }
  return success();
}

void front::Pow3Op::shape_inference() {
  common_shape_inference(getOperation());
}
