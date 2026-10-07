#include "Dialect/Front/IR/Front.h"
#include "Support/Dnnl/Dnnl.h"
#include "Support/Module.h"

int64_t front::RsqrtOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::RsqrtOp::init(InferenceParameter &p) { return success(); }
void front::RsqrtOp::deinit(InferenceParameter &p) {}

LogicalResult front::RsqrtOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
  float eps = 1e-5;
  float molecular = 1.0;
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    auto sqrt = std::sqrt(val + eps);
    p.outputs[0][i] = molecular / sqrt;
  }
  return success();
}

void front::RsqrtOp::shape_inference() {
  common_shape_inference(getOperation());
}
