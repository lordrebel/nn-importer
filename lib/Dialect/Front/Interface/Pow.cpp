

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::PowOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult front::PowOp::init(InferenceParameter &p) { return success(); }
void front::PowOp::deinit(InferenceParameter &p) {}

LogicalResult front::PowOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
  auto ex = getExponent().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::pow(val, ex);
  }
  return success();
}

void front::PowOp::shape_inference() {
  common_shape_inference(getOperation());

  auto out_shape = module::getShape(getOutput());
  if (module::isShape(getInput())) {
    auto input_v = module::getShapeTensorValue(getInput());
    auto output_v =
        module::commonShapeValInfer(getOperation(), {input_v}, out_shape);
    module::bindShapeTensorValue(getOutput(), output_v);
  }
}
