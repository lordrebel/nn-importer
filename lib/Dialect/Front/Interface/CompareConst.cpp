

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::CompareConstOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::CompareConstOp::init(InferenceParameter &p) {
  return success();
}
void front::CompareConstOp::deinit(InferenceParameter &p) {}

LogicalResult front::CompareConstOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  broadcast_shape_inference(getOperation());
  const auto num_element = module::getNumElements(getOutput());
  const float const_val_ = getConstVal().convertToDouble();
  if (!getInversed()) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      p.outputs[0][i] = compare(p.inputs[0][i], const_val_, getMode());
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      p.outputs[0][i] = compare(const_val_, p.inputs[0][i], getMode());
    }
  }
  return success();
}

void front::CompareConstOp::shape_inference() {
  common_shape_inference(getOperation());
  if (module::isShape(getInput())) {
    auto input_shape_v = module::getShapeTensorValue(getInput());
    auto out_shape = module::getShape(getOutput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_shape_v}, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
