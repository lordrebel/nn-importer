

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::FloorOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::FloorOp::init(InferenceParameter &p) { return success(); }
void front::FloorOp::deinit(InferenceParameter &p) {}

LogicalResult front::FloorOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::floor(val);
  }
  return success();
}

void front::FloorOp::shape_inference() {
  common_shape_inference(getOperation());

  if (module::isShape(getInput())) {
    std::vector<std::vector<int64_t>> input_shapes_v;
    auto input_shape_v = module::getShapeTensorValue(getInput());
    input_shapes_v.push_back(input_shape_v);
    auto out_shape = module::getShape(getOutput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), input_shapes_v, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
