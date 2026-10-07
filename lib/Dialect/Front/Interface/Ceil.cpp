

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::CeilOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::CeilOp::init(InferenceParameter &p) { return success(); }
void front::CeilOp::deinit(InferenceParameter &p) {}
LogicalResult front::CeilOp::inference(InferenceParameter &p) {
  int64_t num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int64_t i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::ceil(val);
  }
  return success();
}

void front::CeilOp::shape_inference() {
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
