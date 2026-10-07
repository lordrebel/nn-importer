

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::SizeOp::getFLOPs() { return 0; }

LogicalResult front::SizeOp::init(InferenceParameter &p) { return success(); }
void front::SizeOp::deinit(InferenceParameter &p) {}

LogicalResult front::SizeOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

// SizeOp is special, will convert to WeightOp
void front::SizeOp::shape_inference() {
  auto shape = module::getShape(getInput());
  std::vector<float> data;
  if (getAxis().has_value()) {
    auto axis = getAxis().value();
    if (axis < 0) {
      axis += shape.size();
    }
    data.push_back(shape[axis]);
  } else {
    for (auto s : shape) {
      data.push_back((float)s);
    }
  }
  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto weight_type =
      RankedTensorType::get({(int64_t)data.size()}, builder.getF32Type());
  auto new_op = front::WeightOp::create(op, "size", data, weight_type);
  getOutput().replaceAllUsesWith(new_op);
}
