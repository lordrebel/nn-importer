

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::ViewOp::getFLOPs() { return 0; }

LogicalResult front::ViewOp::init(InferenceParameter &p) { return success(); }
void front::ViewOp::deinit(InferenceParameter &p) {}

LogicalResult front::ViewOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::ViewOp::shape_inference() {
  auto weight = cast<front::WeightOp>(getShape().getDefiningOp());
  auto shape = weight.read<float>();
  std::vector<int64_t> shape_(shape->begin(), shape->end());
  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto out = getOutput();
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(
      builder.getNamedAttr("shape", builder.getI64ArrayAttr(shape_)));
  auto new_op = builder.create<front::ReshapeOp>(
      getLoc(), out.getType(), ArrayRef<Value>{getInput()}, attrs);
  out.replaceAllUsesWith(new_op.getOutput());
  new_op.shape_inference();
}
