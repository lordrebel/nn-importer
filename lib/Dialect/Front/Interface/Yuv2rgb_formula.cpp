

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::Yuv2rgbFormulaOp::getFLOPs() { return 0; }

LogicalResult front::Yuv2rgbFormulaOp::init(InferenceParameter &p) {
  return success();
}
void front::Yuv2rgbFormulaOp::deinit(InferenceParameter &p) {}

LogicalResult front::Yuv2rgbFormulaOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
}

void front::Yuv2rgbFormulaOp::shape_inference() {
  auto YUV_shape = module::getShape(getYUV());
  auto out_shape = llvm::SmallVector<int64_t>(YUV_shape);
  assert(out_shape.size() >= 2);
  out_shape.insert(out_shape.end() - 2, 3);
  out_shape[out_shape.size() - 2] = out_shape[out_shape.size() - 2] / 3 * 2;
  auto out = getOutput();
  module::setShapeOrVerify(out, out_shape);
}
