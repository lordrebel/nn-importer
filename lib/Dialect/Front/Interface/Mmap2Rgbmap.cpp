

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::Mmap2RgbmapOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::Mmap2RgbmapOp::init(InferenceParameter &p) {
  return success();
}
void front::Mmap2RgbmapOp::deinit(InferenceParameter &p) {}

LogicalResult front::Mmap2RgbmapOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::Mmap2RgbmapOp::shape_inference() {
  auto in_shape = module::getShape(getInput());

  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(in_shape[0]);
  out_shape.push_back(in_shape[1]);
  out_shape.push_back(in_shape[2]);
  out_shape.push_back(in_shape[3] * 6);
  module::setShapeOrVerify(getOutput(), out_shape);
}
