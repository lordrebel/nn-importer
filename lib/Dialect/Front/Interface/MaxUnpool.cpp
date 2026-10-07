

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::MaxUnpoolOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::MaxUnpoolOp::init(InferenceParameter &p) {
  return success();
}
void front::MaxUnpoolOp::deinit(InferenceParameter &p) {}

LogicalResult front::MaxUnpoolOp::inference(InferenceParameter &p) {
  int64_t N, C, H, W;
  int64_t ON, OC, OH, OW;
  module::getNCHW(getInput(), N, C, H, W);
  module::getNCHW(getOutput(), ON, OC, OH, OW);
  auto scale_h_ = getScaleH();
  auto scale_w_ = getScaleW();
  auto num_elem = module::getNumElements(getOutput());

  int64_t NC = N * C;
  std::fill_n(p.outputs[0], num_elem, 0.0f);
#pragma omp parallel for schedule(static, omp_schedule(NC))
  for (int idx = 0; idx < NC; ++idx) {
    auto input_data = p.inputs[0] + idx * H * W;
    auto mask_data = p.inputs[1] + idx * H * W;
    auto output_data = p.outputs[0] + idx * OH * OW;
    for (int i = 0; i < H * W; ++i) {
      int offset = static_cast<int>(mask_data[i]);
      if (offset >= H * W * scale_h_ * scale_w_) {
        llvm_unreachable("out of range");
      }
      output_data[offset] = input_data[i];
    }
  }
  return success();
}

void front::MaxUnpoolOp::shape_inference() {
  int64_t N, C, H, W;
  module::getNCHW(getInput(), N, C, H, W);
  auto scale_h_ = getScaleH();
  auto scale_w_ = getScaleW();
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(N);
  out_shape.push_back(C);
  out_shape.push_back(scale_h_ * H);
  out_shape.push_back(scale_w_ * W);
  module::setShapeOrVerify(getOutput(), out_shape);
}
