

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::UnpackOp::getFLOPs() { return 0; }

LogicalResult front::UnpackOp::init(InferenceParameter &p) { return success(); }
void front::UnpackOp::deinit(InferenceParameter &p) {}

LogicalResult front::UnpackOp::inference(InferenceParameter &p) {
  auto axis_ = getAxis();
  auto in_shape = module::getShape(getInput());
  auto num_ = in_shape[axis_];

  int64_t high = 1, inner = 1;
  for (int64_t i = 0; i < axis_; ++i)
    high *= in_shape[i];
  for (int64_t i = axis_; i < in_shape.size(); ++i) {
    inner *= in_shape[i];
  }

  // auto out_p = p.outputs[0];
  // for (int64_t i = 0; i < high; ++i) {
  //   for (int64_t j = 0; j < num_; ++j) {
  //     memcpy(out_p, p.inputs[idt.index()] + i * idt.value(),
  //            idt.value() * sizeof(float));
  //     out_p += idt.value();
  //   }
  // }
#pragma omp parallel for schedule(static, omp_schedule(high))
  for (int i = 0; i < high; ++i) {
    int64_t index = i * in_shape[axis_] * inner;
    for (int j = 0; j < num_; ++j) {
      memcpy(p.outputs[j] + i * inner, p.inputs[0] + index + j * inner,
             inner * sizeof(float));
    }
  }

  return success();
}

void front::UnpackOp::shape_inference() {}
