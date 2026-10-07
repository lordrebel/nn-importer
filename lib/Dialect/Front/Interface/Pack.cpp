

#include "Dialect/Front/IR/Front.h"
#include "Support/Module.h"
using namespace im;
using namespace mlir;

int64_t front::PackOp::getFLOPs() { return 0; }

LogicalResult front::PackOp::init(InferenceParameter &p) { return success(); }
void front::PackOp::deinit(InferenceParameter &p) {}

LogicalResult front::PackOp::inference(InferenceParameter &p) {
  auto axis_ = getAxis();
  auto values_count_ = getValuesCount();
  auto op0_shape =
      mlir::cast<RankedTensorType>(getInputs()[0].getType()).getShape();

  int64_t high = 1;
  for (int64_t i = 0; i < axis_; ++i)
    high *= op0_shape[i];
  // Split the elements to high and low parts and view the lower parts as a
  // single one. We can merge those elemnets more efficiently.
  // [a,b,c,d] -> [a*b, c*d] \
  //     ^                    | ---> [a*b, 2, c*d] --> [a, b, 2, c, d]
  // [a,b,c,d] -> [a*b, c*d] /                                ^
  //     ^
  SmallVector<int64_t> tailNum(values_count_);
  for (auto idt : llvm::enumerate(getInputs())) {
    tailNum[idt.index()] =
        mlir::cast<RankedTensorType>(idt.value().getType()).getNumElements() /
        high;
  }
  auto out_p = p.outputs[0];
  for (int64_t i = 0; i < high; ++i) {
    for (auto idt : llvm::enumerate(tailNum)) {
      memcpy(out_p, p.inputs[idt.index()] + i * idt.value(),
             idt.value() * sizeof(float));
      out_p += idt.value();
    }
  }

  return success();
}

void front::PackOp::shape_inference() {}
