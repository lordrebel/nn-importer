

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t onnx_rounding(double f) {
  if (f > 0) {
    int a = f;
    double b = f - a;
    if (fabs(b - 0.5) < 1e-8) {
      return a + (a & 1);
    } else
      return std::clamp<int64_t>(std::nearbyintf(f), -128, 127);
  } else {
    int a = f;
    double b = a - f;
    if (fabs(b - 0.5) < 1e-8) {
      return a - (a & 1);
    } else
      return std::clamp<int64_t>(std::nearbyintf(f), -128, 127);
  }
}

int64_t front::QuantizeLinearOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::QuantizeLinearOp::init(InferenceParameter &p) {
  return success();
}
void front::QuantizeLinearOp::deinit(InferenceParameter &p) {}

LogicalResult front::QuantizeLinearOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
  auto shape = mlir::cast<RankedTensorType>(getInput().getType()).getShape();
  auto zero_point = module::getI32Array(getYZeroPoint());
  auto raw_zero_point = *zero_point;
  auto scale = module::getF64Array(getYScale());
  auto raw_scale = *scale;
  ASSERT_THIS(raw_scale.size() == raw_zero_point.size() &&
              "zero point & scale size missmatch");
  if (raw_zero_point.size() == 1) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      auto val = p.inputs[0][i];
      p.outputs[0][i] = onnx_rounding(val / raw_scale[0] + raw_zero_point[0]);
    }
  } else {
    ASSERT_THIS(getAxis() == 0 && "Cannot handle axis!=0");
    ASSERT_THIS(raw_scale.size() == shape[getAxis()] &&
                "zero point & input shape missmatch");
    int64_t res = 1;
    for (int i = 1; i < shape.size(); i++)
      res *= shape[i];
#pragma omp parallel for schedule(static, omp_schedule(shape[0]))
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < res; j++) {
        auto val = p.inputs[0][i * res + j];
        p.outputs[0][i * res + j] =
            onnx_rounding(val / raw_scale[i] + raw_zero_point[i]);
      }
    }
  }
  return success();
}

void front::QuantizeLinearOp::shape_inference() {
  common_shape_inference(getOperation());
}
