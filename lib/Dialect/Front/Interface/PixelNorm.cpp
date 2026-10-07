

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::PixelNormOp::getFLOPs() {
  const bool have_weight = !mlir::isa<NoneType>(getWeight().getType());
  const bool have_bias = !mlir::isa<NoneType>(getBias().getType());
  const int64_t num_elem = module::getNumElements(getOutput());
  return num_elem * (10 + have_weight + have_bias);
}

LogicalResult front::PixelNormOp::init(InferenceParameter &p) {
  return success();
}
void front::PixelNormOp::deinit(InferenceParameter &p) {}

LogicalResult front::PixelNormOp::inference(InferenceParameter &p) {
  const float eps = getEps().convertToDouble();
  const auto input_shape = module::getShape(getInput());

  int outer_dim = input_shape[0];
  int channel = input_shape[1];
  int inner_dim = 1;
  for (int i = 2; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  const bool have_weight = !mlir::isa<NoneType>(getWeight().getType());
  const bool have_bias = !mlir::isa<NoneType>(getBias().getType());

  const float *input_data = p.inputs[0];
  const float *weight_data = have_weight ? p.inputs[1] : nullptr;
  const float *bias_data = have_bias ? p.inputs[2] : nullptr;
  float *output_data = p.outputs[0];

  const int num_iter = outer_dim * inner_dim;
#pragma omp parallel for schedule(static, omp_schedule(num_iter))
  for (int i = 0; i < num_iter; ++i) {
    const int p = i / inner_dim;
    const int q = i % inner_dim;
    const float *input_i = input_data + p * channel * inner_dim + q;
    float *output_i = output_data + p * channel * inner_dim + q;
    float mean = 0, rstd = 0;
    for (int j = 0; j < channel; ++j) {
      mean += input_i[j * inner_dim];
    }
    mean /= channel;
    for (int j = 0; j < channel; ++j) {
      const float dij = input_i[j * inner_dim] - mean;
      rstd += dij * dij;
    }
    rstd /= channel;
    rstd += eps;
    rstd = 1.0f / std::sqrt(rstd);
    for (int j = 0; j < channel; ++j) {
      output_i[j * inner_dim] = input_i[j * inner_dim] - mean;
      output_i[j * inner_dim] *= rstd;
      if (have_weight)
        output_i[j * inner_dim] *= weight_data[j];
      if (have_bias)
        output_i[j * inner_dim] += bias_data[j];
    }
  }
  return success();
}

void front::PixelNormOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  std::vector<int64_t> wb_shape(input_shape.size(), 1);
  wb_shape[1] = input_shape[1];
  RankedTensorType newType;
  if (auto weight_op =
          dyn_cast_or_null<WeightOp>(getWeight().getDefiningOp())) {
    newType =
        RankedTensorType::get(wb_shape, module::getElementType(weight_op));
    getWeight().setType(newType);
  }
  if (auto bias_op = dyn_cast_or_null<WeightOp>(getBias().getDefiningOp())) {
    newType = RankedTensorType::get(wb_shape, module::getElementType(bias_op));
    getBias().setType(newType);
  }
  common_shape_inference(getOperation());
}
