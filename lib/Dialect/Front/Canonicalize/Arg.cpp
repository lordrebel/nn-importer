#include "Support/OpRewriterPatternEx.h"

#define DEBUG_TYPE "Arg_canonicalize"

using namespace im::front;
using namespace im::trait;

struct FrontTransposeArg : public OpRewriterPatternEx<ArgOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  FrontTransposeArg(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ArgOp>(context, "FrontTransposeArg") {}

  LogicalResult matchAndRewriteImpl(ArgOp op,
                                    PatternRewriter &rewriter) const override {

    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->hasOneUse() || !isa<PermuteOp>(formerOp)) {
      return failure();
    }
    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getIndices());
    auto permuteOp = cast<PermuteOp>(formerOp);
    auto old_axis = op.getAxis();
    auto permute_order = module::getI64Array(permuteOp.getOrder());
    auto permute_order_len = permute_order->size();
    int order_mask[permute_order_len - 1];
    auto Keepdim = op.getKeepdims();
    memset(order_mask, 0, sizeof(int) * (permute_order_len - 1));
    int order_dim = 0;
    for (int i = 0; i < permute_order_len; i++) {
      if (i == old_axis)
        continue;
      order_mask[order_dim++] = permute_order->at(i);
    }
    for (int i = 0; i < permute_order_len - 2; i++) {
      if (order_mask[i] < order_mask[i + 1])
        continue;
      return failure();
    }
    auto arg_axis = permute_order->at(old_axis);
    op->setAttr("axis", rewriter.getI64IntegerAttr(arg_axis));
    op->setOperand(0, permuteOp.getInput());
    std::vector<int64_t> out_shape(output_shape.size(), 0);
    int out_dim_count = 0;
    for (int i = 0; i < input_shape.size(); i++) {
      if (i == old_axis) {
        if (!Keepdim)
          continue;
        out_shape[out_dim_count] = 1;
        out_dim_count += 1;
      } else {
        out_shape[out_dim_count] = input_shape[i];
        out_dim_count += 1;
      }
    }
    // reshape of arg.indices
    auto out_indices_type = module::getStorageType(op.getIndices());
    auto new_indices_type = RankedTensorType::get(out_shape, out_indices_type);
    std::string out_indices_name =
        module::getName(op.getIndices()).str() + "_r_Reshape";
    auto indices_loc = NameLoc::get(rewriter.getStringAttr(out_indices_name));
    rewriter.setInsertionPointAfter(op);
    auto rs_indices_op = rewriter.create<ReshapeOp>(
        indices_loc, new_indices_type, ValueRange{op.getIndices()});
    op.getIndices().replaceAllUsesExcept(rs_indices_op.getOutput(),
                                         rs_indices_op);
    // reshape of arg.values
    auto values_flag = mlir::isa<NoneType>(op.getValues().getType());
    if (!values_flag) {
      auto out_values_type = module::getStorageType(op.getValues());
      auto new_values_type = RankedTensorType::get(out_shape, out_values_type);
      std::string out_values_name =
          module::getName(op.getValues()).str() + "_r_Reshape";
      auto values_loc = NameLoc::get(rewriter.getStringAttr(out_values_name));
      rewriter.setInsertionPointAfter(op);
      auto rs_values_op = rewriter.create<ReshapeOp>(
          values_loc, new_values_type, ValueRange{op.getValues()});
      op.getValues().replaceAllUsesExcept(rs_values_op.getOutput(),
                                          rs_values_op);
    }
    if (permuteOp.getOutput().getUsers().empty()) {
      rewriter.eraseOp(permuteOp);
    }
    return success();
  }
};

void ArgOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<FrontTransposeArg>(context, "FrontTransposeArg");
}
