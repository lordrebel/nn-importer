
#include "mlir/Transforms/Passes.h"
#include "Conversion/Pass.h"
#include "Dialect/Front/Transform/Pass.h"
#include "Dialect/Front/IR/Front.h"
#include<mlir/Dialect/Quant/QuantOps.h>
#include<mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include<mlir/Dialect/Func/IR/FuncOps.h>

namespace im {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::tosa::TosaDialect, mlir::func::FuncDialect, im::front::FrontDialect,
              mlir::quant::QuantizationDialect,
              mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
}

void registerAllPasses() {
  mlir::registerCanonicalizer();
  mlir::registerConversionPasses();
  front::registerFrontPasses();
}

} // namespace im
