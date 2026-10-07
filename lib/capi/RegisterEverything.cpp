#include "capi/RegisterEverything.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/MLIRContext.h"
#include "Dialect/Front/IR/Front.h"

void mlirRegisterAllDialects(MlirDialectRegistry registry) {
  static_cast<mlir::DialectRegistry *>(registry.ptr)
      ->insert<mlir::func::FuncDialect, im::front::FrontDialect,
               mlir::quant::QuantizationDialect>();
}
