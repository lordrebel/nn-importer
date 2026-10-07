#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "Dialect/Front/IR/Front.h"

namespace im::front {
using namespace mlir;
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInitPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createDeinitPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createProcessorAssignPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createImportCalibrationTablePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createQDQConvertPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExtraOptimizePass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAddPostprocessPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createShapeInferPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createPruningPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createStructOptimizePass();

void WeightFolder(mlir::Operation *op);

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Dialect/Front/Transform/Pass.h.inc"
}
