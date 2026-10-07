#pragma once

#include "mlir/IR/Dialect.h"
namespace im {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace im
