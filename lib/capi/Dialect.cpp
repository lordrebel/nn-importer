#include "mlir/CAPI/Registration.h"
#include "Dialect/Front/IR/Front.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Front, front, im::front::FrontDialect)
