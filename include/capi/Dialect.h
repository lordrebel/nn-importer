
#ifndef NN_IMPORTER_C_DIALECTS_H
#define NN_IMPORTER_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Front, front);

#ifdef __cplusplus
}
#endif

#endif // NN_IMPORTER_C_DIALECTS_H
