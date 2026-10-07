
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "Dialect/Front/IR/FrontDialect.h.inc"
#include "Interface/FlopsInterface.h"
#include "Interface/InferenceInterface.h"
#include "Interface/ShapeInterface.h"
#include "Support/AttrStruct.h"
#include "Support/TensorFile.h"
#include "Trait/Traits.h"
#include "llvm/ADT/TypeSwitch.h"
#define GET_ATTRDEF_CLASSES
#include "Dialect/Front/IR/FrontAttr.h.inc"
#define GET_OP_CLASSES
#include "Dialect/Front/IR/Front.h.inc"
