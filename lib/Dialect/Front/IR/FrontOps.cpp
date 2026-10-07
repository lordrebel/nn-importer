
#include "Dialect/Front/IR/Front.h"
#include "Support/Module.h"
using namespace im::front;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "Dialect/Front/IR/FrontDialect.cpp.inc"

void FrontDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
      >();
  addOperations<
#define GET_OP_LIST
#include "Dialect/Front/IR/Front.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Front Operator Definitions.
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES

#define GET_OP_CLASSES
#include "Dialect/Front/IR/Front.cpp.inc"
