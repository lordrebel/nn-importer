#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace im {
namespace trait {

namespace impl {
mlir::LogicalResult verifyTpuTypeRestrictTrait(mlir::Operation *op);
mlir::LogicalResult verifyInOutSameShapeTrait(mlir::Operation *op);
} // namespace impl



template <typename ConcreteType>
class ScalarProducer
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ScalarProducer> {};

template <typename ConcreteType>
class ScalarConsumer
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ScalarConsumer> {};

// If a op has this trait, it means that relu follow this op can be fused to
// this op
template <typename ConcreteType>
class SupportFuseRelu
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportFuseRelu> {};

template <typename ConcreteType>
class SupportPermuteMove
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportPermuteMove> {};

template <typename ConcreteType>
class SupportConstant
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportConstant> {};

} // namespace trait
} // namespace im
