#ifndef INCLUDE_TRAIT_TRAITS_TD
#define INCLUDE_TRAIT_TRAITS_TD

include "mlir/IR/OpBase.td"
class NNImporterOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::im::trait";
}

def ScalarProducer : NNImporterOpTrait<"ScalarProducer">;
def ScalarConsumer : NNImporterOpTrait<"ScalarConsumer">;
def SupportPermuteMove : NNImporterOpTrait<"SupportPermuteMove">;
def SupportConstant: NNImporterOpTrait<"SupportConstant">;
def SupportFuseRelu : NNImporterOpTrait<"SupportFuseRelu">;

#endif // INCLUDE_TRAIT_TRAITS_TD
