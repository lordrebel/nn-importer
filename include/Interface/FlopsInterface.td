#ifndef INCLUDE_INTERFACE_FLOPSINTERFACE_TD
#define INCLUDE_INTERFACE_FLOPSINTERFACE_TD
include "mlir/IR/OpBase.td"

def FlopsInterface : OpInterface<"FlopsInterface"> {
  let description = [{
  }];
  let cppNamespace = "::im";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getFLOPs",
        /*args=*/(ins)
      >,
  ];
}
#endif // INCLUDE_INTERFACE_FLOPSINTERFACE_TD
