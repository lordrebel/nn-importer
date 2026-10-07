
#ifndef INTERFACE_SHAPEINTERFACE_TD
#define INTERFACE_SHAPEINTERFACE_TD

include "mlir/IR/OpBase.td"

def ShapeInterface : OpInterface<"ShapeInterface"> {
  let description = [{
  }];
  let cppNamespace = "::im";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          if output shape is dynamic, inference output shape by input shape;
          or verify output shape
        }],
        /*retType=*/"void",
        /*methodName=*/"shape_inference",
        /*args=*/(ins)
      >,
  ];
}
#endif // INTERFACE_SHAPEINTERFACE_TD
