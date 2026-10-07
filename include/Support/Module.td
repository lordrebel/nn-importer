
#ifndef Module_Attr_h
#define Module_Attr_h

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinDialect.td"

// A string attribute whose value are one of the values in `cases`.

def Module_State: I32EnumAttr<"State", "module state",[
      I32EnumAttrCase<"FRONT_F32", 0>,
      I32EnumAttrCase<"FRONT_CALIBRATED", 1>,
      I32EnumAttrCase<"FRONT_QUANTIZED", 2>,
      I32EnumAttrCase<"TOSA_F32", 1000>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::im::module";
}

def Module_Chip: I32EnumAttr<"Chip", "module chip",[
      I32EnumAttrCase<"ALL", 0>,
     
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::im::module";
}

def Module_Mode: I32EnumAttr<"Mode", "module mode",[
      I32EnumAttrCase<"INT8",  0>,
      I32EnumAttrCase<"UINT8", 1>,
      I32EnumAttrCase<"INT4",  2>,
      I32EnumAttrCase<"BF16",  3>,
      I32EnumAttrCase<"F16",   4>,
      I32EnumAttrCase<"F32",   5>,
      I32EnumAttrCase<"W8F16", 6>,
      I32EnumAttrCase<"W8BF16",7>,
      I32EnumAttrCase<"W4F16", 8>,
      I32EnumAttrCase<"W4BF16",9>,
      I32EnumAttrCase<"F8",10>,
      I32EnumAttrCase<"F8E4M3",11>,
      I32EnumAttrCase<"F8E5M2",12>,
      I32EnumAttrCase<"W4F8E4M3",13>,
      I32EnumAttrCase<"W4F8E5M2",14>,
      I32EnumAttrCase<"W4INT8",15>,
      I32EnumAttrCase<"INT8F16DYN", 16>,
      I32EnumAttrCase<"INT8BF16DYN", 17>,
      I32EnumAttrCase<"INT4F16DYN", 18>,
      I32EnumAttrCase<"INT4BF16DYN", 19>,
      I32EnumAttrCase<"F8E4M3F16DYN", 20>,
      I32EnumAttrCase<"F8E4M3BF16DYN", 21>,
      I32EnumAttrCase<"F4F16DYN", 22>,
      I32EnumAttrCase<"F4BF16DYN", 23>,
      I32EnumAttrCase<"MXF4F16DYN", 24>,
      I32EnumAttrCase<"MXF4BF16DYN", 25>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::im::module";
}

def Module_Platform: I32EnumAttr<"Platform", "module platform",[
      I32EnumAttrCase<"ONNX",    0>,
      I32EnumAttrCase<"TORCH",   1>,
      I32EnumAttrCase<"TFLITE",  2>,
      I32EnumAttrCase<"CAFFE",   3>,
      I32EnumAttrCase<"TPULANG", 4>,
      I32EnumAttrCase<"FX", 5>,
      I32EnumAttrCase<"LLM", 6>,
      I32EnumAttrCase<"LLM_QUANTIZED", 7>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::im::module";
}

def Module_AddrMode: I32EnumAttr<"AddrMode", "address mode",[
      I32EnumAttrCase<"BASIC",  0, "basic">,
      I32EnumAttrCase<"IO_ALONE", 1, "io_alone">,
      I32EnumAttrCase<"IO_TAG", 2, "io_tag">,
      I32EnumAttrCase<"IO_TAG_FUSE", 3, "io_tag_fuse">,
      I32EnumAttrCase<"IO_RELOC", 4, "io_reloc">,
      I32EnumAttrCase<"IN_REUSE", 5, "in_reuse">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::im::module";
}

def Module_FrontRunMode: I32EnumAttr<"FrontRunMode", "front run mode",[
      I32EnumAttrCase<"STATIC",  0>,
      I32EnumAttrCase<"DYNAMIC", 1>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::im::module";
}

#endif // Module_Attr_h
