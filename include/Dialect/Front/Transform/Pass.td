#ifndef FRONT_TRANSFORM_PASS_TD
#define FRONT_TRANSFORM_PASS_TD
include "mlir/Pass/PassBase.td"

def Init : Pass<"init", "ModuleOp"> {
  let summary = "init module";
  let constructor = "createInitPass()";
  let dependentDialects = ["im::front::FrontDialect"];
  let options = [
    Option<"freq", "freq", "uint64_t", /*default=*/"",
           "chip frequance">,
    Option<"weight_in_mem", "weight_in_mem", "bool", /*default=*/"false",
           "whether to save weight in memory instead of hard disk.">,
    Option<"level", "level", "int64_t", /*default=*/"",
           "log level">,
  ];
}

def Deinit : Pass<"deinit", "ModuleOp"> {
  let summary = "deinit module";
  let constructor = "createDeinitPass()";
  let dependentDialects = ["im::front::FrontDialect"];
  let options = [
    Option<"no_save_weight", "no_save_weight", "bool", /*default=*/"false",
           "whether to save weight.">,
  ];
}

def ProcessorAssign : Pass<"processor-assign", "ModuleOp"> {
  let summary = "Assign chip type";
  let constructor = "createProcessorAssignPass()";
  let dependentDialects = ["im::front::FrontDialect"];
  let options = [
    Option<"chip", "chip", "std::string", /*default=*/"",
           "chip: ">,
    Option<"mode", "mode", "std::string", /*default=*/"",
           "default quantization mode: INT8/BF16/F16/F32/F8/F8E4M3/F8E5M2/INT8F16DYN/INT8BF16DYN/INT4F16DYN/INT4BF16DYN/F8E4M3F16DYN/F8E4M3BF16DYN/F4F16DYN/F4BF16DYN/MXF4F16DYN/MXF4BF16DYN">,
    Option<"num_device", "num_device", "int64_t", /*default=*/"1",
           "num of devices to distributed.">,
    Option<"num_core", "num_core", "int64_t", /*default=*/"1",
           "core_num=1: Set how many cores will be used to run model in parallel.">,
    Option<"addr_mode", "addr_mode", "std::string", /*default=*/"\"auto\"", "addr assign mode">,
    Option<"high_precision", "high_precision", "bool", /*default=*/"false", "force some ops goto fp32">,
  ];
}

def ImportCalibrationTable : Pass<"import-calibration-table", "ModuleOp"> {
  let summary = "Import calibration table by tpuc-opt";
  let constructor = "createImportCalibrationTablePass()";
  let options = [
    Option<"tableFile", "file", "std::string", /*default=*/"",
           "calibration table file path">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"true",
           "true for asymmetric quantization, or false for symmetric">,
  ];
  let dependentDialects = ["im::front::FrontDialect", "mlir::quant::QuantizationDialect"];
}

def QDQConvert : Pass<"convert-qdq-to-calibrated-dialect", "ModuleOp"> {
  let summary = "Convert from qdq model to regular quantized model";
  let constructor = "createQDQConvertPass()";
  let dependentDialects = ["im::front::FrontDialect", "mlir::quant::QuantizationDialect"];
}

def ExtraOptimize : Pass<"extra-optimize", "ModuleOp"> {
  let summary = "after top optimize in mlir by tpuc-opt";
  let constructor = "createExtraOptimizePass()";
  let dependentDialects = ["im::front::FrontDialect"];
}

def ShapeInfer : Pass<"shape-infer", "ModuleOp"> {
  let summary = "do shape inference for each op";
  let constructor = "createShapeInferPass()";
  let dependentDialects = ["im::front::FrontDialect"];
}

def Pruning : Pass<"pruning", "ModuleOp"> {
  let summary = "do pruning for matmul op";
  let constructor = "createPruningPass()";
  let dependentDialects = ["im::front::FrontDialect"];
  let options = [
      Option<"config", "config", "std::string", /*default=*/"",
             "path of config_file.">
  ];
}



def AddPostprocess : Pass<"add-postprocess", "ModuleOp"> {
  let summary = "post handle in mlir by importer-opt";
  let constructor = "createAddPostprocessPass()";
  let dependentDialects = ["im::front::FrontDialect"];
  let options = [
      Option<"type", "type", "std::string", /*default=*/"",
             "type of add-postprocess.">
  ];
}

def StructOptimize : Pass<"struct-optimize", "ModuleOp"> {
  let summary = "struct optimization";
  let constructor = "createStructOptimizePass()";
  let dependentDialects = ["im::front::FrontDialect"];
}

#endif // FRONT_TRANSFORM_PASS_TD
