# introduction

this project inspired by [tpu-mlir](https://github.com/sophgo/tpu-mlir). Its used to convert onnx/torch/tflite/caffe to mlir ir. Although there are many mlir-style project like
[torch-mlir](), [onnx-mlir](), [stableHLO]() using conversion pass convert ai-framework model to mlir ir.But consider the  project maintenance and manage dependency library versioning, this project utilizes MLIR Python bindings to support the conversion of ONNX, PyTorch, and TFLite models into MLIR IR.
