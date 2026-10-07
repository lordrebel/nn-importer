# introduction

this project inspired by [tpu-mlir](https://github.com/sophgo/tpu-mlir). Its used to convert onnx/torch/tflite/caffe to mlir ir. Although there are many mlir-style project like
[torch-mlir](https://github.com/llvm/torch-mlir), [onnx-mlir](https://github.com/onnx/onnx-mlir), [stableHLO](https://github.com/openxla/stablehlo) using conversion pass convert ai-framework model to mlir ir.But consider the  project maintenance and manage dependency library versioning, this project utilizes MLIR Python bindings to support the conversion of ONNX, PyTorch, and TFLite models into MLIR IR.

## build

for your env maybe need apply patches in [here](./patches/)  

### llvm build  
  
```bash
cd ..
git clone --branch llvmorg-19.1.7 --depth 1  https://gitee.com/mirrors/LLVM.git
cd LLVM&&mkdir -p build && cd build
cmake -G Ninja ../llvm     -DLLVM_ENABLE_PROJECTS="mlir"     -DLLVM_INSTALL_UTILS=ON     -DLLVM_TARGETS_TO_BUILD=""     -DLLVM_ENABLE_ASSERTIONS=ON     -DMLIR_INCLUDE_TESTS=OFF     -DLLVM_INSTALL_GTEST=ON     -DMLIR_ENABLE_BINDINGS_PYTHON=ON     -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../llvm_release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_BINDINGS=ON   -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_PIC=ON  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON -DPython3_EXECUTABLE=$(which python3)

cmake --build . --target install
```
### onednn build

```bash
ONEDNN_VERSION="5aabea153825347afa92a2d9f69dd893246bea45"
git clone https://github.com/oneapi-src/oneDNN.git && \
    cd oneDNN && git checkout ${ONEDNN_VERSION} && \
    mkdir -p build && cd build && \
    cmake -G Ninja .. \
    -DDNNL_CPU_RUNTIME=OMP \
    -DDNNL_BUILD_TESTS=OFF \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX=../../onednn_release && \
    cmake --build . --target install 
```

### build nn-importer

```bash
source env.source && bash build.sh
```

## test

`nn_importer.py` is the entry point. The frontend is selected by the suffix of `--model_def`: `.onnx` for onnx, `.pt` for torchscript, `.prototxt` + `--model_data xx.caffemodel` for caffe, `.mlir` for an already imported mlir.

The examples below use the models shipped in [models/](./models), `yolov5s.onnx` and `yolov5s.pt`.

```bash
source env.source    # exports PYTHONPATH/PATH; activate your venv first if you use one
cd models
```

### onnx model import

```bash
nn_importer.py --model_def ./yolov5s.onnx --model_name yolov5 --input_shapes [[1,3,640,640]] --mean 0.0,0.0,0.0 --scale 0.0039216,0.0039216,0.0039216 --keep_aspect_ratio --pixel_format rgb --mlir yolov5_top.mlir
```

### torch model import

The `.pt` file must be a torchscript model (exported with `torch.jit.trace` or `torch.jit.script`), because the importer loads it with `torch.jit.load`.

```bash
nn_importer.py --model_def ./yolov5s.pt --model_name yolov5 --input_shapes [[1,3,640,640]] --mean 0.0,0.0,0.0 --scale 0.0039216,0.0039216,0.0039216 --keep_aspect_ratio --pixel_format rgb --mlir yolov5_top_torch.mlir
```

Both commands first write the frontend mlir, then optimize it with `importer-opt`:

```
Save mlir file: yolov5_top_origin.mlir
[Running]: importer-opt yolov5_top_origin.mlir --struct-optimize --shape-infer -canonicalize --extra-optimize -o yolov5_top.mlir
[Success]: importer-opt yolov5_top_origin.mlir --struct-optimize --shape-infer -canonicalize --extra-optimize -o yolov5_top.mlir
Mlir file generated:yolov5_top.mlir
```

### output files

| file                                          | description                                                                                        |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `<name>_origin.mlir`                          | mlir right after the frontend conversion                                                           |
| `<name>.mlir`                                 | mlir after `importer-opt`, this is the file given by `--mlir`                                      |
| `<name>_front_f32_all_weight.npz`             | weights referenced by the final mlir                                                               |
| `<name>_opt.onnx`, `<name>_opt.onnx.prototxt` | onnx only: result of onnx-sim/constant folding, the `.onnx` is removed at the end unless `--debug` |

So the two examples above produce `yolov5_top.mlir` (onnx) and `yolov5_top_torch.mlir` (torch), both with `module @yolov5` and `yolov5_front_f32_all_weight.npz`.

### options used above

| option                          | description                                                                        |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| `--model_name`                  | module name inside the mlir (`module @yolov5`), also the prefix of the weight file |
| `--model_def`                   | input model file, its suffix selects the frontend                                  |
| `--input_shapes`                | input shapes, e.g. `[[1,3,640,640]]`, or `[[1,3,224,224],[10]]` for several inputs |
| `--mean`, `--scale`             | per channel preprocess values, emitted into the `front.Input` op                   |
| `--keep_aspect_ratio`           | letterbox resize, the area which is not taken is filled by `--pad_value`           |
| `--pixel_format`                | pixel format of the data sent into the model: `rgb`, `bgr`, `gray`, ...            |
| `--mlir`                        | output mlir file, must end with `.mlir`                                            |
| `--test_input`, `--test_result` | optional, run the source model and the generated mlir and compare them             |
| `--debug`                       | keep all intermediate files                                                        |

Run `nn_importer.py --help` for the full option list.


# TODO

- [ ] change front dialect
- [ ] support calibration
- [ ] create docker env

