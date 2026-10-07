import importlib
import sys
import numpy as np
import argparse
import os
import shutil
import subprocess
import shlex
import fcntl
import time

g_mlir_module = None

def show_fake_cmd(in_npz: str, model: str, out_npz: str, dump_all_tensors=False, use_cuda=False):
    dump_all = "--dump_all_tensors" if dump_all_tensors else ""
    cuda = "--cuda" if use_cuda else ""
    print("[CMD]: model_runner.py --input {} --model {} --output {} {} {}".format(
        in_npz, model, out_npz, dump_all, cuda))

def mlir_inference(inputs: dict,
                   mlir_file: str,
                   dump_all: bool = True,
                   mute: bool = False,
                   out_fixed: bool = False,
                   use_cuda: bool = False,
                   log_level: str = 'normal') -> dict:
    if mute or log_level == "quiet":
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
    try:
        if not use_cuda:
            return _mlir_inference_by_cpu(inputs, mlir_file, dump_all, out_fixed)
        else:
           raise NotImplementedError("CUDA inference is not implemented yet.")
    finally:
        if mute or log_level == "quiet":
            os.dup2(sys.__stdout__.fileno(), sys.stdout.fileno())
            os.dup2(sys.__stderr__.fileno(), sys.stderr.fileno())


def _mlir_inference_by_cpu(inputs: dict,
                           mlir_file: str,
                           dump_all: bool = True,
                           out_fixed: bool = False) -> dict:
    import pymlir
    pymlir.set_mem_mode("value_mem")
    from utils.mlir_parser import MlirParser
    global g_mlir_module
    if g_mlir_module != None:
        g_mlir_module = None
    g_mlir_module = pymlir.module()
    g_mlir_module.load(mlir_file)
    parser = MlirParser(mlir_file)
    only_one = len(inputs) == 1
    if only_one:
        assert (len(g_mlir_module.input_names) == 1)
    for name in g_mlir_module.input_names:
        if not only_one:
            assert (name in inputs)
            input = inputs[name]
        else:
            input = list(inputs.values())[0]
        if input.dtype == np.int8 or input.dtype == np.uint8:
            g_mlir_module.set_tensor_from_int(name, input.astype(np.float32), input.shape)
        else:
            g_mlir_module.set_tensor(name, input.astype(np.float32), input.shape)
    tensors = dict()
    g_mlir_module.invoke(not out_fixed)
    tensors = g_mlir_module.get_all_tensor()
    if dump_all:
        return tensors
    outputs = dict()
    for name in g_mlir_module.output_names:
        outputs[name] = tensors[name]
        # assume output of op has the same name
        op_type = parser.get_op_type_by_op_name(name)
        if op_type == "tpu.Cast":
            pre_op = parser.get_pre_op_by_op_name(name)[0]
            if pre_op in tensors:
                outputs[pre_op] = tensors[pre_op]
    return outputs


def onnx_inference(inputs: dict, onnx_file: str, dump_all: bool = True) -> dict:
    import onnx
    import onnxruntime
    from onnxruntime_extensions import PyOrtFunction

    def generate_onnx_with_all(onnx_file: str):
        # for dump all activations
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = []
        model = onnx.load(onnx_file)
        no_list = ["Cast", "Constant", "Dropout", "Loop"]

        # tested committed #c3cea486d https://github.com/microsoft/onnxruntime.git
        for x in model.graph.node:
            if x.op_type in no_list:
                continue
            for name in x.output:
                if not name:
                    continue
                intermediate_layer_value_info = onnx.helper.ValueInfoProto()
                intermediate_layer_value_info.name = name
                model.graph.output.append(intermediate_layer_value_info)
                output_keys.append(intermediate_layer_value_info.name + '_' + x.op_type)
        dump_all_tensors_onnx = onnx_file.replace('.onnx', '_all.onnx', 1)
        try:
            onnx.save(model, dump_all_tensors_onnx)
        except Exception as E:
            if "The proto size is larger than the 2 GB limit." in str(E):
                print(
                    "LOG: Try to save {} by using save_as_external_data to save tensors separately from the model file."
                    .format(dump_all_tensors_onnx))
                onnx.save(model,
                          dump_all_tensors_onnx,
                          save_as_external_data=True,
                          location="model_runner_external_data",
                          convert_attribute=True)
            else:
                raise E
        return output_keys, dump_all_tensors_onnx

    output_keys = []
    if dump_all:
        output_keys, onnx_file = generate_onnx_with_all(onnx_file)
    try:
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file,
                                               providers=['CPUExecutionProvider'],
                                               sess_options=so)
    except Exception as E:
        if "is not a registered function/op" in str(E):
            sess = PyOrtFunction.from_model(onnx_file)
            session = sess.ort_session
    inodes = session.get_inputs()
    only_one = len(inputs) == 1
    if only_one:
        assert (len(inodes) == 1)
    data = {}
    for node in inodes:
        name = node.name
        dtype = np.float32
        if node.type == 'tensor(int64)':
            dtype = np.int64
        elif node.type == 'tensor(bool)':
            dtype = np.bool_
        elif node.type == 'tensor(int32)':
            dtype = np.int32
        elif node.type == 'tensor(int16)':
            dtype = np.int16
        if not only_one:
            assert (name in inputs)
            data[name] = inputs[name].astype(dtype)
        else:
            data[name] = list(inputs.values())[0].astype(dtype)
    outs = session.run(None, data)
    outputs = dict()
    if not dump_all:
        onodes = session.get_outputs()
        for node, out in zip(onodes, outs):
            outputs[node.name] = out.astype(np.float32)
        return outputs
    else:
        output_num = len(outs) - len(output_keys)
        outs = outs[output_num:]
        os.remove(onnx_file)
        return dict(filter(lambda x: isinstance(x[1], np.ndarray), zip(output_keys, outs)))


def caffe_inference(inputs: dict, prototxt: str, caffemodel: str, dump_all: bool = True) -> dict:
    import caffe
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    only_one = len(inputs) == 1
    if only_one:
        assert (len(net.inputs) == 1)
    for in_ in net.inputs:
        if not only_one:
            assert (in_ in inputs)
            input = inputs[in_]
        else:
            input = list(inputs.values())[0]
        net.blobs[in_].reshape(*input.shape)
        net.blobs[in_].data[...] = input
    out = net.forward()
    if dump_all:
        blobs_dict = dict(inputs)
        for name, layer in net.layer_dict.items():
            if layer.type == "Split":
                continue
            if layer.type == "Slice":
                continue
            tops = net.top_names[name]
            for t in tops:
                blobs_dict[t] = net.blobs[t].data.copy()
        return blobs_dict
    else:
        return out

def torch_inference(inputs: dict, model: str, dump_all: bool = True) -> dict:
    import torch

    if dump_all:
        from transform.TorchInterpreter import TorchInterpreter
        net = TorchInterpreter(model)
        net.run_model(inputs)
        return net.ref_tensor
    net = torch.jit.load(model, map_location=torch.device('cpu'))
    net.eval()
    in_tensors = [torch.from_numpy(v) for k, v in inputs.items()]
    with torch.no_grad():
        out_tensors = net(*in_tensors)

    names = []
    graph_alive = net.inlined_graph
    for out in graph_alive.outputs():
        if out.node().kind() == 'prim::TupleConstruct' or out.node().kind(
        ) == 'prim::ListConstruct':
            ins = out.node().inputs()
            names.extend([i.debugName() for i in ins])
        else:
            names.append(out.debugName())

    idx = 0

    def torch_outputs(outputs: dict, names: list, tensors):
        nonlocal idx
        if isinstance(tensors, torch.Tensor):
            outputs[names[idx]] = tensors.numpy()
            idx += 1
            return
        if isinstance(tensors, tuple) or isinstance(tensors, list):
            for t in tensors:
                torch_outputs(outputs, names, t)
        else:
            raise RuntimeError("Not Implemented")

    outputs = {}
    torch_outputs(outputs, names, out_tensors)
    return outputs

def free_mlir_module():
    global g_mlir_module
    g_mlir_module = None
