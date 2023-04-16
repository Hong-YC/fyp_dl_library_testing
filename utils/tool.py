import numpy as np
import onnxruntime
import onnx
from onnx import numpy_helper
import torch
import collections
from src.cases_generation.torch_model import TorchModel
from onnx_tf.backend import prepare
import tensorflow as tf
import onnx_tf
import json


# some helper functions to check weight

# convert tensor to numpy
def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# differential testing
# at this stage assume tensor1 and tensor2 are both numpy array
def diff_test(tensor1,tensor2,thres = 10e-1,norm = "L2"):
    if norm == "L2":
        return np.sum((tensor1-tensor2)**2) <= thres
    if norm == "L1":
        return np.sum(np.abs(tensor1-tensor2)) <= thres
    if norm == "Linf":
        return np.max(np.abs(tensor1-tensor2)) <= thres


def extract_pytorch_weight(model):
    return [to_numpy(para) for para in model.parameters()]


def extract_ort_weight(model):
    weights = model.graph.initializer
    return [numpy_helper.to_array(weights[i]) for i in range(len(weights))]


def ort_inference(onnx_path, input):
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs

#  input a onnx model and a numpy array, output the inference result in tensorflow
def tf_inference(onnx_model, np_arr):
    tf_rep = prepare(onnx_model)
    tf_input = tf.convert_to_tensor(np_arr, dtype = tf.float32)
    return tf_rep.run(tf_input)



# extract intermediate outputs of each layers of PyTorch models
def extract_inter_output_pytorch(model,input):
    # a helper function of forward hook in PyTorch
    def get_activation_py(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    activation = {}
    i=0
    for name, module in model.named_modules():
        if i==0:
            i+=1
            continue
        module.register_forward_hook(get_activation_py(str(name)))
        output = model(input)
        i+=1
    return activation

# extract intermediate outputs of each layers of ONNX models
def extract_inter_output_onnx(model,input):
    modelse=model
    for node in modelse.graph.node:
        for output in node.output:
            modelse.graph.output.extend([onnx.ValueInfoProto(name=output)])

    modelse = modelse.SerializeToString()
    session = onnxruntime.InferenceSession(modelse)
    outputs = [x.name for x in session.get_outputs()]
    input_name = session.get_inputs()[0].name
    outs = session.run(outputs, {input_name:input})
    outs = collections.OrderedDict(zip(outputs, outs))
    return outs

# if tensorflow inference output is different from others, extract tf model's intermediate output (output format is numpy array)
# the input model is PyTorch json file, input is a numpy array
# only call this function when differing in final prediction
def extract_inter_output(pytorch_json_path, input):
    output = {}
    onnx_output_list = []
    tf_output_list = []
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    with open(pytorch_json_path, 'r') as f:
        m_info = json.load(f)
    dic = m_info["model_structure"]

    for key in dic:
        if key == '0':
            continue
        # first generate PyTorch model
        temp_info = m_info.copy()
        model_subset = dict(list(dic.items())[:int(key)+1])
        temp_info['model_structure'] = model_subset
        temp_info['output_id_list'] = list(key)
        py_model = TorchModel(temp_info)
        py_model = py_model.to(device)

        # convert the PyTorch model into ONNX
        input_shape = input.shape
        dummy_input = torch.ones(*input_shape).to(device)
        model_onnx_path = "model.onnx"
        torch.onnx.export(
            py_model, dummy_input, model_onnx_path,
            export_params=True,
            opset_version=11,  # version can be >=7 <=16
            # We define axes as dynamic to allow batch_size > 1
        )
        //output onnx inference
        onnx_output_list.append(ort_inference(model_onnx_path,input))

        # conver the ONNX model into tensorflowRep
        onnx_model = onnx.load("model.onnx")
        # load onnx into tf_rep model
        tf_rep = prepare(onnx_model)
        # tf_rep.export_graph("output/model.pb")
        tf_input = tf.convert_to_tensor(input, dtype = tf.float32)
        tf_output = tf_rep.run(tf_input)
        tf_output_list.append(np.array(tf_output))
    output['onnx_output'] = onnx_output_list
    output['tf_output'] = tf_output_list
    return output

        