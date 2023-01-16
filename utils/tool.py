import numpy as np
import onnxruntime
from onnx import numpy_helper
import torch
import collections


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