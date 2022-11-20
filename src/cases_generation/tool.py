import numpy as np
import onnxruntime
from onnx import numpy_helper


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