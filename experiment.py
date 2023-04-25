import torch
import onnx
import onnxruntime
import json
from src.cases_generation.torch_model import TorchModel
from src.cases_generation.model_info_generator import ModelInfoGenerator
from utils.tool import to_numpy
from utils.db_manager import DbManager
from pathlib import Path
from torchsummary import summary
import onnx_tf
import tensorflow as tf
import numpy as np
from utils.tool import diff_test

from onnx_tf.backend import prepare
import onnx_tf
import tensorflow as tf
# import cv2
import logging, os
# from tensorflow.python.ops import *


def __norm_delta(x, y, mode="max"):
    if x is None or y is None:
            return None
    # infinite norm
    if mode=="max":
        return float(np.max(np.abs(x-y)))
    # l2 norm
    elif mode=="l2":
        return float(np.sum((x-y)*(x-y)))
    else:
        return None

if __name__ == '__main__':

    model_info_path = '/data/fyp23-dlltest/Hong/fyp_dl_library_testing/data/dummy_output/000118/models/model.json'
    training_inputs_path = '/data/fyp23-dlltest/Hong/fyp_dl_library_testing/data/dummy_output/000118/dataset/inputs.npz'

    
    # Load the model info
    with open(model_info_path, 'r') as f:
        m_info = json.load(f)

    
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"

    # Generate the model using model_info
    model = TorchModel(m_info)
    model.load_state_dict(torch.load('/data/fyp23-dlltest/Hong/fyp_dl_library_testing/data/dummy_output/000118/models/torch.pt'))
    
    model = model.to(device)


    # # Load input data
    training_inputs = [*np.load(training_inputs_path).values()]
    training_input = torch.from_numpy(training_inputs[0]).to(torch.float32)
    
    # print("Input shape: ", training_inputs[0].shape)
    torch_training_input = training_input.to(device)

    O_torch=list(model(torch_training_input).values())[0].detach().cpu().numpy()
    print(O_torch.flatten()[-5:])

    input_shape = m_info['model_structure']['0']['args']['shape']
    input_shape.insert(0, 1)

    dummy_input = torch.ones(input_shape).to(device)



    torch.onnx.export(
        model, dummy_input, "test_model.onnx",
        export_params=True,
        output_names=["out"],
        opset_version=12
    )

    sess = onnxruntime.InferenceSession("test_model.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    #prepare input
    training_input = np.float32(training_inputs[0])

    # Run inference
    result = sess.run(None, {input_name: training_input})
    O_onnx=result[0]
    print(O_onnx.flatten()[-5:])

    # O_torch=list(model(torch_training_input).values())[0].detach().cpu().numpy()
    # print(O_torch.flatten()[-5:])

    print(__norm_delta(O_onnx, O_torch, mode='max'))

