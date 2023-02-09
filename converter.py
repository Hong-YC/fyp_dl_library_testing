import torch
import onnx
import onnxruntime
import numpy as np
from torchsummary import summary
from src.cases_generation.torch_model import TorchModel
import json
import collections

class Converter(object):

    def __init__(self, dir_torch, dir_json, dir_onnx):
        super().__init__()
        self.dir_torch=dir_torch
        self.dir_json=dir_json
        self.dir_onnx=dir_onnx
        
    def convert(self, version=12):
        with open(self.dir_json, 'r') as f:
            model_info = json.load(f)
        model_torch = TorchModel(model_info)
        model_torch.load_state_dict(torch.load(self.dir_torch))
        input_shape = model_info['model_structure']['0']['args']['shape']
        # Batch size equals 1
        input_shape.insert(0, 1)
        dummy_input = torch.ones(input_shape)
        try:
            torch.onnx.export(
                model_torch, dummy_input, self.dir_onnx,
                export_params=True,
                output_names=["out"],
                opset_version=version
            )
        except:
            # log into database if conversion fail
            
