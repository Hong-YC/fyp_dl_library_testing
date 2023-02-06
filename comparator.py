import torch
import argparse
import sys
import json
from pathlib import Path
import onnx
import onnxruntime
import numpy as np
import collections
from utils.tool import *
from src.cases_generation.torch_model import TorchModel

pair={}
pair['Conv2d']='Conv'
pair['ReLU']='Relu'
pair['Linear']='Gemm'
pair['Softmax']='Softmax'
pair['Flatten']='Flatten'
pair['reshape']='Reshape'
pair['LeakyReLU']='LeakyRelu'

class comparator(object):

    def __init__(self, dir_torch, dir_json, dir_onnx):
        super().__init__()
        self.dir_torch=dir_torch
        self.dir_json=dir_json
        self.dir_onnx=dir_onnx

    def preprare(self, input):
        #prepare models
        with open(self.dir_json, 'r') as f:
            self.model_info = json.load(f)
        self.model_torch=TorchModel(self.model_info)
        self.model_torch.load_state_dict(torch.load(self.dir_torch))
        
        #input shape
        self.input_shape=self.model_info['model_structure']['0']['args']['shape']
        self.input_shape.insert(0, 1)
        
        #load onnx model
        self.model_onnx = onnx.load(self.dir_onnx)
        
        #extract outputs
        self.output_torch=extract_inter_output_pytorch(self.model_torch,input)
        self.name_list=list(self.output_torch.keys())
            
        outs=extract_inter_output_onnx(self.model_onnx,input.numpy())
        self.output_onnx={}
        self.output_onnx['out']=outs['out']
        cou=0
        for node in self.model_onnx.graph.node:
            if cou==len(self.name_list):
                break
            if pair[self.model_info['model_structure'][self.name_list[cou][-1]]['type']] in node.name:
                self.output_onnx[self.name_list[cou]]=outs[node.output[0]]
                cou=cou+1

    def compare_inference(self, input):
        O_torch=list(self.model_torch(input).values())[0].detach().numpy()
        O_onnx=self.output_onnx['out']
        dif=norm_delta(O_torch, O_onnx)
        return dif


    def norm_delta(self, x, y, mode="max"):
        # infinite norm
        if mode=="max":
            if x is None or y is None:
                return None
            return float(np.max(np.abs(x-y)))
        elif mode=="l2":
            if x is None or y is None:
                return None
            return float(np.sum((x-y)*(x-y)))
        else:
            return None

    def model_output_delta(self, outputs1, outputs2, output_id_list):
        if outputs1 is None or outputs2 is None:
            return None
        output_deltas = {}
        for layer_name in output_id_list:
            o1=outputs1[layer_name]
            o2=outputs2[layer_name]
            output_deltas.append(layer_name,norm_delta(o1, o2))
        return output_deltas

    def layer_output_ratio(self, outputs1, outputs2, output_id_list, epsilon=1e-7):
        if outputs1 is None or outputs2 is None:
            return None
        output_deltas=model_output_delta(outputs1, outputs2, output_id_list)
        output_ratio={}
        pre_layer=None
        for layer, delta in output_deltas.items():
            if pre_layer==None:
                pre_layer=layer
            else:
                output_ratio[layer]=abs(output_deltas[layer]-output_deltas[pre_layer])/(epsilon+output_deltas[pre_layer])
                pre_layer=layer
        return output_ratio
            
    def compare_pytorch_onnx(self, model_pytorch, model_onnx, inputs):
        layer_name = [item[0] for item in model_pytorch._modules.items()]
        log_dir=path("./logs")
        for data in inputs:
            output_pytorch=extract_inter_output_pytorch(model_pytorch,torch.tensor(data))
            output_onnx=extract_inter_output_onnx(model_onnx,data)
            output_delta=output_ratio=None
            try:
                output_delta=model_output_delta(output_pytorch,output_onnx,layer_name)
                output_ratio=layer_output_ratio(output_pytorch, output_onnx, layer_name)
            except Exception:
                import traceback
                log_dir.mkdir(parents=True, exist_ok=True)
                with (log_dir / 'comparation.log').open(mode='a', encoding='utf-8') as f:
                    f.write(f"[ERROR] Crash when calculate inconsistencies between pytorch and onnx using {data}\n")
                    traceback.print_exc(file=f)
                    f.write("\n\n")