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
from utils.db_manager import *
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

    def __init__(self, dir_torch, dir_json, dir_onnx, db_m, model_id):
        super().__init__()
        self.__dir_torch=dir_torch
        self.__dir_json=dir_json
        self.__dir_onnx=dir_onnx
        self.__db_m=db_m
        self.__model_id=model_id
        
        #prepare models
        with open(self.__dir_json, 'r') as f:
            self.__model_info = json.load(f)
        self.__model_torch=TorchModel(self.__model_info)
        self.__model_torch.load_state_dict(torch.load(self.__dir_torch))
        
        #load onnx model
        self.__model_onnx = onnx.load(self.__dir_onnx)
        
    def extract(self, input):
        #extract outputs
        self.__output_torch=extract_inter_output_pytorch(self.__model_torch,input)
        self.__name_list=list(self.__output_torch.keys())
            
        outs=extract_inter_output_onnx(self.__model_onnx,input.numpy())
        self.__output_onnx={}
        self.__output_onnx['out']=outs['out']
        cou=0
        for node in self.__model_onnx.graph.node:
            if cou==len(self.__name_list):
                break
            if pair[self.__model_info['model_structure'][self.__name_list[cou][-1]]['type']] in node.name:
                self.__output_onnx[self.__name_list[cou]]=outs[node.output[0]]
                cou=cou+1

    def compare_inference(self, input, epsilon=1e-4):
        O_torch=None
        O_onnx=None
        crash=False
        try:
            O_torch=list(self.__model_torch(input).values())[0].detach().numpy()
        except Exception:
            crash=True
            self.__db_m.update_model_crash_backends(self.__model_id, ['torch'])
        try:
            sess = onnxruntime.InferenceSession(self.__dir_onnx)
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
            ins=input.numpy()
            result = sess.run(None, {input_name: ins})
            O_onnx=result[0]
        except Exception:
            crash=True
            self.__db_m.update_model_crash_backends(self.__model_id, ['onnx'])
        if crash:
            return None
        
        comparable=True
        if (True in np.isinf(O_torch)):
            update_model_inf_backends(self.__model_id, ['torch'])
            comparable=False
        if (True in np.isnan(O_torch)):
            update_model_nan_backends(self.__model_id, ['torch'])
            comparable=False
        if (True in np.isinf(O_onnx)):
            update_model_inf_backends(self.__model_id, ['onnx'])
            comparable=Falsev
        if (True in np.isnan(O_onnx)):
            update_model_nan_backends(self.__model_id, ['onnx'])
            comparable=False
        
        if not comparable:
            return None
        
        dif=norm_delta(O_torch, O_onnx)
        if dif>epsilon:
            add_inconsistencies([(self.__model_id,input,dif)])
            
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

    def model_output_delta(self):
        if self.__output_onnx is None or self.__output_torch is None:
            return None
        output_deltas = {}
        for layer_name in self.__name_list:
            o1=self.__output_onnx[layer_name]
            o2=self.__output_torch[layer_name]
            output_deltas.append(layer_name,norm_delta(o1, o2))
        return output_deltas

    def layer_output_ratio(self, epsilon=1e-7):
        if self.__output_onnx is None or self.__output_torch is None:
            return None
        output_deltas=model_output_delta()
        output_ratio={}
        pre_layer=None
        for layer, delta in output_deltas.items():
            if pre_layer==None:
                pre_layer=layer
            else:
                output_ratio[layer]=abs(output_deltas[layer]-output_deltas[pre_layer])/(epsilon+output_deltas[pre_layer])
                pre_layer=layer
        return output_ratio