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

def norm_delta(x, y, mode="max"):
    # infinite norm
    if norm=="max":
        if x is None or y is None:
            return None
        return float(np.max(np.abs(x-y)))
    elif norm=="l2":
        if x is None or y is None:
            return None
        return float(np.sum((x-y)*(x-y)))
    else:
        return None

def model_output_delta(outputs1, outputs2, output_id_list):
    if outputs1 is None or outputs2 is None:
        return None
    output_deltas = {}
    for layer_name in output_id_list:
        o1=outputs1[layer_name]
        o2=outputs2[layer_name]
        output_deltas.append(layer_name,norm_delta(o1, o2))
    return output_deltas

def layer_output_ratio(outputs1, outputs2, output_id_list, epsilon=1e-7):
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
        
def compare_pytorch_onnx(model_pytorch, model_onnx, inputs):
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
        