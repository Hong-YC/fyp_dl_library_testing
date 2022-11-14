import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings
from utils import torch_layer
from functools import partial

from utils import get_layer_func

warnings.filterwarnings("ignore")


def generate_layer(layer_info: dict):
    # generate layer from the layer info
    layer_type, layer_args, pre_layers, output_shape = tuple(map(layer_info.get, ['type', 'args', 'pre_layers', 'output_shape']))
    layer = get_layer_func(layer_type)

    # Hong: we remove name temporarily
    name = layer_args.pop('name', None)

    # Hong: Layers from torch module require input, thus we use partial here
    if layer_type in torch_layer:
        layer_w_arg = partial(layer, **layer_args)
    else:
        layer_w_arg = layer(**layer_args)

    if name is not None:
        # Hong: add name back
        layer_args['name'] = name

    return layer_w_arg


def __generate_model(json_path: str, weight_range: Tuple[float]):
    # load model structure info
    with open(json_path, 'r') as f:
        model_info = json.load(f)

    input_id_list, output_id_list = model_info['input_id_list'], model_info['output_id_list']

    input_list, output_list, layer_dict = [], [], {}
    for layer_id, layer_info in model_info['model_structure'].items():  # 按拓扑排序遍历
        layer_id = int(layer_id)
        # 生成层
        layer, inbound_layers_idx, ouput_shape = generate_layer(layer_info)

        # 层拼接
        if layer_id in input_id_list:
            layer_dict[layer_id] = layer  # input_object
            input_list.append(layer_dict[layer_id])

        else:
            inbound_layers = [layer_dict[i] for i in inbound_layers_idx]
            layer_dict[layer_id] = layer(inbound_layers[0] if len(inbound_layers) == 1 else inbound_layers)  # 对layers进行连接

        if layer_id in output_id_list:
            output_list.append(layer_dict[layer_id])

        # 检查形状
        from keras import backend as K
        if K.int_shape(layer_dict[layer_id]) != tuple(ouput_shape):
            raise Exception(f"[Debug] layer_id: {layer_id} expected shape: {tuple(ouput_shape)}  actual shape: {K.int_shape(layer_dict[layer_id])}")

    return keras.Model(inputs=input_list, outputs=output_list)

