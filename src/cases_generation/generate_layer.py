
from utils.utils import get_layer_func, torch_layer
from functools import partial

def generate_layer(layer_info: dict):
    # generate layer from the layer info
    # layer_type, layer_args, pre_layers, output_shape = tuple(map(layer_info.get, ['type', 'args', 'pre_layers', 'output_shape']))
    layer_type, layer_args = tuple(map(layer_info.get, ['type', 'args']))
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