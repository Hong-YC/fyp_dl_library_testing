

def construct_layer_name(layer_id, layer_type, cell_type=''):
    return str(layer_id).zfill(2) + '_' + layer_type + (('' if cell_type == '' else '_') + cell_type)


def get_layer_func(layer_type):
    import torch
    if layer_type in torch_layer:
        return getattr(torch, layer_type)
    else:
        return getattr(torch.nn, layer_type)


torch_layer = [
    'reshape'
]

torch_nn_layer = [
    'Linear',
    'Conv2d',
    'Flatten'
]
seq_layer_types = [
    'Linear',
    'Conv2d',
    'Flatten'
]

layer_types = seq_layer_types


if __name__ == '__main__':
    c = get_layer_func("Conv2d")
    print(c(32, 3, (1024, 1024)))
