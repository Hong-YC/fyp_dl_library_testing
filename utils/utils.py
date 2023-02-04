import os

use_gpu = True

def construct_layer_name(layer_id, layer_type, cell_type=''):
    return str(layer_id).zfill(2) + '_' + layer_type + (('' if cell_type == '' else '_') + cell_type)


def get_layer_func(layer_type):
    import torch
    if layer_type in torch_layer:
        return getattr(torch, layer_type)
    else:
        return getattr(torch.nn, layer_type)

def get_HH_mm_ss(td):
    days, seconds = td.days, td.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return hours, minutes, secs



torch_layer = [
    'reshape'
]

# torch_nn_layer = [
#     'Linear',
#     'Conv2d',
#     'Flatten'
# ]
seq_layer_types = [
    'Linear',
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'Flatten'
]

activation_layer_types = [
    # 'activation',
    'ReLU',
    'Softmax',
    'LeakyReLU',
    # 'PReLU',
    # 'ELU',
    # 'thresholded_ReLU',
]

# layer_types = seq_layer_types
layer_types = seq_layer_types + activation_layer_types + torch_layer


# TODO: Improve rules
activation_cond = (lambda **kwargs: kwargs['e1'] in [*seq_layer_types, *torch_layer])

dim_3_cond = (lambda **kwargs: kwargs.get('input_dim', None) == 3)
dim_4_cond = (lambda **kwargs: kwargs.get('input_dim', None) == 4)
dim_5_cond = (lambda **kwargs: kwargs.get('input_dim', None) == 5)

# Rules for connecting layers
layer_conditions = {
    **{layer_name: activation_cond for layer_name in activation_layer_types},
    'Conv1d': dim_3_cond,
    'Conv2d': dim_4_cond,
    'Conv3d': dim_5_cond,
    # Since we skip the batch dim when flattening
    'Flatten': (lambda **kwargs: kwargs.get('input_dim', None) >= 3)
}


if __name__ == '__main__':
    c = get_layer_func("Conv2d")
    print(c(32, 3, (1024, 1024)))
