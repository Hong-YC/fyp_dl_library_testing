import math
from functools import reduce


class OutputShapeCalculator(object):

    def __init__(self):
        super().__init__()

    def linear_layer(self, input_shape, out_features, **kwargs):
        return (*input_shape[:-1], out_features)

    # def reshape_layer(self, target_shape, **kwargs):
    #     return (None, *target_shape)

    # def flatten_layer(self, input_shape):
    #     return (None, reduce(lambda x, y: x*y, input_shape[1:])) if len(input_shape) >= 2 else (input_shape[0], 1)


    def conv_layer(self, input_shape, dim_num, kernel_size, padding, strides, out_channels, **kwargs):

        old_shape = input_shape[-dim_num:]
        if padding == "valid":
            new_shape = [math.floor((old_shape[i] + (kernel_size[i]-1) - 1) / strides[i] + 1) for i in range(dim_num)]
        elif padding == "same":
            new_shape = old_shape

        return (*input_shape[:-dim_num - 1], out_channels, *new_shape)


