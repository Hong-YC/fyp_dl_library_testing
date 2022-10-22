import math


class OutputShapeCalculator(object):

    def __init__(self):
        super().__init__()

    def conv_layer(self, input_shape, dim_num, kernel_size, padding, strides, out_channels, **kwargs):

        old_shape = input_shape[-dim_num:]
        if padding == "valid":
            new_shape = [math.floor((old_shape[i] + (kernel_size[i]-1) - 1) / strides[i] + 1) for i in range(dim_num)]
        elif padding == "same":
            new_shape = old_shape

        return (*input_shape[:-dim_num - 1], out_channels, *new_shape)


