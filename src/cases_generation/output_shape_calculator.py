import math
from functools import reduce


class OutputShapeCalculator(object):

    def __init__(self):
        super().__init__()

    def linear_layer(self, input_shape, out_features, **kwargs):
        return (*input_shape[:-1], out_features)

    def reshape_layer(self, output_shape, **kwargs):
        return (None, *output_shape)

    def flatten_layer(self, input_shape, **kwargs):
        """
        We flatten all dimension except batch size, if input_shape is (batch,) we output (batch, 1)
        Hong: here input_shape[0] should be None
        """
        return (input_shape[0], reduce(lambda x, y: x*y, input_shape[1:])) if len(input_shape) >= 2 else (input_shape[0], 1)


    def conv_layer(self, input_shape, dim_num, kernel_size, padding, stride, out_channels, **kwargs):

        old_shape = input_shape[-dim_num:]


        if padding == "valid":
            new_shape = [math.floor((old_shape[i] - (kernel_size[i]-1) - 1) / stride[i]) + 1 for i in range(dim_num)]
        elif padding == "same":
            new_shape = old_shape

        return (*input_shape[:-dim_num - 1], out_channels, *new_shape)

    def activation_layer(self, input_shape):
        return input_shape

    # def pooling_layer(self, input_shape, dim_num, kernel_size, stride, padding, dilation, data_format, **kwargs):
    #     is_channels_last = (data_format == "channels_last")
    #     old_steps = input_shape[-1-dim_num:-1] if is_channels_last else input_shape[-dim_num:]

    #     plus = [2*padding - dilation*(kernel_size[i]-1) -1 for i in range(dim_num)]
    #     # plus = [0 if padding == 'valid' else kernel_size[i] - 1 for i in range(dim_num)]
    #     strides = kernel_size if stride == 0 else stride
    #     new_steps = [(old_steps[i] + plus[i]) // stride[i] + 1 for i in range(dim_num)]
    #     return (*input_shape[:-1-dim_num], *new_steps, input_shape[-1]) if is_channels_last else (*input_shape[:-dim_num], *new_steps)


    def pool1D_layer(self, input_shape, kernel_size, stride, padding, dilation, ceil_mode, **kwargs):
        length_out = (input_shape[-1] + 2* padding - dilation*(kernel_size-1)-1) // stride + 1 if not ceil_mode else math.ceil((input_shape[-1] + 2* padding - dilation*(kernel_size-1)-1) / stride + 1)
        return (*input_shape[:-1],length_out)
        # return self.pooling_layer(input_shape, 1, [kernel_size], [kernel_size] if stride == 0 else [stride], padding, dilation, "channels_last", **kwargs)
    
    def pool2D_layer(self, input_shape, kernel_size, stride, padding, dilation, ceil_mode, **kwargs):
        H_out = (input_shape[-2] + 2*padding[0] - dilation[0]*(kernel_size[0]-1)-1) // stride[0] + 1 if not ceil_mode else math.ceil((input_shape[-2] + 2*padding[0] - dilation[0]*(kernel_size[0]-1)-1) / stride[0] + 1)
        W_out = (input_shape[-1] + 2*padding[1] - dilation[1]*(kernel_size[1]-1)-1) // stride[1] + 1 if not ceil_mode else math.ceil((input_shape[-1] + 2*padding[1] - dilation[1]*(kernel_size[1]-1)-1) / stride[1] + 1)
        return (*input_shape[:-2],H_out,W_out)

    def pool3D_layer(self, input_shape, kernel_size, stride, padding, dilation, ceil_mode, **kwargs):
        D_out = (input_shape[-3] + 2*padding[0] - dilation[0]*(kernel_size[0]-1)-1) // stride[0] + 1 if not ceil_mode else math.ceil((input_shape[-3] + 2*padding[0] - dilation[0]*(kernel_size[0]-1)-1) / stride[0] + 1)
        H_out = (input_shape[-2] + 2*padding[1] - dilation[1]*(kernel_size[1]-1)-1) // stride[1] + 1 if not ceil_mode else math.ceil((input_shape[-2] + 2*padding[1] - dilation[1]*(kernel_size[1]-1)-1) / stride[1] + 1)
        W_out = (input_shape[-1] + 2*padding[2] - dilation[2]*(kernel_size[2]-1)-1) // stride[2] + 1 if not ceil_mode else math.ceil((input_shape[-1] + 2*padding[2] - dilation[2]*(kernel_size[2]-1)-1) / stride[2] + 1)
        return (*input_shape[:-3],D_out,H_out,W_out)

    def AvgPool1d_layer(self, input_shape,kernel_size,stride,padding,ceil_mode,**kwargs):
        length_out = math.floor((input_shape[-1]+2*padding-kernel_size)/stride + 1) if not ceil_mode else math.ceil((input_shape[-1]+2*padding-kernel_size)/stride + 1)
        return (*input_shape[:-1],length_out)

    def AvgPool2d_layer(self, input_shape, kernel_size, stride, padding,ceil_mode, **kwargs):
        H_out = math.floor((input_shape[-2] + 2*padding[0] - kernel_size[0])/stride[0] + 1) if not ceil_mode else math.ceil((input_shape[-2] + 2*padding[0] - kernel_size[0])/stride[0] + 1)
        W_out = math.floor((input_shape[-1] + 2*padding[1] - kernel_size[1])/stride[1] + 1) if not ceil_mode else math.ceil((input_shape[-1] + 2*padding[1] - kernel_size[1])/stride[1] + 1)
        return (*input_shape[:-2],H_out,W_out)
    
    def AvgPool3d_layer(self, input_shape, kernel_size, stride, padding,ceil_mode, **kwargs):
        D_out = math.floor((input_shape[-3] + 2*padding[0] - kernel_size[0])/stride[0] + 1) if not ceil_mode else math.ceil((input_shape[-3] + 2*padding[0] - kernel_size[0])/stride[0] + 1)
        H_out = math.floor((input_shape[-2] + 2*padding[1] - kernel_size[1])/stride[1] + 1) if not ceil_mode else math.ceil((input_shape[-2] + 2*padding[1] - kernel_size[1])/stride[1] + 1)
        W_out = math.floor((input_shape[-1] + 2*padding[2] - kernel_size[2])/stride[2] + 1) if not ceil_mode else math.ceil((input_shape[-1] + 2*padding[2] - kernel_size[2])/stride[2] + 1)
        return (*input_shape[:-2],D_out,H_out,W_out)