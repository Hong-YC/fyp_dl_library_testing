from typing import Tuple, Optional
from .output_shape_calculator import OutputShapeCalculator
from .variable_generator import VariableGenerator
import math,random
from utils.utils import layer_types



class LayerInfoGenerator(object):
    def __init__(self, variable_generator: VariableGenerator, selector: Optional[int] = None):
        super().__init__()
        self.layer_infos = LayerInfo(variable_generator)

        from utils.utils import layer_types
        self.__layer_funcs = {name: getattr(self.layer_infos, name + '_layer') for name in layer_types}
        self.__selector = selector

    def generate(self, input_shape: Tuple[Optional[int]], last_layer: Optional[str] = None, pool: Optional[list] = None, element: Optional[str] = None):
        """
        Generate a random layer name, parameter dict, and output shape
        :param input_shape:
        :param last_layer:
        :param pool: A set of layer types to choose from
        :return:
        """
        input_dim = len(input_shape)
        # normal_pool = set(seq_layer_types+RNN_layer_types+activation_layer_types)
        # pool = list(set(pool) & normal_pool) if pool is not None else list(normal_pool)
        pool = list(set(layer_types))
        element = self.__selector.choose_element(pool=pool,
                                                 e1=last_layer,
                                                 input_dim=input_dim)
        if element is None:  # No suitable layer type
            return None, None, input_shape

        # Hong: Temperal code
        # if element is None:
        #     element = 'Softmax'
        
        return self.__layer_funcs[element](input_shape=input_shape)


class LayerInfo(object):

    def __init__(self, variable_generator: VariableGenerator):
        super().__init__()
        self.__output_shape = OutputShapeCalculator()
        self.__random = variable_generator

    def activation_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(activation=self.__random.activation_func())
        print('----------------',args)
        return 'activation', args, self.__output_shape.activation_layer(input_shape)

    def reshape_layer(self, input_shape: Tuple[Optional[int]], output_shape: Optional[Tuple[Optional[int]]] = None):
        if output_shape is None: # generate an output shape by randomly shuffle the input_shape
            output_shape = self.__random.target_shape(input_shape[1:])
        else:
            output_shape = output_shape[1:]
        args = dict(
            shape=(-1, *output_shape)
        )
        return 'reshape', args, self.__output_shape.reshape_layer(output_shape)

    def Linear_layer(self, input_shape: Tuple[Optional[int]], out_features: Optional[int] = None):
        """
        A layer that applies a linear transformation to the last dimension of the input tensor
        """

        if out_features is None:
            out_features = self.__random.ele_size()
        args = dict(
            in_features=input_shape[-1],
            out_features=out_features,
            bias=self.__random.boolean(),

        )
        return 'Linear', args, self.__output_shape.linear_layer(input_shape, **args)

    def Flatten_layer(self, input_shape: Tuple[Optional[int]], start_dim: Optional[int]=None, end_dim: Optional[int]=None):
        """
        Flatten the input tensor except batch size
        TODO: Flatten the input tensor within the start and end dimension
        """
        args = dict(
            start_dim=1, # Skip the batch dim
            end_dim=-1
        )
        return 'Flatten', args, self.__output_shape.flatten_layer(input_shape=input_shape)

    def Conv1d_layer(self, input_shape: Tuple[Optional[int]]):
        """
        Generate a 1D convolution layer with random parameters
        :param input_shape: 3D (N, C, L)
        :return: 'conv1D', generated_arguments, output_shape
        """

        out_channels, kernel_size, strides, padding, groups = self.__random.conv_args(input_shape, dim_num=1)

        args = dict(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,  # Improvement: Add int or tuple input
            # padding_mode=self.__random.choice(['zeros', 'reflect', 'replicate', 'circular']),
            padding_mode=self.__random.choice(['reflect', 'replicate', 'circular']),
            # dilation=dilation,   Improvement: Add support for dilation
            groups=groups,
            bias=self.__random.boolean()
        )

        return 'Conv1d', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=1, **args)



    def Conv2d_layer(self, input_shape: Tuple[Optional[int]]):
        """
        Generate a 2D convolution layer with random parameters
        :param input_shape: 4D (N, C, H, W )
        :return: 'conv2D', generated_arguments, output_shape
        """

        out_channels, kernel_size, strides, padding, groups = self.__random.conv_args(input_shape, dim_num=2)

        args = dict(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,  # Improvement: Add int or tuple input
            # padding_mode=self.__random.choice(['zeros', 'reflect', 'replicate', 'circular']),
            padding_mode=self.__random.choice(['reflect', 'replicate', 'circular']),
            # dilation=dilation,   Improvement: Add support for dilation
            groups=groups,
            bias=self.__random.boolean()
        )

        return 'Conv2d', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=2, **args)


    def Conv3d_layer(self, input_shape: Tuple[Optional[int]]):
        """
        Generate a 3D convolution layer with random parameters
        :param input_shape: 5D (N, C, D, H, W )
        :return: 'conv3D', generated_arguments, output_shape
        """

        out_channels, kernel_size, strides, padding, groups = self.__random.conv_args(input_shape, dim_num=3)

        args = dict(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,  # Improvement: Add int or tuple input
            # padding_mode=self.__random.choice(['zeros', 'reflect', 'replicate', 'circular']),
            padding_mode=self.__random.choice(['reflect', 'replicate', 'circular']),
            # dilation=dilation,   Improvement: Add support for dilation
            groups=groups,
            bias=self.__random.boolean()
        )

        return 'Conv3d', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=3, **args)

    def ReLU_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
        )
        return 'ReLU', args, self.__output_shape.activation_layer(input_shape)

    def Softmax_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
        )
        return 'Softmax', args, self.__output_shape.activation_layer(input_shape)

    def LeakyReLU_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            negative_slope = self.__random.small_val(),
            inplace=False,
        )
        return 'LeakyReLU', args, self.__output_shape.activation_layer(input_shape)

    def PReLU_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            num_parameters = 1,  # Jingyu todo: support channels? how to tell which index is channel
            init = self.__random.small_val(),
        )
        return 'PReLU', args, self.__output_shape.activation_layer(input_shape)

    def ELU_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            alpha=self.__random.small_val(),
            inplace=self.__random.choice([True, False])
        )
        return 'ELU', args, self.__output_shape.activation_layer(input_shape)

    def Threshold_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            threshold_layer=self.__random.small_val(),
            value=self.__random.randint_in_range([-10**10,1010]),
            inplace=self.__random.choice([True, False])
        )
        return 'thresholded_ReLU', args, self.__output_shape.activation_layer(input_shape)

    def MaxPool1d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        kernel_size = self.__random.kernel_size(input_shape[2:])[0]
        args = dict(
            kernel_size = kernel_size, # input_shape:
            stride = self.__random.sizes_with_limitation(input_shape[2:])[0] ,
            padding = self.__random.sizes_with_limitation_zero([kernel_size//2])[0],
            dilation = self.__random.sizes_with_limitation([input_shape[2]//kernel_size])[0],
            # return_indices = self.__random.choice([True, False]),
            ceil_mode = self.__random.choice([True, False]),
        )
        return 'MaxPool1d', args, self.__output_shape.pool1D_layer(input_shape=input_shape, **args)

    def MaxPool2d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        window_limitation = input_shape[2:] # (H,W)
        kernel_size = self.__random.kernel_size(window_limitation) # (H,W)
        args = dict(
            kernel_size = kernel_size,
            stride = self.__random.sizes_with_limitation(window_limitation) if self.__random.boolean() else kernel_size,
            padding = self.__random.sizes_with_limitation_zero([i//2 for i in kernel_size]),
            dilation = self.__random.sizes_with_limitation([window_limitation[i]//kernel_size[i] for i in range(2)]),
            # return_indices = self.__random.choice([True, False]),
            ceil_mode = self.__random.choice([True, False]),
        )
        return 'MaxPool2d', args, self.__output_shape.pool2D_layer(input_shape=input_shape,**args)

    def MaxPool3d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        window_limitation = input_shape[2:] # (D,H,W)
        kernel_size = self.__random.kernel_size(window_limitation) # (D,H,W)
        args = dict(
            kernel_size = kernel_size,
            stride = self.__random.sizes_with_limitation(window_limitation) if self.__random.boolean() else kernel_size,
            padding = self.__random.sizes_with_limitation_zero([i//2 for i in kernel_size]),
            dilation = self.__random.sizes_with_limitation([window_limitation[i]//kernel_size[i] for i in range(3)]),
            # return_indices = self.__random.choice([True, False]),
            ceil_mode = self.__random.choice([True, False]),
        )
        return 'MaxPool3d', args, self.__output_shape.pool3D_layer(input_shape=input_shape,**args)

    def AvgPool1d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        kernel_size = self.__random.kernel_size(input_shape[2:])[0]
        args = dict(
            kernel_size = kernel_size,
            stride = self.__random.sizes_with_limitation(input_shape[2:])[0] ,
            padding = self.__random.sizes_with_limitation_zero([kernel_size//2])[0],
            ceil_mode = self.__random.choice([True, False]),
            count_include_pad = self.__random.choice([True, False]),
        )
        return 'AvgPool1d', args, self.__output_shape.AvgPool1d_layer(input_shape=input_shape, **args)

    def AvgPool2d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        window_limitation = input_shape[2:] # (H,W)
        kernel_size = self.__random.kernel_size(window_limitation) # (H,W)
        args = dict(
            kernel_size = kernel_size,
            stride = self.__random.sizes_with_limitation(window_limitation) if self.__random.boolean() else kernel_size,
            padding = self.__random.sizes_with_limitation_zero([i//2 for i in kernel_size]),
            ceil_mode = self.__random.choice([True, False]),
            count_include_pad = self.__random.choice([True, False]),
            divisor_override = None if self.__random.boolean() else self.__random.randint_in_range([1,10**10])
        )
        
        return 'AvgPool2d', args, self.__output_shape.AvgPool2d_layer(input_shape=input_shape, **args)

    def AvgPool3d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        window_limitation = input_shape[2:] # (D,H,W)
        kernel_size = self.__random.kernel_size(window_limitation) # (D,H,W)
        args = dict(
            kernel_size = kernel_size,
            stride = self.__random.sizes_with_limitation(window_limitation) if self.__random.boolean() else kernel_size,
            padding = self.__random.sizes_with_limitation_zero([i//2 for i in kernel_size]),
            ceil_mode = self.__random.choice([True, False]),
            count_include_pad = self.__random.choice([True, False]),
            divisor_override = None if self.__random.boolean() else self.__random.randint_in_range([1,10**10])
        )
        
        return 'AvgPool3d', args, self.__output_shape.AvgPool3d_layer(input_shape=input_shape, **args)

    def FractionalMaxPool2d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''        
        window_limitation = input_shape[2:] # (H,W)
        kernel_size = self.__random.kernel_size(window_limitation) # (H,W)
        output_size = None if self.__random.boolean() else self.__random.sizes_with_limitation([math.ceil(i/2) for i in kernel_size])
        args = dict(
            kernel_size = kernel_size,
            output_size = output_size,
            output_ratio = None if output_size else [random.uniform(0,1) for i in range(2)],
            # return_indices = self.__random.choice([True, False]),
        )
        return 'FractionalMaxPool2d', args, self.__output_shape.FractionalMaxPool2d_layer(input_shape=input_shape, **args)

    def FractionalMaxPool3d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''        
        window_limitation = input_shape[2:] # (T,H,W)
        kernel_size = self.__random.kernel_size(window_limitation) # (T,H,W)
        output_size = None if self.__random.boolean() else self.__random.sizes_with_limitation([math.ceil(i/2) for i in kernel_size])
        args = dict(
            kernel_size = kernel_size,
            output_size = output_size,
            output_ratio = None if output_size else [random.uniform(0,1) for i in range(3)],
            # return_indices = self.__random.choice([True, False]),
        )
        return 'FractionalMaxPool3d', args, self.__output_shape.FractionalMaxPool3d_layer(input_shape=input_shape, **args)

    def BatchNorm1d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict(
            num_features = input_shape[-2],
            eps=self.__random.small_val(),
            momentum=None if self.__random.boolean() else self.__random.small_val(),
            affine=self.__random.choice([True, False]),
            track_running_stats=self.__random.choice([True, False]),
        )
        return 'BatchNorm1d', args, self.__output_shape.BatchNorm_layer(input_shape=input_shape)

    def BatchNorm2d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        args = dict(
            num_features = input_shape[-3],
            eps=self.__random.small_val(),
            momentum=None if self.__random.boolean() else self.__random.small_val(),
            affine=self.__random.choice([True, False]),
            track_running_stats=self.__random.choice([True, False]),
        )
        return 'BatchNorm2d', args, self.__output_shape.BatchNorm_layer(input_shape=input_shape)

    def BatchNorm3d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        args = dict(
            num_features = input_shape[-4],
            eps=self.__random.small_val(),
            momentum=None if self.__random.boolean() else self.__random.small_val(),
            affine=self.__random.choice([True, False]),
            track_running_stats=self.__random.choice([True, False]),
        )
        return 'BatchNorm3d', args, self.__output_shape.BatchNorm_layer(input_shape=input_shape)

    def LazyBatchNorm1d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict(
            eps=self.__random.small_val(),
            momentum=None if self.__random.boolean() else self.__random.small_val(),
            affine=self.__random.choice([True, False]),
            track_running_stats=self.__random.choice([True, False]),
        )
        return 'LazyBatchNorm1d', args, self.__output_shape.BatchNorm_layer(input_shape=input_shape)        

    def LazyBatchNorm2d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        args = dict(
            eps=self.__random.small_val(),
            momentum=None if self.__random.boolean() else self.__random.small_val(),
            affine=self.__random.choice([True, False]),
            track_running_stats=self.__random.choice([True, False]),
        )
        return 'LazyBatchNorm2d', args, self.__output_shape.BatchNorm_layer(input_shape=input_shape) 

    def LazyBatchNorm3d_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        args = dict(
            eps=self.__random.small_val(),
            momentum=None if self.__random.boolean() else self.__random.small_val(),
            affine=self.__random.choice([True, False]),
            track_running_stats=self.__random.choice([True, False]),
        )
        return 'LazyBatchNorm3d', args, self.__output_shape.BatchNorm_layer(input_shape=input_shape) 

    def GroupNorm_layer(self, input_shape: Tuple[Optional[int]]):
        ''' 
        Input: (N,C,*), where C= num_channels
        Output: (N,C,*), same shape as input
        '''
        args = dict(
            num_groups = self.__random.choice(self.__random.getFactor(input_shape[1])),
            num_channels = input_shape[1],
            eps=self.__random.small_val(),
            affine=self.__random.choice([True, False]),
        )
        return 'GroupNorm', args, self.__output_shape.BatchNorm_layer(input_shape=input_shape) 


if __name__ == '__main__':
    var_gen = VariableGenerator({'tensor_element_size_range': [10, 100], 'tensor_dimension_range': [2, 5]})
    lay_info_gen = LayerInfoGenerator(var_gen)
    info = lay_info_gen.generate((32, 3, 1024, 1024))
    print(info[0], info[1], info[2])
