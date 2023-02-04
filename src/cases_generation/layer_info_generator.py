from typing import Tuple, Optional
from .output_shape_calculator import OutputShapeCalculator
from .variable_generator import VariableGenerator


class LayerInfoGenerator(object):
    def __init__(self, variable_generator: VariableGenerator, selector: Optional[int] = None):
        super().__init__()
        self.layer_infos = LayerInfo(variable_generator)

        from utils.utils import layer_types
        self.__layer_funcs = {name: getattr(self.layer_infos, name + '_layer') for name in layer_types}

        # self.__selector = selector

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
        # element = self.__selector.choose_element(pool=pool,
        #                                          e1=last_layer,
        #                                          input_dim=input_dim)
        # if element is None:  # No suitable layer type
        #     return None, None, input_shape

        # Hong: Temperal code
        if element is None:
            element = 'Softmax'
        
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

    def Max_Pool1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        kernel_size = self.__random.kernel_size(input_shape[2:])[0]
        args = dict(
            kernel_size = kernel_size, # input_shape:
            stride = self.__random.sizes_with_limitation(input_shape[2:])[0] ,
            padding = self.__random.sizes_with_limitation_zero([kernel_size//2])[0],
            dilation = self.__random.sizes_with_limitation([input_shape[2]//kernel_size])[0],
            return_indices = self.__random.choice([True, False]),
            ceil_mode = self.__random.choice([True, False]),
        )
        return 'Max_Pool1D', args, self.__output_shape.pool1D_layer(input_shape=input_shape, **args)

if __name__ == '__main__':
    var_gen = VariableGenerator({'tensor_element_size_range': [10, 100], 'tensor_dimension_range': [2, 5]})
    lay_info_gen = LayerInfoGenerator(var_gen)
    info = lay_info_gen.generate((32, 3, 1024, 1024))
    print(info[0], info[1], info[2])
