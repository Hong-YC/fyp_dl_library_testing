from typing import Tuple, Optional
from output_shape_calculator import OutputShapeCalculator
from variable_generator import VariableGenerator


class LayerInfoGenerator(object):
    def __init__(self, variable_generator: VariableGenerator, selector: Optional[int] = None):
        super().__init__()
        self.layer_infos = LayerInfo(variable_generator)

        from utils import layer_types
        self.__layer_funcs = {name: getattr(self.layer_infos, name + '_layer') for name in layer_types}

        # self.__selector = selector

    def generate(self, input_shape: Tuple[Optional[int]], last_layer: Optional[str] = None, pool: Optional[list] = None):
        """
        Generate a random layer name, parameter dict, and output shape
        :param input_shape:
        :param last_layer:
        :param pool:
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
        element = 'Conv2d'

        return self.__layer_funcs[element](input_shape=input_shape)

class LayerInfo(object):

    def __init__(self, variable_generator: VariableGenerator):
        super().__init__()
        self.__output_shape = OutputShapeCalculator()
        self.__random = variable_generator

    def Conv2d_layer(self, input_shape: Tuple[Optional[int]]):
        """
        Generate a 2D convolution layer with random parameters
        :param input_shape: 4D (N, C, H, W )
        :return: 'conv2D', generated_arguments, output_shape
        """

        out_channels, kernel_size, strides, groups = self.__random.conv_args(input_shape, dim_num=2)

        args = dict(
            in_channels=input_shape[1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=self.__random.choice(["valid", "same"]),  # Improvement: Add int or tuple input
            padding_mode=self.__random.choice(['zeros', 'reflect', 'replicate', 'circular']),
            # dilation=dilation,   Improvement: Add support for dilation
            groups=groups,
            bias=self.__random.boolean()
        )

        return 'Conv2d', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=2, **args)


if __name__ == '__main__':
    var_gen = VariableGenerator({'tensor_element_size_range': [10, 100]})
    lay_info_gen = LayerInfoGenerator(var_gen)
    info = lay_info_gen.generate((32, 3, 1024, 1024))
    print(info[1], info[2])
