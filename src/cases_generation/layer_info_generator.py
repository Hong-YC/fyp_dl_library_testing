from typing import Tuple, Optional
from output_shape_calculator import OutputShapeCalculator
from variable_generator import VariableGenerator


class LayerInfo(object):

    def __init__(self, variable_generator: VariableGenerator):
        super().__init__()
        self.__output_shape = OutputShapeCalculator()
        self.__random = variable_generator

    def conv2D_layer(self, input_shape: Tuple[Optional[int]]):
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

        return 'conv2D', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=2, **args)

if __name__ == '__main__':
    var_gen = VariableGenerator({'tensor_element_size_range': [10, 100]})
    lay_info_gen = LayerInfo(var_gen)
    info = lay_info_gen.conv2D_layer((32, 3, 1024, 1024))


    print(info[1], info[2])