import random
from typing import Tuple, List
import math


class VariableGenerator(object):

    def __init__(self, config: dict):
        super().__init__()
        self.__tensor_ele_size_range = config['tensor_element_size_range']

    def kernel_size(self, window_max_shape: Tuple[int]) -> List[int]:
        length = random.randint(1, min(window_max_shape))
        return [length for _ in window_max_shape]

    def conv_args(self, input_shape: Tuple[int], dim_num: int):
        out_channels = self.ele_size()
        window_limitation = input_shape[2:2 + dim_num]
        kernel_size = self.kernel_size(window_limitation)
        stride_limitation = [window_limitation[i] // kernel_size[i] for i in range(dim_num)]
        strides = self.sizes_with_limitation(stride_limitation)

        # Generate the groups parameter
        gcd = math.gcd(input_shape[1], out_channels)
        groups = self.choice(self.getFactor(gcd))

        return out_channels, kernel_size, strides, groups

    def choice(self, seq: list):
        return random.choice(seq)

    def boolean(self) -> bool:
        """
        Return a random boolean value
        :return: True or False
        """
        return random.random() < 0.5

    def ele_size(self) -> int:
        """
        Return a random tensor element size within the range
        :return:
        """
        return random.randint(*self.__tensor_ele_size_range)

    def sizes_with_limitation(self, limitations: Tuple[int]) -> List[int]:
        """
        Return a random generated list with the same dimension and satisfy
        the limitations
        :param limitations:
        :return:
        """
        return [random.randint(1, limit) for limit in limitations]

    def getFactor(self, input_number: int) -> List[int]:
        """
        Get all factors of the input
        :param input_number:
        :return:
        """
        factors = []
        for i in range(1, input_number + 1):
            if input_number % i == 0:
                factors.append(i)

        return factors

