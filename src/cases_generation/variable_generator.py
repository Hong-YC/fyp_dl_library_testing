import random
from typing import Tuple, List, Optional, Any,Iterable
import math


class VariableGenerator(object):

    def __init__(self, config: dict):
        super().__init__()
        self.__tensor_dim_range = config['tensor_dimension_range']
        self.__tensor_ele_size_range = config['tensor_element_size_range']
        self.__weight_val_range = config['weight_value_range']
        self.__small_val_range = config['small_value_range']

    def input_object(self, shape: Optional[Tuple[Optional[int]]] = None):
        """
        Generate a input object
        :param shape: The shape of the generated object (If None, generate randomly)
        """

        if shape is None:
            shape = self.shape()
        args = dict(
            shape=shape[1:],
            batch_shape=None
        )
        return 'input_object', args, shape

    def shape(self, dim: Optional[int] = None) -> Tuple[None, Any]:
        """
        Return a randomly generated shape element,
        for example if dim = 4, return (None, 3, 32, 37)
        """
        if dim is None:
            dim = random.randint(*self.__tensor_dim_range)
        return (
            None,
            *tuple(random.choices(range(self.__tensor_ele_size_range[0], self.__tensor_ele_size_range[1] + 1),
                                  k=dim - 1))
        )

    def target_shape(self, old_shape: Tuple[int]) -> Tuple[int]:
        """
        Return a shuffled old_shape, here old_shape doesn't include the Batch axis
        """
        res_shape = list(old_shape)
        random.shuffle(res_shape)
        return tuple(res_shape)

    def kernel_size(self, window_max_shape: Tuple[int]) -> List[int]:
        length = random.randint(1, min(window_max_shape))
        return [length for _ in window_max_shape]

    def val_size(self, must_positive: bool = False) -> float:
        '''随机返回一个weight值
        '''
        a, b = self.__weight_val_range
        if must_positive:
            a = max(a, 0)
        return random.random() * (b - a) + a

    def activation_func(self):
        '''随机返回一个激活函数
        '''
        return random.choice([
            'relu',
            'sigmoid',
            'Softmax',
            'softplus',
            # 'softsign'  # x / (abs(x) + 1) 会导致值都很接近0.99,
            'tanh',
            'selu',
            'elu',
            'linear',
        ])

    def conv_args(self, input_shape: Tuple[int], dim_num: int):

        # TODO: Support padding equals number!!!
        out_channels = self.ele_size()
        window_limitation = input_shape[2:2 + dim_num]
        kernel_size = self.kernel_size(window_limitation)
        stride_limitation = [window_limitation[i] // kernel_size[i] for i in range(dim_num)]
        strides = self.sizes_with_limitation(stride_limitation)
        if len(strides) != sum(strides):
            padding = 'valid'
        else:
            padding = self.choice(["valid", "same"])
        # Generate the groups parameter
        gcd = math.gcd(input_shape[1], out_channels)
        groups = self.choice(self.getFactor(gcd))

        return out_channels, kernel_size, strides, padding, groups

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
    
    def small_val(self):
        """
        Return a random float value within range [a,b]
        """
        a, b = self.__small_val_range
        return random.random() * (b - a) + a

    def kernel_size(self, window_max_shape: Tuple[int]) -> List[int]:
        length = random.randint(1, min(window_max_shape))
        return [random.randint(1, min(window_max_shape)) for _ in window_max_shape]
    
    def randint_in_range(self, ran: Iterable) -> int:
        '''
        Return a random integer within range [a,b]
        '''
        return random.randint(*ran)    

    # def sizes_with_limitation(self, limitations: Tuple[int]) -> List[int]:
    #     """
    #     Return a random generated list with the same dimension and satisfy
    #     the limitations
    #     :param limitations:
    #     :return:
    #     """
    #     return [random.randint(1, limit) for limit in limitations]
    def sizes_with_limitation(self, window_max_shape: Tuple[int]) -> List[int]:
        return [random.randint(1, min(window_max_shape)) for _ in window_max_shape]
    def sizes_with_limitation_zero(self, window_max_shape: Tuple[int]) -> List[int]:
        """
        Return a random generated list with the same dimension and satisfy
        the limitations
        :param limitations:
        :return:
        """
        return [random.randint(0, min(window_max_shape)) for _ in window_max_shape]

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


if __name__ == '__main__':
    config = {
        'tensor_dimension_range': 10,
        'tensor_element_size_range': (2, 1024)
    }

    g = VariableGenerator(config=config)
    print(g.shape(4))



