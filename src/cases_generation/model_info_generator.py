from typing import Optional, Tuple, List
from pathlib import Path
from functools import partial, reduce

from .variable_generator import VariableGenerator
from .layer_info_generator import LayerInfoGenerator


class ModelInfoGenerator(object):
    def __init__(self, config: dict, selector):
        super().__init__()
        self.__random = VariableGenerator(config['var'])
        self.__layer_generator = LayerInfoGenerator(self.__random, selector)

    def generate_seq_model(self, node_num: int, start_id: int = 0, pre_layer_id: Optional[int] = None,
                           pre_layer_type: Optional[str] = None,
                           input_shape: Optional[Tuple[Optional[int]]] = None,
                           output_shape: Optional[Tuple[Optional[int]]] = None,
                           pool: Optional[list] = None, cell_type: str = ''):
        """
        Generate chain sequential model, return a dictionary describing model structure
        , input shape and output shape
        :param node_num: Number of node, if specify output shape, then require >= 3 (doesn't include input object)
        :param start_id: starting index for nodes
        :return: (model_info_dict, input_shape, output_shape)

        Hong: Can input_shape be None???? Yes, if None then randomly generate one
        """
        if pre_layer_id is not None and input_shape is None:
            raise ValueError("input_shape of seq model should be provided.")

        model_structure = {}
        cur_shape = input_shape  # current shape
        pre_layers = [] if pre_layer_id is None else [pre_layer_id]
        layer_type = pre_layer_type
        layer_name = None
        skip = 0   # THe number of nodes that are skipped
        i = start_id
        while i < start_id + node_num:
            if not pre_layers:  # input layer
                layer_type, layer_args, cur_shape = self.__random.input_object(shape=cur_shape)
                input_shape = cur_shape  # record the input shape

            # Last three layers will be flatten、dense and reshape if output shape is specified
            elif output_shape and i >= start_id + node_num - 3:
                if i == start_id + node_num - 3 and len(cur_shape) <= 2:  # No need flatten
                    skip += 1
                    i += 1
                    continue

                # Using partial, we can set partial of the parameter of a function
                last_three_layers = [self.__layer_generator.layer_infos.flatten_layer,
                                     partial(self.__layer_generator.layer_infos.dense_layer, units=reduce(lambda x, y: x * y, output_shape[1:])),
                                     partial(self.__layer_generator.layer_infos.reshape_layer, output_shape=output_shape)]

                layer_type, layer_args, cur_shape = last_three_layers[i - (start_id + node_num - 3)](cur_shape)

            else:  # Middle layer
                layer_type, layer_args, cur_shape = self.__layer_generator.generate(cur_shape, last_layer=layer_type,
                                                                                    pool=pool)
                if layer_type is None:  # This means there is no suitable layer
                    skip += 1
                    i += 1
                    continue

            j = i - skip
            layer_name = construct_layer_name(j, layer_type, cell_type)
            print(f"{layer_name}: {cur_shape}")

            # 形状太大的就抛出错误
            if self.__shape_too_big(cur_shape):
                raise ValueError("Shape too big!!")

            model_structure[j] = dict(type=layer_type,
                                      args=dict(**layer_args, name=layer_name),
                                      pre_layers=pre_layers,
                                      output_shape=cur_shape)
            pre_layers = [j]
            i += 1

        return (
            dict(model_structure=model_structure,
                 input_id_list=[start_id],
                 output_id_list=[start_id + node_num - skip - 1]),
            {construct_layer_name(start_id, 'input_object', cell_type): input_shape} if pre_layer_id is None else {},
            {layer_name: cur_shape},
            node_num - skip
        )
