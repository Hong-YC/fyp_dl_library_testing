from typing import Optional, Tuple, List
from pathlib import Path
from functools import partial, reduce
from utils.utils import construct_layer_name
from .variable_generator import VariableGenerator
from .layer_info_generator import LayerInfoGenerator
from .generate_layer import generate_layer



import json


class ModelInfoGenerator(object):
    def __init__(self, config: dict, generate_mode, db_manager, selector=None):
        super().__init__()
        self.__node_num_range = config['node_num_range']
        self.__random = VariableGenerator(config['var'])
        self.__layer_generator = LayerInfoGenerator(self.__random, selector)
        self.__generate_mode = generate_mode
        self.__db_manager = db_manager

    def generate(self, save_dir: str):
        """
        Generate model information (Hong: Only support sequential for now)
        Can specify number of node

        Return:
            json_path, input_shapes, model_id, exp_dir
        """
        # Randomly select number of nodes if not specified
        node_num = self.__random.randint_in_range(self.__node_num_range)           

        if self.__generate_mode == 'seq':
            # TODO: Use the following code instead for final version (Random sequence of layer w random output shape)
            # model_info, input_shapes, output_shapes, node_num = self.generate_seq_model(node_num=node_num)
            output_shape = self.__random.shape()
            model_info, input_shapes, output_shapes, node_num = self.generate_seq_model(node_num=node_num, output_shape=output_shape)
        else:
            raise ValueError(f"UnKnown generate mode '{self.__generate_mode}'")


        # Register model in the database
        dataset_name = model_info.get('dataset_name', None)
        model_id = self.__db_manager.register_model(dataset_name, node_num)

        # Create experiment and model folders
        exp_dir = Path(save_dir) / str(model_id).zfill(6)
        model_dir = exp_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)

        json_path = model_dir / 'model.json'
        with open(str(json_path), 'w') as f:
            json.dump(model_info, f)

        return json_path, input_shapes, output_shapes, model_id, str(exp_dir)
    def generate_seq_model(self, node_num: int, start_id: int = 0, pre_layer_id: Optional[int] = None,
                           pre_layer_type: Optional[str] = None,
                           input_shape: Optional[Tuple[Optional[int]]] = None,
                           output_shape: Optional[Tuple[Optional[int]]] = None,
                           pool: Optional[list] = None, cell_type: str = '',
                            element: Optional[str] = None):
        """
        Generate sequential model, return a model info description dict, input and output shape of the model
        :param node_num: Number of node. Required >= 3 if output shape are specified. (doesn't include input object)
        :param start_id: Initial id for starting node (default = 0)
        :param input_shape: input shape of the model (if None, use random shape)
        :param output_shape: output shape of the model (if None, use random shape)
        :param pool: specify the layer pool
        :return: (model_info_dict, input_shape, output_shape)
        """
        if pre_layer_id is not None and input_shape is None:
            raise ValueError("input_shape of seq model should be provided.")
        if output_shape is not None and node_num < 3:
            raise ValueError("node number should be greater than 2 when output shape is specified")

        model_structure = {}
        cur_shape = input_shape  # current shape
        pre_layers = [] if pre_layer_id is None else [pre_layer_id]
        layer_type = pre_layer_type
        layer_name = None
        skip = 0  # The number of nodes that are skipped
        i = start_id
        while i < start_id + node_num:
            if not pre_layers:  # input layer
                layer_type, layer_args, cur_shape = self.__random.input_object(shape=cur_shape)
                input_shape = cur_shape  # record the input shape

            # Last three layers will be flatten, dense and reshape if output shape is specified
            elif output_shape and i >= start_id + node_num - 3:
                # # No need flatten (Hong: Skip for now, seems redundant)
                # if i == start_id + node_num - 3 and len(cur_shape) <= 2:
                #     skip += 1
                #     i += 1
                #     continue

                # Using partial, we can set partial of the parameter of a function
                # Flatten layer will flatten the input tensor (Batch, D1, D2, ...) to (Batch, -1)
                # Linear layer will transform the tensor to (Batch, multiply_output_shape)
                # Reshape layer will transform the tensor to (Batch, *output_shape)
                last_three_layers = [self.__layer_generator.layer_infos.Flatten_layer,
                                     partial(self.__layer_generator.layer_infos.Linear_layer,
                                             out_features=reduce(lambda x, y: x * y, output_shape[1:])),
                                     partial(self.__layer_generator.layer_infos.reshape_layer,
                                             output_shape=output_shape)]

                layer_type, layer_args, cur_shape = last_three_layers[i - (start_id + node_num - 3)](cur_shape)

            else:  # Middle layer
                import torch
                dummy_input = torch.randn((2, *cur_shape[1:]))
                #TODO Check whether the generated layer is valid, if not then regenerate                                                                     
                while True:
                    try:
                        layer_type, layer_args, cur_shape = self.__layer_generator.generate(cur_shape, last_layer=layer_type, pool=pool, element=element)
                        layer_dict = dict(type=layer_type, args=dict(**layer_args, name="Dummy_name"))
                        torch_layer = generate_layer(layer_dict)
                        out = torch_layer(dummy_input)                        
                        break
                    except Exception:
                        import traceback
                        print("[ERROR] Fail to generate layer")
                        traceback.print_exc()       
                                                                        

                if layer_type is None:  # This means there is no suitable layer
                    skip += 1
                    i += 1
                    continue

            j = i - skip
            layer_name = construct_layer_name(j, layer_type, cell_type)
            print(f"{layer_name}: {cur_shape}")

            # Raise error if the generated shape is too big
            if self.__shape_too_big(cur_shape):
                raise ValueError("Shape too big or is zero!!")

            model_structure[str(j)] = dict(type=layer_type,
                                           args=dict(**layer_args, name=layer_name),
                                           pre_layers=pre_layers,
                                           output_shape=cur_shape)
            pre_layers = [str(j)]
            i += 1

        return (
            dict(model_structure=model_structure,
                 input_id_list=[str(start_id)],
                 output_id_list=[str(start_id + node_num - skip - 1)]),
            {construct_layer_name(start_id, 'input_object', cell_type): input_shape} if pre_layer_id is None else {},
            {layer_name: cur_shape},
            node_num - skip
        )

    def __shape_too_big(self, cur_shape):
        temp = 1
        for e in cur_shape[1:]:
            if e > 1e6 or e <= 0:
                return True
            temp *= e
        return temp > 1e8


if __name__ == '__main__':
    config = {
        'var': {
            'tensor_dimension_range': (4, 4),
            'tensor_element_size_range': (2, 64)
        }
    }
    from utils.db_manager import DbManager
    db_manager = DbManager(str(Path.cwd() / 'data' / 'dummy.db'))

    m_info_generator = ModelInfoGenerator(config, 'seq', db_manager)

    m_info = m_info_generator.generate_seq_model(6, output_shape=(None, 3, 4))
    # print(m_info)
    # print(m_info[0]['model_structure'])
    # print(m_info[0]['input_id_list'])
    # print(m_info[0]['output_id_list'])
    # print(m_info[1])

    print(m_info[0]['model_structure'])
    # actual_layer = generate_layer(m_info[0]['model_structure'][i])
    # print(actual_layer[0])
