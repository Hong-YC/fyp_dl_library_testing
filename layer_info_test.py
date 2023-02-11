from src.cases_generation.variable_generator import VariableGenerator
from src.cases_generation.layer_info_generator import LayerInfoGenerator
from src.cases_generation.torch_model import generate_layer
import torch



if __name__ == '__main__':
    # var_gen = VariableGenerator({'tensor_element_size_range': [10, 100], 'tensor_dimension_range': [2, 5]})
    # lay_info_gen = LayerInfoGenerator(var_gen)
    # info = lay_info_gen.generate((32, 3, 1024), element='Conv1d')
    # print(info[0], info[1], info[2])
    var_gen = VariableGenerator({
    'tensor_element_size_range': [10, 100], 
    'tensor_dimension_range': [2, 5], 
    'weight_value_range':[-10.0,10.0],
    'small_value_range':(0, 1)})
    lay_info_gen = LayerInfoGenerator(var_gen)
    layer_type, layer_args, cur_shape = lay_info_gen.generate((10,3,40,100), element='FractionalMaxPool2d')
    layer_dict = dict(type=layer_type, args=dict(**layer_args, name="Dummy_name"))
    print(layer_args)                             
    torch_layer = generate_layer(layer_dict)
    print(torch_layer)
    shape = (None, 16, 50, 32)
    input = torch.randn((2, *shape[1:]))
    out = torch_layer(input)