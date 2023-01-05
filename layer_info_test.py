from src.cases_generation.variable_generator import VariableGenerator
from src.cases_generation.layer_info_generator import LayerInfoGenerator




if __name__ == '__main__':
    var_gen = VariableGenerator({'tensor_element_size_range': [10, 100], 'tensor_dimension_range': [2, 5]})
    lay_info_gen = LayerInfoGenerator(var_gen)
    info = lay_info_gen.generate((32, 3, 3, 1024, 1024), element='Conv3d')
    print(info[0], info[1], info[2])
