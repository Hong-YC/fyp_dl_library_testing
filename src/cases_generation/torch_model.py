import torch.nn as nn
from functools import partial
from generate_one import generate_layer
from model_info_generator import ModelInfoGenerator
from utils import torch_layer
from torchsummary import summary


class TorchModel(nn.Module):
    def __init__(self, model_info: dict):
        super(TorchModel, self).__init__()
        self.input_id_list = model_info['input_id_list']
        self.output_id_list = model_info['output_id_list']
        self.model_structure = model_info['model_structure']
        self.torch_layers = {}
        layer_dict = {}

        # Process model structure by topological order
        for layer_id, layer_info in self.model_structure.items():
            # layer_id = int(layer_id)
            layer_type = layer_info.get("type")
            # Generate layers
            if layer_id not in self.input_id_list:
                if layer_type not in torch_layer:
                    layer_dict[layer_id] = generate_layer(layer_info)
                else:
                    self.torch_layers[layer_id] = generate_layer(layer_info)

        self.torch_nn_layers = nn.ModuleDict(layer_dict)

    def forward(self, input):
        """
        :param input_dict: key: id(str), value: input_tensor
        """
        # Store the intermediate output of each layer
        # output_dict = input_dict.copy()
        output_dict = {self.input_id_list[0]: input}
        result_dict = {}
        # TODO: Finish the forward function!!
        for layer_id, layer_info in self.model_structure.items():
            layer_type = layer_info.get("type")
            if layer_id not in self.input_id_list:
                inbound_layers_idx = layer_info.get("pre_layers")
                inbound_layers_output = [output_dict[i] for i in inbound_layers_idx]
                print("Shape: ", inbound_layers_output[0].size())
                cur_layer = self.torch_layers[layer_id] if layer_type in torch_layer else self.torch_nn_layers[layer_id]
                # store the output of current layer
                output_dict[layer_id] = cur_layer(inbound_layers_output[0]) if \
                    len(inbound_layers_output) == 1 else cur_layer[layer_id](*inbound_layers_output)
                # Store the result if it is an output layer
                if layer_id in self.output_id_list:
                    result_dict[layer_id] = output_dict[layer_id]

        return result_dict


if __name__ == '__main__':
    config = {
        'var': {
            'tensor_dimension_range': (4, 4),
            'tensor_element_size_range': (2, 64)
        }
    }
    m_info_generator = ModelInfoGenerator(config)

    m_info = m_info_generator.generate_seq_model(6, output_shape=(None, 3, 4))
    model = TorchModel(m_info[0])

    print(m_info[0]['model_structure'])
    input_shape = m_info[1]["00_input_object"]
    summary(model, input_shape[1:])


