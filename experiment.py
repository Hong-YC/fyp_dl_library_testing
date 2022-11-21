import torch
import onnx
import onnxruntime
from src.cases_generation.torch_model import TorchModel
from src.cases_generation.model_info_generator import ModelInfoGenerator
from src.utils.tool import to_numpy
from torchsummary import summary

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

    # print(m_info[0]['model_structure'])
    # print(model)
    input_shape = m_info[1]["00_input_object"]
    # print(input_shape)
    # summary(model, input_shape[1:])
    input_shape = (1,) + input_shape[1:]
    print(input_shape)
    dummy_input = torch.ones(*input_shape)
    print(dummy_input)
    model_onnx_path = "./src/onnx_model/model.onnx"
    # torch.onnx.export(
    #     model, dummy_input, model_onnx_path,
    #     export_params=True,
    #     opset_version=11,
    #     # We define axes as dynamic to allow batch_size > 1
    # )
    
    model = onnx.load(model_onnx_path)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))
    input_sample = torch.ones(*input_shape)
    print(input_sample)
    ort_session = onnxruntime.InferenceSession(model_onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_sample)}
    ort_outputs = ort_session.run(None, ort_inputs)
    