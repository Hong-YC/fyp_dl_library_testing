import torch
import onnx
import onnxruntime
import json
from src.cases_generation.torch_model import TorchModel
from src.cases_generation.model_info_generator import ModelInfoGenerator
from utils.tool import to_numpy
from utils.db_manager import DbManager
from pathlib import Path
from torchsummary import summary

if __name__ == '__main__':
    config = {
        'var': {
            'tensor_dimension_range': (5,5),
            'tensor_element_size_range': (2, 64),
            'weight_value_range' : (5,6),
            'small_value_range' : (1,3)
        },
        'node_num_range': (5, 5),
    }

    db_manager = DbManager(str(Path.cwd() / 'data' / 'dummy.db'))

    m_info_generator = ModelInfoGenerator(config, 'seq', db_manager)

    m_info = m_info_generator.generate_seq_model(5, output_shape=(None, 3, 4), element = "FractionalMaxPool3d")

    # print('?????????????',m_info)
    with open(str(Path.cwd() / 'models/dummy_model.json'), 'w') as f:
        json.dump(m_info[0], f)

    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    
    model = TorchModel(m_info[0])
    print(model)
    # input_shape = m_info[1]["00_input_object"]
  

    # input_shape = input_shape[1:]
    # print("Input shape: ", input_shape)

    # model.to(device)
    # summary(model, input_shape)

    # ======================================================================
    # input_shape = (1,) + input_shape
    # dummy_input = torch.ones(*input_shape).to(device)
    # # print(dummy_input)
    # model_onnx_path = "./src/onnx_model/model.onnx"

    # torch.onnx.export(
    #     model, dummy_input, model_onnx_path,
    #     export_params=True,
    #     opset_version=11,  # version can be >=7 <=16
    #     # We define axes as dynamic to allow batch_size > 1
    # )
    
    # model = onnx.load(model_onnx_path)
    # onnx.checker.check_model(model)
    # print(onnx.helper.printable_graph(model.graph))
    # input_sample = torch.ones(*input_shape).to(device)
    # print(input_sample.shape)
    # ort_session = onnxruntime.InferenceSession(model_onnx_path)
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_sample)}
    # ort_outputs = ort_session.run(None, ort_inputs)
    
