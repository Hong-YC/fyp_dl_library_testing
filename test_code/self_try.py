import torch
import onnx
import onnxruntime
import json
from src.cases_generation.torch_model import TorchModel
from src.cases_generation.model_info_generator import ModelInfoGenerator
# from utils.tool import to_numpy
from utils.db_manager import DbManager
from pathlib import Path
from torchsummary import summary
import numpy as np
# from dataset import TrainDataset
# from dataset import ToTensor
from utils.tool import extract_inter_output_tensorflow

model_info_path = './data/dummy_output/000007/models/model.json'
training_inputs_path = './data/dummy_output/000007/dataset/inputs.npz'
ground_truths_path = './data/dummy_output/000007/dataset/ground_truths.npz'

training_inputs = [*np.load(training_inputs_path).values()]
input = training_inputs[0]
output_list = extract_inter_output_tensorflow(model_info_path, input)
for i in range(len(output_list)):
    print(output_list[i].shape)
# haha = np.array(output_list[1])
# print(type(haha))





# # Load the model info
# with open(model_info_path, 'r') as f:
#     m_info = json.load(f)

# # print(m_info)

# dic = m_info["model_structure"]
# model_info = m_info.copy()
# # for key in dic:
# #     print(dic[key])
# key = '1'
# first_two = dict(list(dic.items())[:int(key)])
# print(first_two)
# # m_info['model_structure'] = 
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
# model_info['model_structure'] = first_two
# model_info['output_id_list'] = list(key)

# model = TorchModel(model_info)
# print(model)
# h = '0'
# print(h=='0')


# {'model_structure': {'0': {'type': 'input_object', 'args': {'shape': [15, 8], 'batch_shape': None, 'name': '00_input_object'}, 'pre_layers': [], 'output_shape': [None, 15, 8]}, '1': {'type': 'MaxPool1d', 'args': {'kernel_size': 4, 'stride': 7, 'padding': 1, 'dilation': 1, 'ceil_mode': False, 'name': '01_MaxPool1d'}, 'pre_layers': ['0'], 'output_shape': [None, 15, 1]}, '2': {'type': 'Flatten', 'args': {'start_dim': 1, 'end_dim': -1, 'name': '02_Flatten'}, 'pre_layers': ['1'], 'output_shape': [None, 15]}, '3': {'type': 'Linear', 'args': {'in_features': 15, 'out_features': 48, 'bias': False, 'name': '03_Linear'}, 'pre_layers': ['2'], 'output_shape': [None, 48]}, '4': {'type': 'reshape', 'args': {'shape': [-1, 8, 6], 'name': '04_reshape'}, 'pre_layers': ['3'], 'output_shape': [None, 8, 6]}}, 'input_id_list': ['0'], 'output_id_list': ['4']}