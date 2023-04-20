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
from utils.tool import extract_inter_output
from utils.tool import extract_inter_output_pytorch
from utils.tool import localize_buggy_layer


IDs =['000110']
thres = 0.01
# given the list of suspecious PyTorch model ID
# aim to localize the first inconsistent layer
incons_dic = {} # store the inconsistent indices of each suspecious model ID
for id in IDs:
    model_info_path = '../../../Jingyu/fyp_dl_library_testing/data/dummy_output/'+ str(id)+'/models/model.json'
    training_inputs_path = '../../../Jingyu/fyp_dl_library_testing/data/dummy_output/'+str(id)+'/dataset/inputs.npz' 
    training_inputs = [*np.load(training_inputs_path).values()]
    input = training_inputs[0]
    dic = extract_inter_output(model_info_path, input)
    incons_dic[str(id)] = localize_buggy_layer(dic, thres = thres)
print(incons_dic.items())


# # path to the data
# model_info_path = '../../../Jingyu/fyp_dl_library_testing/data/dummy_output/000110/models/model.json'
# training_inputs_path = '../../../Jingyu/fyp_dl_library_testing/data/dummy_output/000110/dataset/inputs.npz'
# ground_truths_path = '../../../Jingyu/fyp_dl_library_testing/data/dummy_output/000110/dataset/ground_truths.npz'

# training_inputs = [*np.load(training_inputs_path).values()]
# input = training_inputs[0]
# with open(model_info_path, 'r') as f:
#         m_info = json.load(f)
# print("input shape {}".format(input.shape))
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
# py_model = TorchModel(m_info)
# py_model = py_model.to(device)

# py_input = torch.from_numpy(input).float().to(device)
# print(type(py_model(py_input)))
# p = py_model(py_input)
# value  =next(iter(p.values()))
# print(np.shape(value))
# for key, value in py_model(py_input).items():
#     print(key, ':', value)
# pytorch_output = extract_inter_output_pytorch(py_model,py_input)
# py_list = list(pytorch_output.keys())

# dic = extract_inter_output(model_info_path, input)
# print(dic['pytorch_output'][-1])
# print(dic['onnx_output'][-1])

# y = [*np.load(ground_truths_path).values()][0]
# print("ground shape: {}".format(np.shape(y)))
# onnx = dic['onnx_output'][-1]
# tf = dic['tf_output'][-1]
# print(np.shape(onnx))
# print(np.shape(tf))

# print("py_len: {}".format(len(py_list)))
# print("py_len: {}".format(len(dic['pytorch_output'])))
# print("onnx_len: {}".format(len(dic['onnx_output'])))
# print("tf_len: {}".format(len(dic['tf_output'])))

# for i in range(4):
#     print("py shape: {}".format(np.shape(dic['pytorch_output'][i])))
#     print("onnx shape: {}".format(np.shape(dic['onnx_output'][i])))
#     print("tf shape: {}".format(np.shape(dic['tf_output'][i])))

print(localize_buggy_layer(dic, thres = 10))
    
# output_list = extract_inter_output_tensorflow(model_info_path, input)
# for i in range(len(output_list)):
#     print(output_list[i].shape)
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