import torch
from torch import nn
from torchsummary import summary
import onnx
import onnxruntime



class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv2d(2, 19, 13, padding='same')
        # self.linear = nn.Linear(19 * 32 * 32, 10)
        self.conv1 = nn.Conv2d(2, 19, 13, padding='valid')
        self.linear = nn.Linear(19 * 20 * 20, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


def main():
    model = CNN()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    input_shape = (2, 32, 32)
    model.to(device)
    summary(model, input_shape)

    input_shape = (1,) + input_shape
    dummy_input = torch.ones(*input_shape).to(device)
    model_onnx_path = "./src/onnx_model/model.onnx"

    torch.onnx.export(
        model, dummy_input, model_onnx_path,
        export_params=True,
        opset_version=14,
    )
    

if __name__ == "__main__":
    main()