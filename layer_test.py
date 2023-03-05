import torch


m = torch.nn.AvgPool3d((2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), ceil_mode=True,
                    count_include_pad=True)
input = torch.randn(20, 4, 5, 4, 2)
output = m(input)
print(output.shape)