import torch
import torch.nn as nn

class ConcatAndConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConcatAndConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, tensor_list):
        # 1. 在通道维度上拼接张量
        concatenated_tensor = torch.cat(tensor_list, dim=1)  # 在通道维度上拼接

        # 2. 应用卷积层
        output_tensor = self.conv(concatenated_tensor)

        return output_tensor

if "__main__"==__name__:
    tensor_list = [torch.randn(2, 32, 256, 256), torch.randn(2, 32, 256, 256), torch.randn(2, 32, 256, 256)]
    in_channels = len(tensor_list) * tensor_list[0].shape[1]  # 输入通道数是列表中所有张量通道数的总和
    out_channels = 32  # 期望输出通道数

    model = ConcatAndConv(in_channels, out_channels)
    output_tensor = model(tensor_list)

    print(output_tensor.shape)  # 输出形状应该是 (2, 32, 256, 256)
