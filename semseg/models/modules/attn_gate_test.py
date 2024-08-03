import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionWeightedSum(nn.Module):
    def __init__(self):
        super(AttentionWeightedSum, self).__init__()

    def forward(self, tensor_list, reference_tensor):
        # 初始化一个张量来存储加权结果
        weighted_tensors = []

        # 计算每个张量的注意力权重并应用
        for tensor in tensor_list:
            # 计算注意力权重（点积）
            attention_scores = torch.sum(reference_tensor * tensor, dim=(1, 2, 3), keepdim=True)

            # 归一化权重（使用 softmax）
            attention_weights = F.softmax(attention_scores, dim=0)

            # 应用权重
            weighted_tensor = tensor * attention_weights
            weighted_tensors.append(weighted_tensor)


        return weighted_tensors


if "__main__"==__name__:
    tensor_list = [torch.randn(2, 32, 256, 256), torch.randn(2, 32, 256, 256), torch.randn(2, 32, 256, 256)]
    reference_tensor = torch.randn(2, 32, 256, 256)

    attention_module = AttentionWeightedSum()
    output_tensor = attention_module(tensor_list, reference_tensor)
    for each in output_tensor:
        print(each.shape)

