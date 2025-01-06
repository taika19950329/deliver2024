# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd
# from timm.models.layers import DropPath as timDrop
# from fvcore.nn import flop_count_table, FlopCountAnalysis
# from mmcv.cnn import build_norm_layer
#
#
# # class DWConv(nn.Module):
# #     def __init__(self, dim=768):
# #         super(DWConv, self).__init__()
# #         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
# #
# #     def forward(self, x):
# #         x = self.dwconv(x)
# #         return x
# #
# #
# # class Mlp(nn.Module):
# #     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
# #         super().__init__()
# #         out_features = out_features or in_features
# #         hidden_features = hidden_features or in_features
# #         self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
# #         self.dwconv = DWConv(hidden_features)
# #         self.act = act_layer()
# #         self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
# #         self.drop = nn.Dropout(drop)
# #
# #     def forward(self, x):
# #         x = self.fc1(x)
# #
# #         x = self.dwconv(x)
# #         x = self.act(x)
# #         x = self.drop(x)
# #         x = self.fc2(x)
# #         x = self.drop(x)
# #
# #         return x
# #
# #
# # class MSPoolAttention(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         pools = [3, 7, 11]
# #         self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
# #         self.pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0] // 2, count_include_pad=False)
# #         self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1] // 2, count_include_pad=False)
# #         self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2] // 2, count_include_pad=False)
# #         self.conv4 = nn.Conv2d(dim, dim, 1)
# #         self.sigmoid = nn.Sigmoid()
# #
# #     def forward(self, x):
# #         u = x.clone()
# #         x_in = self.conv0(x)
# #         x_1 = self.pool1(x_in)
# #         x_2 = self.pool2(x_in)
# #         x_3 = self.pool3(x_in)
# #         x_out = self.sigmoid(self.conv4(x_in + x_1 + x_2 + x_3)) * u
# #         return x_out + u
# #
# #
# # class MSPABlock(nn.Module):
# #     def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
# #                  norm_cfg=dict(type='BN', requires_grad=True)):
# #         super().__init__()
# #         self.norm1 = build_norm_layer(norm_cfg, dim)[1]
# #         self.attn = MSPoolAttention(dim)
# #         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
# #         self.norm2 = build_norm_layer(norm_cfg, dim)[1]
# #         mlp_hidden_dim = int(dim * mlp_ratio)
# #         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
# #         layer_scale_init_value = 1e-2
# #         self.layer_scale_1 = nn.Parameter(
# #             layer_scale_init_value * torch.ones((dim)), requires_grad=True)
# #         self.layer_scale_2 = nn.Parameter(
# #             layer_scale_init_value * torch.ones((dim)), requires_grad=True)
# #
# #         self.is_channel_mix = True
# #         if self.is_channel_mix:
# #             self.avg_pool = nn.AdaptiveAvgPool2d(1)
# #             self.c_nets = nn.Sequential(
# #                 nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
# #                 nn.Sigmoid())
# #
# #     def forward(self, x):
# #         x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))  # 多尺度特征
# #
# #         if self.is_channel_mix:
# #             x_c = self.avg_pool(x)  # H W 做全局的平均池化
# #             x_c = self.c_nets(x_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 得到每个通道的权重
# #             x_c = x_c.expand_as(x)
# #             x_c_mix = x_c * x  # 重新标定，对通道重要性进行建模
# #             x_mlp = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
# #             x = x_c_mix + x_mlp
# #         else:
# #             x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
# #         return x
#
#
# class MSPDWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(MSPDWConv, self).__init__()
#         self.mspdwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
#
#     def forward(self, x):
#         x = self.mspdwconv(x)
#         return x
#
#
# class MSPMlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.msp_fc1 = nn.Conv2d(in_features, hidden_features, 1)
#         self.msp_dwconv = MSPDWConv(hidden_features)
#         self.msp_act = act_layer()
#         self.msp_fc2 = nn.Conv2d(hidden_features, out_features, 1)
#         self.msp_drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.msp_fc1(x)
#
#         x = self.msp_dwconv(x)
#         x = self.msp_act(x)
#         x = self.msp_drop(x)
#         x = self.msp_fc2(x)
#         x = self.msp_drop(x)
#
#         return x
#
#
# class MSPoolAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         pools = [3, 7, 11]
#         self.msp_conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
#         self.msp_pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0] // 2, count_include_pad=False)
#         self.msp_pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1] // 2, count_include_pad=False)
#         self.msp_pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2] // 2, count_include_pad=False)
#         self.msp_conv4 = nn.Conv2d(dim, dim, 1)
#         self.msp_sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         u = x.clone()
#         x_in = self.msp_conv0(x)
#         x_1 = self.msp_pool1(x_in)
#         x_2 = self.msp_pool2(x_in)
#         x_3 = self.msp_pool3(x_in)
#         x_out = self.msp_sigmoid(self.msp_conv4(x_in + x_1 + x_2 + x_3)) * u
#         return x_out + u
#
#
# class MSPABlock(nn.Module):
#     def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
#                  norm_cfg=dict(type='BN', requires_grad=True)):
#         super().__init__()
#         self.msp_norm1 = build_norm_layer(norm_cfg, dim)[1]
#         self.msp_attn = MSPoolAttention(dim)
#         self.msp_drop_path = timDrop(drop_path) if drop_path > 0. else nn.Identity()
#         self.msp_norm2 = build_norm_layer(norm_cfg, dim)[1]
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.msp_mlp = MSPMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         layer_scale_init_value = 1e-2
#         self.msp_layer_scale_1 = nn.Parameter(
#             layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#         self.msp_layer_scale_2 = nn.Parameter(
#             layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#
#         self.msp_is_channel_mix = True
#         if self.msp_is_channel_mix:
#             self.msp_avg_pool = nn.AdaptiveAvgPool2d(1)
#             self.msp_c_nets = nn.Sequential(
#                 nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
#                 nn.Sigmoid())
#
#     def forward(self, x):
#         x = x + self.msp_drop_path(self.msp_layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.msp_attn(self.msp_norm1(x)))  # 多尺度特征
#
#         if self.msp_is_channel_mix:
#             x_c = self.msp_avg_pool(x)  # H W 做全局的平均池化
#             x_c = self.msp_c_nets(x_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 得到每个通道的权重
#             x_c = x_c.expand_as(x)
#             x_c_mix = x_c * x  # 重新标定，对通道重要性进行建模
#             x_mlp = self.msp_drop_path(self.msp_layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.msp_mlp(self.msp_norm2(x)))
#             x = x_c_mix + x_mlp
#         else:
#             x = x + self.msp_drop_path(self.msp_layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.msp_mlp(self.msp_norm2(x)))
#         return x
#
#
# # if __name__ == '__main__':
# #     x = torch.zeros(2, 224 * 224, 64)
# #     c1 = MSDyBlock(64, 64)
# #     outs = c1(x)
# #     print(outs.shape)
# #     print(c1)
# #     print(flop_count_table(FlopCountAnalysis(c1, x)))
#
#
# if __name__ == '__main__':
#     x = torch.ones(2, 224, 224, 64)
#     c1 = MSPABlock(dim=224, mlp_ratio=4)
#     outs = c1(x)
#     # print(outs.shape)
#     # raise Exception
#     print(c1)
#     print(flop_count_table(FlopCountAnalysis(c1, x)))
#     print('done')
#
#
#
#





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from timm.models.layers import DropPath
from fvcore.nn import flop_count_table, FlopCountAnalysis
from mmcv.cnn import build_norm_layer


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class MSPoolAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        pools = [3, 7, 11]
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0] // 2, count_include_pad=False)
        self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1] // 2, count_include_pad=False)
        self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2] // 2, count_include_pad=False)
        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        x_in = self.conv0(x)
        x_1 = self.pool1(x_in)
        x_2 = self.pool2(x_in)
        x_3 = self.pool3(x_in)
        x_out = self.sigmoid(self.conv4(x_in + x_1 + x_2 + x_3)) * u
        return x_out + u


class MSPABlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = MSPoolAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.is_channel_mix = True
        if self.is_channel_mix:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.c_nets = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))  # 多尺度特征

        if self.is_channel_mix:
            x_c = self.avg_pool(x)  # H W 做全局的平均池化
            x_c = self.c_nets(x_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 得到每个通道的权重
            x_c = x_c.expand_as(x)
            x_c_mix = x_c * x  # 重新标定，对通道重要性进行建模
            x_mlp = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
            x = x_c_mix + x_mlp
        else:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


# if __name__ == '__main__':
#     x = torch.zeros(2, 224 * 224, 64)
#     c1 = MSDyBlock(64, 64)
#     outs = c1(x)
#     print(outs.shape)
#     print(c1)
#     print(flop_count_table(FlopCountAnalysis(c1, x)))


if __name__ == '__main__':
    x = [torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    c1 = MSPABlock(dim=128, weight_h=128, mlp_ratio=4)
    outs = c1(x)
    # print(outs.shape)
    # raise Exception
    print(c1)
    print(flop_count_table(FlopCountAnalysis(c1, x)))
    print('done')



