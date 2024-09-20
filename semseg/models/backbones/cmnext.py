import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath
import functools
from functools import partial
from fvcore.nn import flop_count_table, FlopCountAnalysis
from semseg.models.modules.ffm import FeatureFusionModule as FFM
from semseg.models.modules.ffm import FeatureRectifyModule as FRM
from semseg.models.modules.ffm import ChannelEmbed
from semseg.models.modules.mspa import MSPABlock
from semseg.utils.utils import nchw_to_nlc, nlc_to_nchw

from semseg.models.modules.moe_lora import MoE_lora_new, AttentionWeightedSum, ConcatAndConv, \
    FinalConvProcessor
# from segment_anything import sam_model_registry
# from semseg.models.modules.sam_lora import *
# from semseg.models.modules.BasicBlock import TF_3D

# import numpy as np
# from copy import deepcopy
#
# import torch
# import psutil
# import os
#
# # 获取当前进程的 PID
# pid = os.getpid()
# # 获取当前进程的内存信息
# process = psutil.Process(pid)
# #
# # import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler


# class Attention(nn.Module):
#     def __init__(self, dim, head, sr_ratio):
#         super().__init__()
#         self.head = head
#         self.sr_ratio = sr_ratio
#         self.scale = (dim // head) ** -0.5
#         self.q = nn.Linear(dim, dim)
#         self.kv = nn.Linear(dim, dim * 2)
#         self.proj = nn.Linear(dim, dim)
#
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x: Tensor, H, W) -> Tensor:
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
#
#         if self.sr_ratio > 1:
#             x = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
#             x = self.norm(x)
#
#         k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         return x

class _LoRA_q(nn.Module):
    def __init__(self, q: nn.Module, linear_a_q: nn.Module, linear_b_q: nn.Module):
        super().__init__()
        self.q = q
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.dim = q.in_features

    def forward(self, x):
        q = self.q(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        return q + new_q


class _LoRA_kv(nn.Module):
    def __init__(self, kv: nn.Module, linear_a_k: nn.Module, linear_b_k: nn.Module, linear_a_v: nn.Module, linear_b_v: nn.Module):
        super().__init__()
        self.kv = kv
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = kv.in_features

    def forward(self, x):
        kv = self.kv(x)
        k, v = kv.chunk(2, dim=-1)
        new_k = self.linear_b_k(self.linear_a_k(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        k = k + new_k
        v = v + new_v
        return torch.cat((k, v), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio, r):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5

        # Original Q and KV linear layers
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        # LoRA for rgb
        self.lora_rgb_a_q = nn.Linear(dim, r, bias=False)
        self.lora_rgb_b_q = nn.Linear(r, dim, bias=False)
        self.lora_rgb_a_v = nn.Linear(dim, r, bias=False)
        self.lora_rgb_b_v = nn.Linear(r, dim, bias=False)

        self.lora_rgb_q = _LoRA_q(
            self.q,
            self.lora_rgb_a_q,
            self.lora_rgb_b_q
        )
        self.lora_rgb_kv = _LoRA_kv(
            self.kv,
            self.lora_rgb_a_v,
            self.lora_rgb_b_v,
            self.lora_rgb_a_v,  # Reusing the same for k and v
            self.lora_rgb_b_v
        )

        # LoRA for X
        self.lora_X_a_q = nn.Linear(dim, r, bias=False)
        self.lora_X_b_q = nn.Linear(r, dim, bias=False)
        self.lora_X_a_v = nn.Linear(dim, r, bias=False)
        self.lora_X_b_v = nn.Linear(r, dim, bias=False)

        self.lora_X_q = _LoRA_q(
            self.q,
            self.lora_X_a_q,
            self.lora_X_b_q
        )
        self.lora_X_kv = _LoRA_kv(
            self.kv,
            self.lora_X_a_v,
            self.lora_X_b_v,
            self.lora_X_a_v,  # Reusing the same for k and v
            self.lora_X_b_v
        )

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W, type) -> Tensor:
        B, N, C = x.shape

        # Apply LoRA to Q
        if type == 'rgb':
            q = self.lora_rgb_q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        elif type == 'X':
            q = self.lora_X_q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # Apply spatial reduction and normalization
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        # Apply LoRA to KV
        if type == 'rgb':
            k, v = self.lora_rgb_kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        elif type == 'X':
            k, v = self.lora_X_kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


# class MLP(nn.Module):
#     def __init__(self, c1, c2):
#         super().__init__()
#         self.fc1 = nn.Linear(c1, c2)
#         self.dwconv = DWConv(c2)
#         self.fc2 = nn.Linear(c2, c1)
#
#     def forward(self, x: Tensor, H, W) -> Tensor:
#         return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class MLP(nn.Module):
    def __init__(self, c1, c2, r):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.fc2 = nn.Linear(c2, c1)
        self.dwconv = DWConv(c2)

        # LoRA layers for rgb
        self.lora_rgb_a_fc1 = nn.Linear(c1, r, bias=False)
        self.lora_rgb_b_fc1 = nn.Linear(r, c2, bias=False)
        self.lora_rgb_a_fc2 = nn.Linear(c2, r, bias=False)
        self.lora_rgb_b_fc2 = nn.Linear(r, c1, bias=False)

        # LoRA layers for X
        self.lora_X_a_fc1 = nn.Linear(c1, r, bias=False)
        self.lora_X_b_fc1 = nn.Linear(r, c2, bias=False)
        self.lora_X_a_fc2 = nn.Linear(c2, r, bias=False)
        self.lora_X_b_fc2 = nn.Linear(r, c1, bias=False)


    def forward(self, x: Tensor, H, W, type) -> Tensor:
        # Original fc1 output
        out_fc1 = self.fc1(x)
        # LoRA adjustment to fc1
        if type == 'rgb':
            out_fc1_lora = self.lora_rgb_b_fc1(self.lora_rgb_a_fc1(x))
        elif type == 'X':
            out_fc1_lora = self.lora_X_b_fc1(self.lora_X_a_fc1(x))

        out_fc1 = out_fc1 + out_fc1_lora  # Combine original and LoRA

        # Apply depth-wise convolution
        out_dwconv = self.dwconv(out_fc1, H, W)

        # Activation
        out_act = F.gelu(out_dwconv)

        # Original fc2 output
        out_fc2 = self.fc2(out_act)
        # LoRA adjustment to fc2
        if type == 'rgb':
            out_fc2_lora = self.lora_rgb_b_fc2(self.lora_rgb_a_fc2(out_act))
        elif type == 'X':
            out_fc2_lora = self.lora_X_b_fc2(self.lora_X_a_fc2(out_act))
        out_fc2 = out_fc2 + out_fc2_lora  # Combine original and LoRA

        return out_fc2



class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)  # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, num_modals=4):
        super().__init__()
        self.proj = ModuleParallel(nn.Conv2d(c1, c2, patch_size, stride, padding))  # padding=(ps[0]//2, ps[1]//2)
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: list) -> list:
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = self.norm(x)
        return x, H, W


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., is_fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio, 4)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4), 4) if not is_fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim*4))

    def forward(self, x: Tensor, H, W, type) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, type))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W, type))
        return x


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1, -2)
        _, _, Nk, Ck = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))

        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(
            1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x


class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        ) for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]


class ConcatAndReduceChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: 每个输入 Tensor 的通道数 C
        :param out_channels: 降维后的通道数 C
        """
        super(ConcatAndReduceChannels, self).__init__()

        # 使用 1x1 卷积来将 3C 通道降到 C
        self.conv = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)

    def forward(self, x_ext):
        """
        :param x_ext: 一个包含三个形状为 [B, C, H, W] 的 Tensor 的列表
        :return: 输出为一个形状为 [B, C, H, W] 的 Tensor
        """
        # 在通道维度 (dim=1) 上连接 3 个张量
        x = torch.cat(x_ext, dim=1)  # 输出的形状将是 [B, 3C, H, W]

        # 使用 1x1 卷积将通道数从 3C 降到 C
        x = self.conv(x)

        return x


cmnext_settings = {
    # 'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    # 'B2': [[64, 128, 320, 512], [2, 2, 2, 2]],
    # 'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class CMNeXt(nn.Module):
    # def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
    def __init__(self, weight_h_ori, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):  ######
        super().__init__()
        assert model_name in cmnext_settings.keys(), f"Model name should be in {list(cmnext_settings.keys())}"
        embed_dims, depths = cmnext_settings[model_name]
        extra_depths = depths
        self.modals = modals[1:] if len(modals) > 1 else []
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        norm_cfg = dict(type='BN', requires_grad=True)
        # print('backbone cmnext weight_h_ori', weight_h_ori)

        if self.num_modals > 0:  ######
            self.moe1 = MoE_lora_new(3, embed_dims[0], 7, 4, 7 // 2, 6, 1024, True, 2)
            self.moe2 = MoE_lora_new(64, embed_dims[1], 3, 2, 3 // 2, 6, 256, True, 2)
            self.moe3 = MoE_lora_new(128, embed_dims[2], 3, 2, 3 // 2, 6, 128, True, 2)
            self.moe4 = MoE_lora_new(320, embed_dims[3], 3, 2, 3 // 2, 6, 64, True, 2)

        self.attn_gate1 = AttentionWeightedSum()
        self.attn_gate2 = AttentionWeightedSum()
        self.attn_gate3 = AttentionWeightedSum()
        self.attn_gate4 = AttentionWeightedSum()

        self.concat_conv1 = ConcatAndConv(3 * embed_dims[0], embed_dims[0])  ######
        self.concat_conv2 = ConcatAndConv(3 * embed_dims[1], embed_dims[1])
        self.concat_conv3 = ConcatAndConv(3 * embed_dims[2], embed_dims[2])
        self.concat_conv4 = ConcatAndConv(3 * embed_dims[3], embed_dims[3])

        # self.final_conv1 = FinalConvProcessor(embed_dims[0], embed_dims[0])
        # self.final_conv2 = FinalConvProcessor(embed_dims[1], embed_dims[1])
        # self.final_conv3 = FinalConvProcessor(embed_dims[2], embed_dims[2])
        # self.final_conv4 = FinalConvProcessor(embed_dims[3], embed_dims[3])

        self.concat_reduce_channels1 = ConcatAndReduceChannels(3, 3)
        self.concat_reduce_channels2 = ConcatAndReduceChannels(embed_dims[0], embed_dims[0])
        self.concat_reduce_channels3 = ConcatAndReduceChannels(embed_dims[1], embed_dims[1])
        self.concat_reduce_channels4 = ConcatAndReduceChannels(embed_dims[2], embed_dims[2])

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7 // 2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3 // 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3 // 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3 // 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        if self.num_modals > 0:
            num_heads = [1, 2, 5, 8]
            self.FRMs = nn.ModuleList([
                FRM(dim=embed_dims[0], reduction=1),
                FRM(dim=embed_dims[1], reduction=1),
                FRM(dim=embed_dims[2], reduction=1),
                FRM(dim=embed_dims[3], reduction=1)])
            self.FFMs = nn.ModuleList([
                FFM(dim=embed_dims[0], reduction=1, num_heads=num_heads[0], norm_layer=nn.BatchNorm2d),
                FFM(dim=embed_dims[1], reduction=1, num_heads=num_heads[1], norm_layer=nn.BatchNorm2d),
                FFM(dim=embed_dims[2], reduction=1, num_heads=num_heads[2], norm_layer=nn.BatchNorm2d),
                FFM(dim=embed_dims[3], reduction=1, num_heads=num_heads[3], norm_layer=nn.BatchNorm2d)])

        # 冻结参数debug
        # for layer in [self.block1, self.block2, self.block3, self.block4, self.norm1, self.norm2, self.norm3, self.norm4,
        #               self.FRMs, self.FFMs,
        #               self.moe1, self.moe2, self.moe3, self.moe4,
        #               self.concat_conv1, self.concat_conv2, self.concat_conv3, self.concat_conv4,
        #               self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4]:
        #     for param in layer.parameters():
        #         param.requires_grad = False

    def tokenselect(self, x_ext, module):
        x_scores = module(x_ext)
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
        x_f = functools.reduce(torch.max, x_ext)
        return x_f

    def forward(self, x: list) -> list:  ######
        x_cam = x[0]
        if self.num_modals > 0:
            x_ext = x[1:]

        B = x_cam.shape[0]
        outs = []

        # ------ stage 1 ------ #
        ## ------ rgb encoder lora process ------ ##
        # print_memory_usage("Before stage1")
        x_cam, H, W = self.patch_embed1(x_cam)
        for blk in self.block1:
            x_cam = blk(x_cam, H, W, 'rgb')
        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        del x_cam
        torch.cuda.empty_cache()

        if self.num_modals > 0:
            ## ------ MeMe ------ ##
            x_ext_moe, loss_moe1 = self.moe1(x_ext)

            ## ------ diff feature encoder lora process ------ ##
            x_f = self.concat_reduce_channels1(x_ext)
            x_f, _, _ = self.patch_embed1(x_f)
            for blk in self.block1:
                x_f = blk(x_f, H, W, 'X')

            x1_f = self.norm1(x_f + x_ext_moe[0].flatten(2).transpose(1, 2) + x_ext_moe[1].flatten(2).transpose(1, 2)
                              + x_ext_moe[2].flatten(2).transpose(1, 2)).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext = x_ext_moe

            del x_ext_moe, x_f
            torch.cuda.empty_cache()

            ## ------ rgb & X_share fusion ------ ##
            x1_cam, x1_f = self.FRMs[0](x1_cam, x1_f)
            x_fused = self.FFMs[0](x1_cam, x1_f)

            ## ------ rgb & X_diff fusion ------ ##
            x_ext_attn = self.attn_gate1(x_ext, x_fused)
            # expert_combine_output = self.concat_conv1(x_ext_attn)
            # final_fused = self.final_conv1(expert_combine_output, x_fused)
            final_fused = self.concat_conv1(x_ext_attn)
            outs.append(final_fused)

            del final_fused, x_ext_attn
            torch.cuda.empty_cache()
        else:
            outs.append(x1_cam)

        # ------ stage 2 ------ #
        ## ------ rgb encoder lora process ------ ##
        # print_memory_usage("Before stage2")
        x1_cam, H, W = self.patch_embed2(x1_cam)
        for blk in self.block2:
            x1_cam = blk(x1_cam, H, W, 'rgb')
        x2_cam = self.norm2(x1_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        del x1_cam
        torch.cuda.empty_cache()

        if self.num_modals > 0:
            ## ------ MeMe ------ ##
            x_ext_moe, loss_moe2 = self.moe2(x_ext)

            ## ------ diff feature encoder lora process ------ ##
            x_f = self.concat_reduce_channels2(x_ext)
            x_f, _, _ = self.patch_embed2(x_f)
            for blk in self.block2:
                x_f = blk(x_f, H, W, 'X')

            x2_f = self.norm2(x_f + x_ext_moe[0].flatten(2).transpose(1, 2) + x_ext_moe[1].flatten(2).transpose(1, 2)
                              + x_ext_moe[2].flatten(2).transpose(1, 2)).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext = x_ext_moe

            del x_ext_moe, x_f
            torch.cuda.empty_cache()

            ## ------ rgb & X_share fusion ------ ##
            x2_cam, x2_f = self.FRMs[1](x2_cam, x2_f)
            x_fused = self.FFMs[1](x2_cam, x2_f)

            ## ------ rgb & X_diff fusion ------ ##
            x_ext_attn = self.attn_gate2(x_ext, x_fused)
            # expert_combine_output = self.concat_conv2(x_ext_attn)
            final_fused = self.concat_conv2(x_ext_attn)
            outs.append(final_fused)

            del final_fused, x_ext_attn
            torch.cuda.empty_cache()
        else:
            outs.append(x2_cam)

        # ------ stage 3 ------ #
        ## ------ rgb encoder lora process ------ ##
        # print_memory_usage("Before stage3")
        x2_cam, H, W = self.patch_embed3(x2_cam)
        for blk in self.block3:
            x2_cam = blk(x2_cam, H, W, 'rgb')
        x3_cam = self.norm3(x2_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        del x2_cam
        torch.cuda.empty_cache()

        if self.num_modals > 0:
            ## ------ MeMe ------ ##
            x_ext_moe, loss_moe3 = self.moe3(x_ext)

            ## ------ diff feature encoder lora process ------ ##
            x_f = self.concat_reduce_channels3(x_ext)
            x_f, _, _ = self.patch_embed3(x_f)
            for blk in self.block3:
                x_f = blk(x_f, H, W, 'X')

            x3_f = self.norm3(x_f + x_ext_moe[0].flatten(2).transpose(1, 2) + x_ext_moe[1].flatten(2).transpose(1, 2)
                              + x_ext_moe[2].flatten(2).transpose(1, 2)).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext = x_ext_moe

            del x_ext_moe, x_f
            torch.cuda.empty_cache()

            ## ------ rgb & X_share fusion ------ ##
            x3_cam, x3_f = self.FRMs[2](x3_cam, x3_f)
            x_fused = self.FFMs[2](x3_cam, x3_f)

            ## ------ rgb & X_diff fusion ------ ##
            x_ext_attn = self.attn_gate3(x_ext, x_fused)
            # expert_combine_output = self.concat_conv3(x_ext_attn)
            final_fused = self.concat_conv3(x_ext_attn)
            outs.append(final_fused)

            del final_fused, x_ext_attn
            torch.cuda.empty_cache()
        else:
            outs.append(x3_cam)

        # ------ stage 4 ------ #
        ## ------ rgb encoder lora process ------ ##
        # print_memory_usage("Before stage4")
        x3_cam, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x3_cam = blk(x3_cam, H, W, 'rgb')
        x4_cam = self.norm4(x3_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        del x3_cam
        torch.cuda.empty_cache()

        if self.num_modals > 0:
            ## ------ MeMe ------ ##
            x_ext_moe, loss_moe4 = self.moe4(x_ext)

            ## ------ diff feature encoder lora process ------ ##
            x_f = self.concat_reduce_channels4(x_ext)
            x_f, _, _ = self.patch_embed4(x_f)
            for blk in self.block4:
                x_f = blk(x_f, H, W, 'X')

            x4_f = self.norm4(x_f + x_ext_moe[0].flatten(2).transpose(1, 2) + x_ext_moe[1].flatten(2).transpose(1, 2)
                              + x_ext_moe[2].flatten(2).transpose(1, 2)).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext = x_ext_moe

            del x_ext_moe, x_f
            torch.cuda.empty_cache()

            ## ------ rgb & X_share fusion ------ ##
            x4_cam, x4_f = self.FRMs[3](x4_cam, x4_f)
            x_fused = self.FFMs[3](x4_cam, x4_f)

            ## ------ rgb & X_diff fusion ------ ##
            x_ext_attn = self.attn_gate4(x_ext, x_fused)
            # expert_combine_output = self.concat_conv4(x_ext_attn)
            final_fused = self.concat_conv4(x_ext_attn)
            outs.append(final_fused)

            del final_fused, x_ext_attn
            torch.cuda.empty_cache()
        else:
            outs.append(x4_cam)

        return outs, loss_moe1, loss_moe2, loss_moe3, loss_moe4  ######
        # return outs


def print_memory_usage(step=""):
    # GPU 显存使用情况
    if torch.cuda.is_available():
        print(f"[{step}] GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"[{step}] GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    # CPU 内存使用情况
    mem_info = process.memory_info()
    print(f"[{step}] CPU Memory Used: {mem_info.rss / 1024 ** 2:.2f} MB")


# if __name__ == '__main__':
#     modals = ['rgb', 'depth', 'event', 'lidar']
#     device = torch.device('cuda')
#     x = [torch.ones(2, 3, 1024, 1024).to(device), torch.ones(2, 3, 1024, 1024).to(device), (torch.ones(2, 3, 1024, 1024) * 2).to(device),
#          (torch.ones(2, 3, 1024, 1024) * 3).to(device)]
#     # print(int(x[0].shape[2]/4))
#     # raise Exception
#     model = CMNeXt(int(x[0].shape[2] / 4), 'B2', modals).to(device)
#
#     # total = 0
#     # for name, param in model.named_parameters():
#     #     if param.requires_grad:
#     #         layer_params_m = param.numel() / 1e6
#     #         total += layer_params_m
#     #         print(f"层 {name} 参数量: {param.numel()} ({layer_params_m:.6f} M)")
#     # print(f" ({total:.6f} M)")
#     #
#     # raise Exception
#
#     outs = model(x)
#     for y in outs[0]:
#         print(y.shape)
#     print(outs[1:])

if __name__ == '__main__':
    modals = ['rgb', 'depth', 'event', 'lidar']
    device = torch.device('cpu')  # 使用 GPU
    x = [torch.ones(2, 3, 1024, 1024).to(device),
         torch.ones(2, 3, 1024, 1024).to(device),
         (torch.ones(2, 3, 1024, 1024) * 2).to(device),
         (torch.ones(2, 3, 1024, 1024) * 3).to(device)]

    model = CMNeXt(int(x[0].shape[2] / 4), 'B2', modals).to(device)

    # 初始化 GradScaler
    scaler = GradScaler()

    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    for epoch in range(1):  # 只做一个 epoch 作为示例
        # optimizer.zero_grad()

        # 使用混合精度训练
        with autocast():
            outs = model(x)
            # 假设使用一个简单的损失函数
        #     loss = sum(outs[1:])
        #
        # # 使用 GradScaler 缩放损失并进行反向传播
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # 打印输出的形状
        for y in outs[0]:
            print(y.shape)
        print(outs[1:])

