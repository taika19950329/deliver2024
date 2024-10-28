import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Variable
from semseg.models.layers import DropPath
import functools
from functools import partial
from fvcore.nn import flop_count_table, FlopCountAnalysis
from semseg.models.modules.ffm import FeatureFusionModule as FFM
from semseg.models.modules.ffm import FeatureRectifyModule as FRM
from semseg.models.modules.ffm import ChannelEmbed
from semseg.utils.utils import nchw_to_nlc, nlc_to_nchw

from semseg.models.modules.moe_lora import ConcatAndConv
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
#
# import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler


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
    def __init__(self, q: nn.Module, linear_a_q: nn.Module, linear_b_q: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.q = q
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.dim = q.in_features
        self.alpha = alpha  # 加入缩放因子

    def forward(self, x):
        q = self.q(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        return q + self.alpha * new_q  # 乘以缩放因子

class _LoRA_kv(nn.Module):
    def __init__(self, kv: nn.Module, linear_a_k: nn.Module, linear_b_k: nn.Module, linear_a_v: nn.Module, linear_b_v: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.kv = kv
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = kv.in_features
        self.alpha = alpha  # 加入缩放因子

    def forward(self, x):
        kv = self.kv(x)
        k, v = kv.chunk(2, dim=-1)
        new_k = self.linear_b_k(self.linear_a_k(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        k = k + self.alpha * new_k  # 乘以缩放因子
        v = v + self.alpha * new_v  # 乘以缩放因子
        return torch.cat((k, v), dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio, r, alpha=1.0):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.alpha = alpha  # 缩放因子

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
            self.lora_rgb_b_q,
            self.alpha  # 使用缩放因子
        )
        self.lora_rgb_kv = _LoRA_kv(
            self.kv,
            self.lora_rgb_a_v,
            self.lora_rgb_b_v,
            self.lora_rgb_a_v,  # Reusing the same for k and v
            self.lora_rgb_b_v,
            self.alpha  # 使用缩放因子
        )

        # LoRA for depth
        self.lora_depth_a_q = nn.Linear(dim, r, bias=False)
        self.lora_depth_b_q = nn.Linear(r, dim, bias=False)
        self.lora_depth_a_v = nn.Linear(dim, r, bias=False)
        self.lora_depth_b_v = nn.Linear(r, dim, bias=False)

        self.lora_depth_q = _LoRA_q(
            self.q,
            self.lora_depth_a_q,
            self.lora_depth_b_q,
            self.alpha
        )
        self.lora_depth_kv = _LoRA_kv(
            self.kv,
            self.lora_depth_a_v,
            self.lora_depth_b_v,
            self.lora_depth_a_v,  # Reusing the same for k and v
            self.lora_depth_b_v,
            self.alpha
        )

        # LoRA for event
        self.lora_event_a_q = nn.Linear(dim, r, bias=False)
        self.lora_event_b_q = nn.Linear(r, dim, bias=False)
        self.lora_event_a_v = nn.Linear(dim, r, bias=False)
        self.lora_event_b_v = nn.Linear(r, dim, bias=False)

        self.lora_event_q = _LoRA_q(
            self.q,
            self.lora_event_a_q,
            self.lora_event_b_q,
            self.alpha
        )
        self.lora_event_kv = _LoRA_kv(
            self.kv,
            self.lora_event_a_v,
            self.lora_event_b_v,
            self.lora_event_a_v,  # Reusing the same for k and v
            self.lora_event_b_v,
            self.alpha
        )

        # LoRA for lidar
        self.lora_lidar_a_q = nn.Linear(dim, r, bias=False)
        self.lora_lidar_b_q = nn.Linear(r, dim, bias=False)
        self.lora_lidar_a_v = nn.Linear(dim, r, bias=False)
        self.lora_lidar_b_v = nn.Linear(r, dim, bias=False)

        self.lora_lidar_q = _LoRA_q(
            self.q,
            self.lora_lidar_a_q,
            self.lora_lidar_b_q,
            self.alpha
        )
        self.lora_lidar_kv = _LoRA_kv(
            self.kv,
            self.lora_lidar_a_v,
            self.lora_lidar_b_v,
            self.lora_lidar_a_v,  # Reusing the same for k and v
            self.lora_lidar_b_v,
            self.alpha
        )

        # LoRA for share
        self.lora_share_a_q = nn.Linear(dim, r, bias=False)
        self.lora_share_b_q = nn.Linear(r, dim, bias=False)
        self.lora_share_a_v = nn.Linear(dim, r, bias=False)
        self.lora_share_b_v = nn.Linear(r, dim, bias=False)

        self.lora_share_q = _LoRA_q(
            self.q,
            self.lora_share_a_q,
            self.lora_share_b_q,
            self.alpha
        )
        self.lora_share_kv = _LoRA_kv(
            self.kv,
            self.lora_share_a_v,
            self.lora_share_b_v,
            self.lora_share_a_v,  # Reusing the same for k and v
            self.lora_share_b_v,
            self.alpha
        )

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W, type) -> Tensor:
        B, N, C = x.shape

        # Apply LoRA to Q
        if type == 'rgb':
            q = self.lora_rgb_q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        elif type == 'depth':
            q = self.lora_depth_q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        elif type == 'event':
            q = self.lora_event_q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        elif type == 'lidar':
            q = self.lora_lidar_q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        elif type == 'share':
            q = self.lora_share_q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # Apply spatial reduction and normalization
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        # Apply LoRA to KV
        if type == 'rgb':
            k, v = self.lora_rgb_kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        elif type == 'depth':
            k, v = self.lora_depth_kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        elif type == 'event':
            k, v = self.lora_event_kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        elif type == 'lidar':
            k, v = self.lora_lidar_kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        elif type == 'share':
            k, v = self.lora_share_kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

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
        x = x.transpose(1, 2).reshape(B, C, H, W)
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

        # LoRA layers for depth
        self.lora_depth_a_fc1 = nn.Linear(c1, r, bias=False)
        self.lora_depth_b_fc1 = nn.Linear(r, c2, bias=False)
        self.lora_depth_a_fc2 = nn.Linear(c2, r, bias=False)
        self.lora_depth_b_fc2 = nn.Linear(r, c1, bias=False)

        # LoRA layers for event
        self.lora_event_a_fc1 = nn.Linear(c1, r, bias=False)
        self.lora_event_b_fc1 = nn.Linear(r, c2, bias=False)
        self.lora_event_a_fc2 = nn.Linear(c2, r, bias=False)
        self.lora_event_b_fc2 = nn.Linear(r, c1, bias=False)

        # LoRA layers for lidar
        self.lora_lidar_a_fc1 = nn.Linear(c1, r, bias=False)
        self.lora_lidar_b_fc1 = nn.Linear(r, c2, bias=False)
        self.lora_lidar_a_fc2 = nn.Linear(c2, r, bias=False)
        self.lora_lidar_b_fc2 = nn.Linear(r, c1, bias=False)

        # LoRA layers for share
        self.lora_share_a_fc1 = nn.Linear(c1, r, bias=False)
        self.lora_share_b_fc1 = nn.Linear(r, c2, bias=False)
        self.lora_share_a_fc2 = nn.Linear(c2, r, bias=False)
        self.lora_share_b_fc2 = nn.Linear(r, c1, bias=False)

    def forward(self, x: Tensor, H, W, type) -> Tensor:
        # Original fc1 output
        out_fc1 = self.fc1(x)
        # LoRA adjustment to fc1
        if type == 'rgb':
            out_fc1_lora = self.lora_rgb_b_fc1(self.lora_rgb_a_fc1(x))
        elif type == 'depth':
            out_fc1_lora = self.lora_depth_b_fc1(self.lora_depth_a_fc1(x))
        elif type == 'event':
            out_fc1_lora = self.lora_event_b_fc1(self.lora_event_a_fc1(x))
        elif type == 'lidar':
            out_fc1_lora = self.lora_lidar_b_fc1(self.lora_lidar_a_fc1(x))
        elif type == 'share':
            out_fc1_lora = self.lora_share_b_fc1(self.lora_share_a_fc1(x))
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
        elif type == 'depth':
            out_fc2_lora = self.lora_depth_b_fc2(self.lora_depth_a_fc2(out_act))
        elif type == 'event':
            out_fc2_lora = self.lora_event_b_fc2(self.lora_event_a_fc2(out_act))
        elif type == 'lidar':
            out_fc2_lora = self.lora_lidar_b_fc2(self.lora_lidar_a_fc2(out_act))
        elif type == 'share':
            out_fc2_lora = self.lora_share_b_fc2(self.lora_share_a_fc2(out_act))
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
        self.attn = Attention(dim, head, sr_ratio, 4, 0.5)
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


class Disentangle(nn.Module):
    def __init__(self, dim=64):
        super(Disentangle, self).__init__()

        # encoder
        r = 4
        self.fc1 = nn.Conv2d(dim, r, 1)
        self.bn1 = nn.BatchNorm2d(r, momentum=0.1)
        self.fc1a = nn.Conv2d(r, r, 1)
        self.fc1b = nn.Conv2d(r, r, 1)

        self.fc2 = nn.Conv2d(dim, r, 1)
        self.bn2 = nn.BatchNorm2d(r, momentum=0.1)
        self.fc2a = nn.Conv2d(r, r, 1)
        self.fc2b = nn.Conv2d(r, r, 1)

        self.fc3 = nn.Conv2d(dim, r, 1)
        self.bn3 = nn.BatchNorm2d(r, momentum=0.1)
        self.fc3a = nn.Conv2d(r, r, 1)
        self.fc3b = nn.Conv2d(r, r, 1)

        self.fuse_mean = nn.Conv2d(r, r, 1)
        self.fuse_var = nn.Conv2d(r, r, 1)

        self.proj = nn.Conv2d(r, dim, 1)

        # Add final normalization layer
        self.final_bn = nn.BatchNorm2d(dim, momentum=0.1)

    def encode(self, x):
        means = []
        vars = []

        h1 = F.relu(self.bn1(self.fc1(x[0])))
        a1_mean, a1_logvar = self.fc1a(h1), self.fc1b(h1)
        means.append(a1_mean)
        vars.append(a1_logvar)

        h1 = F.relu(self.bn2(self.fc2(x[1])))
        a1_mean, a1_logvar = self.fc2a(h1), self.fc2b(h1)
        means.append(a1_mean)
        vars.append(a1_logvar)

        h1 = F.relu(self.bn3(self.fc3(x[2])))
        a1_mean, a1_logvar = self.fc3a(h1), self.fc3b(h1)
        means.append(a1_mean)
        vars.append(a1_logvar)

        mu_w, sigma_fused_w = weighted_fusion_initialization(means, vars)

        mean = self.fuse_mean(mu_w)
        var = self.fuse_var(sigma_fused_w)

        noise_threshold = mean
        fused_mean, fused_variance = kalman_fusion_feature_map(means, vars, mu_w, sigma_fused_w, noise_threshold)

        del means, vars, mu_w, sigma_fused_w
        proportion = 0.5
        fused_variance = torch.log(fused_variance + 1e-6)

        fused = self.reparametrize(fused_mean, proportion * fused_variance + (1 - proportion) * var)
        output = self.proj(fused)

        # Apply the final normalization layer
        output = self.final_bn(output)

        return output

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.encode(x)
        return output


cmnext_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    # 'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    # 'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
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

        self.concat_conv1 = ConcatAndConv(3 * embed_dims[0], embed_dims[0])  ######
        self.concat_conv2 = ConcatAndConv(3 * embed_dims[1], embed_dims[1])
        self.concat_conv3 = ConcatAndConv(3 * embed_dims[2], embed_dims[2])
        self.concat_conv4 = ConcatAndConv(3 * embed_dims[3], embed_dims[3])

        # self.mam_concat_conv1 = ConcatAndConv(2 * embed_dims[0], embed_dims[0])  ######
        # self.mam_concat_conv2 = ConcatAndConv(2 * embed_dims[1], embed_dims[1])
        # self.mam_concat_conv3 = ConcatAndConv(2 * embed_dims[2], embed_dims[2])
        # self.mam_concat_conv4 = ConcatAndConv(2 * embed_dims[3], embed_dims[3])

        # self.final_conv1 = FinalConvProcessor(embed_dims[0], embed_dims[0])
        # self.final_conv2 = FinalConvProcessor(embed_dims[1], embed_dims[1])
        # self.final_conv3 = FinalConvProcessor(embed_dims[2], embed_dims[2])
        # self.final_conv4 = FinalConvProcessor(embed_dims[3], embed_dims[3])

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

        if self.num_modals > 1:
            self.extra_score_predictor = nn.ModuleList(
                [PredictorConv(embed_dims[i], self.num_modals) for i in range(len(depths))])

        self.prompt_disentangle1 = Disentangle(embed_dims[0])
        self.prompt_disentangle2 = Disentangle(embed_dims[1])
        self.prompt_disentangle3 = Disentangle(embed_dims[2])
        self.prompt_disentangle4 = Disentangle(embed_dims[3])


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
        #               self.FRMs, self.FFMs, self.extra_score_predictor,
        #               self.prompt_disentangle1, self.prompt_disentangle2, self.prompt_disentangle3, self.prompt_disentangle4,
        #               self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4]:
        #     for param in layer.parameters():
        #         param.requires_grad = False

    def tokenselect(self, x_ext, module, fuse):
        x_scores = module(x_ext)
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
        # x_f = functools.reduce(torch.max, x_ext)
        output = fuse(x_ext)
        return output #x_f

    def forward(self, x: list) -> list:  ######
        x_cam = x[0]
        if self.num_modals > 0:
            x_ext = x[1:]

        B = x_cam.shape[0]
        outs = []
        total_infonce_loss = []

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
            ## ------ diff feature encoder lora process ------ ##
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.patch_embed1(x_ext[i])
                if i == 0:
                    for blk in self.block1:
                        x_ext[i] = blk(x_ext[i], H, W, 'depth')
                elif i == 1:
                    for blk in self.block1:
                        x_ext[i] = blk(x_ext[i], H, W, 'event')
                elif i == 2:
                    for blk in self.block1:
                        x_ext[i] = blk(x_ext[i], H, W, 'lidar')
                x_ext[i] = self.norm1(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)

            # x1_f = self.concat_conv1(x_ext)
            x1_f = self.tokenselect(x_ext, self.extra_score_predictor[0],
                                   self.prompt_disentangle1) if self.num_modals > 1 else x_ext[0]

            ## ------ rgb & X_share fusion ------ ##
            x1_cam, x1_f = self.FRMs[0](x1_cam, x1_f)
            x_fused = self.FFMs[0](x1_cam, x1_f)
            outs.append(x_fused)

            ## ------ magic ------ ##
            # 计算各个模态特征与x_fused的相似性
            sim_r = cosine_similarity(x1_cam, x_fused)
            sim_d = cosine_similarity(x_ext[0], x_fused)
            sim_e = cosine_similarity(x_ext[1], x_fused)
            sim_l = cosine_similarity(x_ext[2], x_fused)

            # print(sim_r, sim_d, sim_e, sim_l)

            # 将相似性放入一个张量中，然后进行排序
            similarities = torch.stack([sim_r, sim_d, sim_e, sim_l], dim=1)  # [B, 4]
            ranked_similarities, indices = similarities.sort(dim=1, descending=True)

            # 获取最强(robust) 和 最弱 (fragile) 的特征
            f_rf = get_selected_features(B, indices[:, 0], x1_cam, x_ext[0], x_ext[1], x_ext[2])  # 最强特征
            f_fm = get_selected_features(B, indices[:, -1], x1_cam, x_ext[0], x_ext[1], x_ext[2])  # 最弱特征
            f_rm1 = get_selected_features(B, indices[:, 1], x1_cam, x_ext[0], x_ext[1], x_ext[2])
            f_rm2 = get_selected_features(B, indices[:, 2], x1_cam, x_ext[0], x_ext[1], x_ext[2])

            f_sa = (f_rf + f_fm) / 2.0
            # 剪裁
            f_sa = check_nan_inf(f_sa, "f_sa")
            f_rm1 = check_nan_inf(f_rm1, "f_rm1")
            f_rm2 = check_nan_inf(f_rm2, "f_rm2")
            # 语义一致性训练
            # 计算剩余特征与 f_sa 的相似性
            sim_rm1 = F.smooth_l1_loss(f_rm1, f_sa)
            sim_rm2 = F.smooth_l1_loss(f_rm2, f_sa)

            loss_c1 = (sim_rm1 + sim_rm2) / 2.0

            del x_fused, x1_f, f_sa, f_rf, f_fm, f_rm1, f_rm2
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
            ## ------ diff feature encoder lora process ------ ##
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.patch_embed2(x_ext[i])
                if i == 0:
                    for blk in self.block2:
                        x_ext[i] = blk(x_ext[i], H, W, 'depth')
                elif i == 1:
                    for blk in self.block2:
                        x_ext[i] = blk(x_ext[i], H, W, 'event')
                elif i == 2:
                    for blk in self.block2:
                        x_ext[i] = blk(x_ext[i], H, W, 'lidar')
                x_ext[i] = self.norm2(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)

            # x2_f = self.concat_conv2(x_ext)
            x2_f = self.tokenselect(x_ext, self.extra_score_predictor[1],
                                    self.prompt_disentangle2) if self.num_modals > 1 else x_ext[0]

            ## ------ rgb & X_share fusion ------ ##
            x2_cam, x2_f = self.FRMs[1](x2_cam, x2_f)
            x_fused = self.FFMs[1](x2_cam, x2_f)

            outs.append(x_fused)

            ## ------ magic ------ ##
            # 计算各个模态特征与x_fused的相似性
            sim_r = cosine_similarity(x2_cam, x_fused)
            sim_d = cosine_similarity(x_ext[0], x_fused)
            sim_e = cosine_similarity(x_ext[1], x_fused)
            sim_l = cosine_similarity(x_ext[2], x_fused)

            # 将相似性放入一个张量中，然后进行排序
            similarities = torch.stack([sim_r, sim_d, sim_e, sim_l], dim=1)  # [B, 4]
            ranked_similarities, indices = similarities.sort(dim=1, descending=True)

            # 获取最强(robust) 和 最弱 (fragile) 的特征
            f_rf = get_selected_features(B, indices[:, 0], x2_cam, x_ext[0], x_ext[1], x_ext[2])  # 最强特征
            f_fm = get_selected_features(B, indices[:, -1], x2_cam, x_ext[0], x_ext[1], x_ext[2])  # 最弱特征
            f_rm1 = get_selected_features(B, indices[:, 1], x2_cam, x_ext[0], x_ext[1], x_ext[2])
            f_rm2 = get_selected_features(B, indices[:, 2], x2_cam, x_ext[0], x_ext[1], x_ext[2])

            # mam
            f_sa = (f_rf + f_fm) / 2.0

            # 剪裁
            f_sa = check_nan_inf(f_sa, "f_sa")
            f_rm1 = check_nan_inf(f_rm1, "f_rm1")
            f_rm2 = check_nan_inf(f_rm2, "f_rm2")
            # 语义一致性训练
            # 计算剩余特征与 f_sa 的相似性
            sim_rm1 = F.smooth_l1_loss(f_rm1, f_sa)
            sim_rm2 = F.smooth_l1_loss(f_rm2, f_sa)

            loss_c2 = (sim_rm1 + sim_rm2) / 2.0

            del x_fused, x2_f, f_sa, f_rf, f_fm, f_rm1, f_rm2
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
            ## ------ diff feature encoder lora process ------ ##
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.patch_embed3(x_ext[i])
                if i == 0:
                    for blk in self.block3:
                        x_ext[i] = blk(x_ext[i], H, W, 'depth')
                elif i == 1:
                    for blk in self.block3:
                        x_ext[i] = blk(x_ext[i], H, W, 'event')
                elif i == 2:
                    for blk in self.block3:
                        x_ext[i] = blk(x_ext[i], H, W, 'lidar')
                x_ext[i] = self.norm3(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)

            # x3_f = self.concat_conv3(x_ext)
            x3_f = self.tokenselect(x_ext, self.extra_score_predictor[2],
                                    self.prompt_disentangle3) if self.num_modals > 1 else x_ext[0]

            ## ------ rgb & X_share fusion ------ ##
            x3_cam, x3_f = self.FRMs[2](x3_cam, x3_f)
            x_fused = self.FFMs[2](x3_cam, x3_f)
            outs.append(x_fused)

            ## ------ magic ------ ##
            # 计算各个模态特征与x_fused的相似性
            sim_r = cosine_similarity(x3_cam, x_fused)
            sim_d = cosine_similarity(x_ext[0], x_fused)
            sim_e = cosine_similarity(x_ext[1], x_fused)
            sim_l = cosine_similarity(x_ext[2], x_fused)

            # 将相似性放入一个张量中，然后进行排序
            similarities = torch.stack([sim_r, sim_d, sim_e, sim_l], dim=1)  # [B, 4]
            ranked_similarities, indices = similarities.sort(dim=1, descending=True)

            # 获取最强(robust) 和 最弱 (fragile) 的特征
            f_rf = get_selected_features(B, indices[:, 0], x3_cam, x_ext[0], x_ext[1], x_ext[2])  # 最强特征
            f_fm = get_selected_features(B, indices[:, -1], x3_cam, x_ext[0], x_ext[1], x_ext[2])  # 最弱特征
            f_rm1 = get_selected_features(B, indices[:, 1], x3_cam, x_ext[0], x_ext[1], x_ext[2])
            f_rm2 = get_selected_features(B, indices[:, 2], x3_cam, x_ext[0], x_ext[1], x_ext[2])

            f_sa = (f_rf + f_fm) / 2.0

            # 剪裁
            f_sa = check_nan_inf(f_sa, "f_sa")
            f_rm1 = check_nan_inf(f_rm1, "f_rm1")
            f_rm2 = check_nan_inf(f_rm2, "f_rm2")
            # 语义一致性训练
            # 计算剩余特征与 f_sa 的相似性
            sim_rm1 = F.smooth_l1_loss(f_rm1, f_sa)
            sim_rm2 = F.smooth_l1_loss(f_rm2, f_sa)

            loss_c3 = (sim_rm1 + sim_rm2) / 2.0

            del x_fused, x3_f, f_sa, f_rf, f_fm, f_rm1, f_rm2
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
            ## ------ diff feature encoder lora process ------ ##
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.patch_embed4(x_ext[i])
                if i == 0:
                    for blk in self.block4:
                        x_ext[i] = blk(x_ext[i], H, W, 'depth')
                elif i == 1:
                    for blk in self.block4:
                        x_ext[i] = blk(x_ext[i], H, W, 'event')
                elif i == 2:
                    for blk in self.block4:
                        x_ext[i] = blk(x_ext[i], H, W, 'lidar')
                x_ext[i] = self.norm4(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)

            # x4_f = self.concat_conv4(x_ext)
            x4_f = self.tokenselect(x_ext, self.extra_score_predictor[3],
                                    self.prompt_disentangle4) if self.num_modals > 1 else x_ext[0]

            ## ------ rgb & X_share fusion ------ ##
            x4_cam, x4_f = self.FRMs[3](x4_cam, x4_f)
            x_fused = self.FFMs[3](x4_cam, x4_f)
            outs.append(x_fused)

            ## ------ magic ------ ##
            # 计算各个模态特征与x_fused的相似性
            sim_r = cosine_similarity(x4_cam, x_fused)
            sim_d = cosine_similarity(x_ext[0], x_fused)
            sim_e = cosine_similarity(x_ext[1], x_fused)
            sim_l = cosine_similarity(x_ext[2], x_fused)

            # 将相似性放入一个张量中，然后进行排序
            similarities = torch.stack([sim_r, sim_d, sim_e, sim_l], dim=1)  # [B, 4]
            ranked_similarities, indices = similarities.sort(dim=1, descending=True)

            # 获取最强(robust) 和 最弱 (fragile) 的特征
            f_rf = get_selected_features(B, indices[:, 0], x4_cam, x_ext[0], x_ext[1], x_ext[2])  # 最强特征
            f_fm = get_selected_features(B, indices[:, -1], x4_cam, x_ext[0], x_ext[1], x_ext[2])  # 最弱特征
            f_rm1 = get_selected_features(B, indices[:, 1], x4_cam, x_ext[0], x_ext[1], x_ext[2])
            f_rm2 = get_selected_features(B, indices[:, 2], x4_cam, x_ext[0], x_ext[1], x_ext[2])

            f_sa = (f_rf + f_fm) / 2.0

            # 剪裁
            f_sa = check_nan_inf(f_sa, "f_sa")
            f_rm1 = check_nan_inf(f_rm1, "f_rm1")
            f_rm2 = check_nan_inf(f_rm2, "f_rm2")
            # 语义一致性训练
            # 计算剩余特征与 f_sa 的相似性
            sim_rm1 = F.smooth_l1_loss(f_rm1, f_sa)
            sim_rm2 = F.smooth_l1_loss(f_rm2, f_sa)

            loss_c4 = (sim_rm1 + sim_rm2) / 2.0

            del x_fused, x4_f, f_sa, f_rf, f_fm, f_rm1, f_rm2
            torch.cuda.empty_cache()
        else:
            outs.append(x4_cam)

        return outs, loss_c1, loss_c2, loss_c3, loss_c4  ######
        # return outs


def print_memory_usage(step=""):
    # GPU 显存使用情况
    if torch.cuda.is_available():
        print(f"[{step}] GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"[{step}] GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    # CPU 内存使用情况
    mem_info = process.memory_info()
    print(f"[{step}] CPU Memory Used: {mem_info.rss / 1024 ** 2:.2f} MB")


def check_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"{name} contains NaN or Inf!")
        tensor = torch.clamp(tensor, min=-1e6, max=1e6)  # 将其限制在一个范围内
    return tensor


def compute_infonce_loss(anchor, positive, negatives):
    """
    计算 InfoNCE 损失
    anchor: 锚点特征 (如 RGB 特征) [B, D]
    positive: 正样本特征 (如 Depth 特征) [B, D]
    negatives: 负样本特征 (batch 内其他样本) [B, N-1, D], N 是 batch size
    """
    # 检查输入是否有 NaN 或 Inf
    # assert torch.isfinite(anchor).all(), "Anchor contains NaN or Inf"
    # assert torch.isfinite(positive).all(), "Positive contains NaN or Inf"
    # assert torch.isfinite(negatives).all(), "Negatives contain NaN or Inf"

    # L2 归一化
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    negatives = F.normalize(negatives, p=2, dim=-1)

    # 计算正样本的相似度 (anchor 和正样本的点积)
    positive_sim = torch.sum(anchor * positive, dim=-1)  # [B]

    # 计算负样本的相似度 (anchor 和负样本的点积)
    negative_sim = torch.einsum('bd,bnd->bn', anchor, negatives)  # [B, N-1]

    # 拼接正负样本的相似度，正样本位于第 0 类
    logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)  # [B, 1 + (N-1)]

    # 对 logits 进行裁剪，防止极端数值
    logits = torch.clamp(logits, min=-50, max=50)

    # 使用温度缩放
    logits = logits / 0.07

    # 使用交叉熵损失，正样本对应的标签应为 0
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)  # [B]

    # 计算 InfoNCE 损失
    loss = F.cross_entropy(logits, labels)
    return loss


def get_selected_features(batch_size, indices, *features):
    selected_features = []
    for b in range(batch_size):
        selected_features.append(features[indices[b]][b])
    return torch.stack(selected_features)


# 展平后计算余弦相似性 (默认沿着channel维度进行)
def cosine_similarity(f1, f2):
    return F.cosine_similarity(f1.flatten(1), f2.flatten(1), dim=1)


def weighted_fusion_initialization(means, variances, epsilon=1e-6):
    # Stack means and variances along a new dimension for modalities, resulting in shape [m, b, c, h, w]
    means_stack = torch.stack(means, dim=0)
    variances_stack = torch.stack(variances, dim=0)

    # Clamp variances to prevent division by zero
    variances_stack = torch.clamp(variances_stack, min=epsilon)

    # Calculate inverse variance weights
    weights = 1 / variances_stack  # Shape [m, b, c, h, w]

    # Compute the weighted sum of means and the sum of weights
    weighted_sum_means = (means_stack * weights).sum(dim=0)  # Shape [b, c, h, w]
    sum_weights = weights.sum(dim=0)  # Shape [b, c, h, w]

    # Calculate the initial fused mean and variance
    mu_init = weighted_sum_means / sum_weights  # Shape [b, c, h, w]
    sigma_init_squared = 1 / sum_weights  # Initial variance, shape [b, c, h, w]

    return mu_init, sigma_init_squared


def kalman_fusion_feature_map(means, variances, mu_fused, sigma_fused_squared, noise_threshold):
    """
    Fuse the means and variances of multiple feature maps using Kalman filter principles.

    Args:
        means (torch.Tensor): Tensor of shape (n_features, batch_size, height, width) for means of each feature map.
        variances (torch.Tensor): Tensor of shape (n_features, batch_size, height, width) for variances of each feature map.
        noise_threshold (torch.Tensor): Feature map of shape (batch_size, height, width) for thresholding at each spatial location.

    Returns:
        torch.Tensor: Fused mean of shape (batch_size, height, width).
        torch.Tensor: Fused variance of shape (batch_size, height, width).
    """
    # Initialize fused mean and variance with the first feature map
    epsilon = 1e-6
    mean_means = torch.mean(torch.stack(means), dim=0)
    mean_vars = torch.mean(torch.stack(variances), dim=0)

    if torch.all(mean_means == 0) and torch.all(mean_vars == 0):
        return mean_means, mean_vars

    # Loop through remaining feature maps
    for i in range(len(means)):
        # Clamp variances to prevent division by zero
        sigma_fused_squared = torch.clamp(sigma_fused_squared, min=epsilon)
        variances_i_clamped = torch.clamp(variances[i], min=epsilon)

        K = sigma_fused_squared / (sigma_fused_squared + variances_i_clamped)

        outliers = torch.abs(means[i] - mu_fused) > noise_threshold * torch.sqrt(sigma_fused_squared)

        # Update fused mean and variance
        mu_fused = torch.where(outliers, mu_fused, mu_fused + K * (means[i] - mu_fused))
        sigma_fused_squared = torch.where(outliers, sigma_fused_squared + epsilon, (1 - K) * sigma_fused_squared)

        # Debugging: Check for NaN values
    return mu_fused, sigma_fused_squared



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
    device = torch.device('cuda')  # 使用 GPU
    x = [torch.ones(2, 3, 1024, 1024).to(device),
         torch.ones(2, 3, 1024, 1024).to(device),
         (torch.ones(2, 3, 1024, 1024) * 2).to(device),
         (torch.ones(2, 3, 1024, 1024) * 3).to(device)]

    model = CMNeXt(int(x[0].shape[2] / 4), 'B0', modals).to(device)

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