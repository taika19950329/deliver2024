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

from semseg.models.modules.moe_lora import MoE_lora, AllInOne_lora, MoE_lora_new, MoE_lora_rgb, AttentionWeightedSum, ConcatAndConv, \
    FinalConvProcessor
from segment_anything import sam_model_registry
from semseg.models.modules.sam_lora import *
from semseg.models.modules.BasicBlock import TF_3D

import numpy as np


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

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


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


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
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not is_fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
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


cmnext_settings = {
    # 'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    # 'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
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

        # if self.num_modals > 0: ######
        #     self.moe1 = MoE_lora(3, 64, 7, 4, 7//2, 3, 1024, True, 2)
        #     self.moe2 = MoE_lora(64, 128, 3, 2, 3//2, 3, 256, True, 2)
        #     self.moe3 = MoE_lora(128, 320, 3, 2, 3//2, 3, 128, True, 2)
        #     self.moe4 = MoE_lora(320, 512, 3, 2, 3//2, 3, 64, True, 2)
        #
        # if self.num_modals > 0:
        #     self.fusion1 = TF_3D(embedding_dim=64, volumn_size=256, nhead=4, method="TF")
        #     self.fusion2 = TF_3D(embedding_dim=128, volumn_size=128, nhead=4, method="TF")
        #     self.fusion3 = TF_3D(embedding_dim=320, volumn_size=64, nhead=4, method="TF")
        #     self.fusion4 = TF_3D(embedding_dim=512, volumn_size=32, nhead=4, method="TF")

        # if self.num_modals > 0:
        #     self.allinone_moe1 = AllInOne_lora(3, 32, 7, 4, 4, 7//2, 6, 1024, True, 2)
        #     self.allinone_moe2 = AllInOne_lora(64, 64, 3, 2, 2, 3//2, 6, 256, True, 2)
        #     self.allinone_moe3 = AllInOne_lora(128, 160, 3, 2, 2, 3//2, 6, 128, True, 2)
        #     self.allinone_moe4 = AllInOne_lora(320, 256, 3, 2, 2, 3 // 2, 6, 64, True, 2)

        if self.num_modals > 0:  ######
            self.moe1 = MoE_lora_new(3, embed_dims[0], 7, 4, 7 // 2, 3, 1024, True, 2)
            self.moe2 = MoE_lora_new(64, embed_dims[1], 3, 2, 3 // 2, 3, 256, True, 2)
            self.moe3 = MoE_lora_new(128, embed_dims[2], 3, 2, 3 // 2, 3, 128, True, 2)
            self.moe4 = MoE_lora_new(320, embed_dims[3], 3, 2, 3 // 2, 3, 64, True, 2)
        # if self.num_modals > 0:  ######
        #     self.moe1 = MoE_lora_rgb(3, embed_dims[0], 7, 4, 7 // 2, 3, 1024, True, 2)
        #     self.moe2 = MoE_lora_rgb(64, embed_dims[1], 3, 2, 3 // 2, 3, 256, True, 2)
        #     self.moe3 = MoE_lora_rgb(128, embed_dims[2], 3, 2, 3 // 2, 3, 128, True, 2)
        #     self.moe4 = MoE_lora_rgb(320, embed_dims[3], 3, 2, 3 // 2, 3, 64, True, 2)
        # if self.num_modals > 0:
        #     self.sam1 = sam_model_registry["vit_b"](
        #         checkpoint="/home/yi/Documents/DELIVER/checkpoints/pretrained/sam/sam_vit_b_01ec64.pth")
        #     self.sam2 = sam_model_registry["vit_b"](
        #         checkpoint="/home/yi/Documents/DELIVER/checkpoints/pretrained/sam/sam_vit_b_01ec64.pth")
        #     self.sam3 = sam_model_registry["vit_b"](
        #         checkpoint="/home/yi/Documents/DELIVER/checkpoints/pretrained/sam/sam_vit_b_01ec64.pth")
        #     self.sam4 = sam_model_registry["vit_b"](
        #         checkpoint="/home/yi/Documents/DELIVER/checkpoints/pretrained/sam/sam_vit_b_01ec64.pth")
        #     self.lora_sam1 = LoRA_Sam(self.sam1, 4)
        #     self.lora_sam2 = LoRA_Sam(self.sam2, 4)
        #     self.lora_sam3 = LoRA_Sam(self.sam3, 4)
        #     self.lora_sam4 = LoRA_Sam(self.sam4, 4)

        # if self.num_modals > 0:
        #     self.change_out_ch1 = nn.Conv2d(256, 64, 1)
        #     self.change_out_ch2 = nn.Conv2d(256, 128, 1)
        #     self.change_out_ch3 = nn.Conv2d(256, 320, 1)
        #     self.change_out_ch4 = nn.Conv2d(256, 512, 1)
        #
        # if self.num_modals > 0:
        #     self.change_in_ch1 = nn.Conv2d(64, 3, 1)
        #     self.change_in_ch2 = nn.Conv2d(128, 3, 1)
        #     self.change_in_ch3 = nn.Conv2d(320, 3, 1)

        self.attn_gate1 = AttentionWeightedSum()
        self.attn_gate2 = AttentionWeightedSum()
        self.attn_gate3 = AttentionWeightedSum()
        self.attn_gate4 = AttentionWeightedSum()

        self.concat_conv1 = ConcatAndConv(3 * embed_dims[0], embed_dims[0])  ######
        self.concat_conv2 = ConcatAndConv(3 * embed_dims[1], embed_dims[1])
        self.concat_conv3 = ConcatAndConv(3 * embed_dims[2], embed_dims[2])
        self.concat_conv4 = ConcatAndConv(3 * embed_dims[3], embed_dims[3])

        self.final_conv1 = FinalConvProcessor(embed_dims[0], embed_dims[0])
        self.final_conv2 = FinalConvProcessor(embed_dims[1], embed_dims[1])
        self.final_conv3 = FinalConvProcessor(embed_dims[2], embed_dims[2])
        self.final_conv4 = FinalConvProcessor(embed_dims[3], embed_dims[3])

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7 // 2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3 // 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3 // 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3 // 2)

        # if self.num_modals > 0:
        #     self.extra_downsample_layers = nn.ModuleList([
        #         PatchEmbedParallel(3, embed_dims[0], 7, 4, 7//2, self.num_modals),
        #         *[PatchEmbedParallel(embed_dims[i], embed_dims[i+1], 3, 2, 3//2, self.num_modals) for i in range(3)]
        #     ])
        if self.num_modals > 1:
            self.extra_score_predictor = nn.ModuleList(
                [PredictorConv(embed_dims[i], self.num_modals) for i in range(len(depths))])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        if self.num_modals > 0:
            # self.shared_extra_block1 = nn.ModuleList(
            #     [MSPABlock(embed_dims[0], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[0])])  # --- MSPABlock
            # self.shared_extra_norm1 = ConvLayerNorm(embed_dims[0])
            self.shared_extra_block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
            self.shared_extra_norm1 = nn.LayerNorm(embed_dims[0])

        if self.num_modals > 0:  ######
            # self.diff1_extra_block1 = nn.ModuleList(
            #     [MSPABlock(embed_dims[0], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[0])])  # --- MSPABlock
            # self.diff1_extra_norm1 = ConvLayerNorm(embed_dims[0])
            # 
            # self.diff2_extra_block1 = nn.ModuleList(
            #     [MSPABlock(embed_dims[0], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[0])])  # --- MSPABlock
            # self.diff2_extra_norm1 = ConvLayerNorm(embed_dims[0])
            # 
            # self.diff3_extra_block1 = nn.ModuleList(
            #     [MSPABlock(embed_dims[0], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[0])])  # --- MSPABlock
            # self.diff3_extra_norm1 = ConvLayerNorm(embed_dims[0])

            self.diff1_extra_block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
            self.diff1_extra_norm1 = nn.LayerNorm(embed_dims[0])

            self.diff2_extra_block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
            self.diff2_extra_norm1 = nn.LayerNorm(embed_dims[0])

            self.diff3_extra_block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
            self.diff3_extra_norm1 = nn.LayerNorm(embed_dims[0])

            # self.extra_block1_diff4 = nn.ModuleList(
            #     [MSPABlock(embed_dims[0], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[0])])  # --- MSPABlock
            # self.extra_norm1_diff4 = ConvLayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        if self.num_modals > 0:
            # self.shared_extra_block2 = nn.ModuleList(
            #     [MSPABlock(embed_dims[1], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[1])])
            # self.shared_extra_norm2 = ConvLayerNorm(embed_dims[1])
            self.shared_extra_block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
            self.shared_extra_norm2 = nn.LayerNorm(embed_dims[1])

        if self.num_modals > 0:  ######
            # self.diff1_extra_block2 = nn.ModuleList(
            #     [MSPABlock(embed_dims[1], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[1])])
            # self.diff1_extra_norm2 = ConvLayerNorm(embed_dims[1])
            #
            # self.diff2_extra_block2 = nn.ModuleList(
            #     [MSPABlock(embed_dims[1], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[1])])
            # self.diff2_extra_norm2 = ConvLayerNorm(embed_dims[1])
            #
            # self.diff3_extra_block2 = nn.ModuleList(
            #     [MSPABlock(embed_dims[1], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[1])])
            # self.diff3_extra_norm2 = ConvLayerNorm(embed_dims[1])

            self.diff1_extra_block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
            self.diff1_extra_norm2 = nn.LayerNorm(embed_dims[1])

            self.diff2_extra_block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
            self.diff2_extra_norm2 = nn.LayerNorm(embed_dims[1])

            self.diff3_extra_block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
            self.diff3_extra_norm2 = nn.LayerNorm(embed_dims[1])

            # self.extra_block2_diff4 = nn.ModuleList(
            #     [MSPABlock(embed_dims[1], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[1])])
            # self.extra_norm2_diff4 = ConvLayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        if self.num_modals > 0:
            # self.shared_extra_block3 = nn.ModuleList(
            #     [MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[2])])
            # self.shared_extra_norm3 = ConvLayerNorm(embed_dims[2])
            self.shared_extra_block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
            self.shared_extra_norm3 = nn.LayerNorm(embed_dims[2])

        if self.num_modals > 0:  ######
            # self.diff1_extra_block3 = nn.ModuleList(
            #     [MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[2])])
            # self.diff1_extra_norm3 = ConvLayerNorm(embed_dims[2])
            #
            # self.diff2_extra_block3 = nn.ModuleList(
            #     [MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[2])])
            # self.diff2_extra_norm3 = ConvLayerNorm(embed_dims[2])
            #
            # self.diff3_extra_block3 = nn.ModuleList(
            #     [MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[2])])
            # self.diff3_extra_norm3 = ConvLayerNorm(embed_dims[2])

            self.diff1_extra_block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
            self.diff1_extra_norm3 = nn.LayerNorm(embed_dims[2])

            self.diff2_extra_block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
            self.diff2_extra_norm3 = nn.LayerNorm(embed_dims[2])

            self.diff3_extra_block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
            self.diff3_extra_norm3 = nn.LayerNorm(embed_dims[2])

            # self.extra_block3_diff4 = nn.ModuleList(
            #     [MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[2])])
            # self.extra_norm3_diff4 = ConvLayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        if self.num_modals > 0:
            # self.shared_extra_block4 = nn.ModuleList(
            #     [MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[3])])
            # self.shared_extra_norm4 = ConvLayerNorm(embed_dims[3])
            self.shared_extra_block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
            self.shared_extra_norm4 = nn.LayerNorm(embed_dims[3])

        if self.num_modals > 0:  ######
            # self.diff1_extra_block4 = nn.ModuleList(
            #     [MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[3])])
            # self.diff1_extra_norm4 = ConvLayerNorm(embed_dims[3])
            #
            # self.diff2_extra_block4 = nn.ModuleList(
            #     [MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[3])])
            # self.diff2_extra_norm4 = ConvLayerNorm(embed_dims[3])
            #
            # self.diff3_extra_block4 = nn.ModuleList(
            #     [MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[3])])
            # self.diff3_extra_norm4 = ConvLayerNorm(embed_dims[3])

            self.diff1_extra_block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
            self.diff1_extra_norm4 = nn.LayerNorm(embed_dims[3])

            self.diff2_extra_block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
            self.diff2_extra_norm4 = nn.LayerNorm(embed_dims[3])

            self.diff3_extra_block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
            self.diff3_extra_norm4 = nn.LayerNorm(embed_dims[3])

            # self.extra_block4_diff4 = nn.ModuleList(
            #     [MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[3])])
            # self.extra_norm4_diff4 = ConvLayerNorm(embed_dims[3])

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

        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam)
        for blk in self.block1:
            x_cam = blk(x_cam, H, W)
        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # x_cam = self.lora_sam1.sam.image_encoder(x_cam)
        # x1_cam = F.interpolate(x_cam, size=(256, 256), mode="bilinear", align_corners=False)
        # x1_cam = self.change_out_ch1(x1_cam).permute(0, 2, 3, 1)
        # x1_cam = self.norm1(x1_cam).permute(0, 3, 1, 2)

        if self.num_modals > 0:
            # x_ext, loss_moe1 = self.moe1(x_ext)
            # x_ext, _, _ = self.extra_downsample_layers[0](x_ext)
            # x_f = self.fusion1(x_ext)
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[0]) if self.num_modals > 1 else x_ext[0]
            # x_ext, x_f, loss_moe1 = self.allinone_moe1(x_ext)
            # x_ext, x_f, loss_moe1 = self.moe1(x_ext)
            x_ext, x_f, loss_moe1 = self.moe1(x_ext)
            B, C, H, W = x_f.shape
            x_f = x_f.view(B, C, -1).permute(0, 2, 1)
            for blk in self.shared_extra_block1:
                x_f = blk(x_f, H, W)

            x_ext[0] = x_ext[0].view(B, C, -1).permute(0, 2, 1)
            x_ext[1] = x_ext[1].view(B, C, -1).permute(0, 2, 1)
            x_ext[2] = x_ext[2].view(B, C, -1).permute(0, 2, 1)
            for blk1 in self.diff1_extra_block1:
                x_ext[0] = blk1(x_ext[0], H, W)
            for blk2 in self.diff2_extra_block1:
                x_ext[1] = blk2(x_ext[1], H, W)
            for blk3 in self.diff3_extra_block1:
                x_ext[2] = blk3(x_ext[2], H, W)
            # for blk4 in self.extra_block1_diff4:
            #     x_ext[3] = blk4(x_ext[3])

            x_ext[0] = self.diff1_extra_norm1(x_ext[0]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext[1] = self.diff2_extra_norm1(x_ext[1]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext[2] = self.diff3_extra_norm1(x_ext[2]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            # x_ext[3] = self.extra_norm1_diff4(x_ext[3])
            x1_f = self.shared_extra_norm1(x_f).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x1_cam, x1_f = self.FRMs[0](x1_cam, x1_f)
            x_fused = self.FFMs[0](x1_cam, x1_f)

            x_ext_attn = self.attn_gate1(x_ext, x_fused)
            expert_combine_output = self.concat_conv1(x_ext_attn)
            final_fused = self.final_conv1(expert_combine_output, x_fused)
            outs.append(final_fused)
            # outs.append(x_fused)
            # x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x1_f for x_ in x_ext] if self.num_modals > 1 else [x1_f]
        else:
            outs.append(x1_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x1_cam)
        for blk in self.block2:
            x_cam = blk(x_cam, H, W)
        x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # x1_cam = self.change_in_ch1(x1_cam)
        # x1_cam = F.interpolate(x1_cam, size=(1024, 1024), mode="bilinear", align_corners=False)
        # x_cam = self.lora_sam2.sam.image_encoder(x1_cam)
        # x2_cam = F.interpolate(x_cam, size=(128, 128), mode="bilinear", align_corners=False)
        # x2_cam = self.change_out_ch2(x2_cam).permute(0, 2, 3, 1)
        # x2_cam = self.norm2(x2_cam).permute(0, 3, 1, 2)
        # print(x2_cam.shape)
        # raise Exception

        if self.num_modals > 0:
            # x_ext, loss_moe2 = self.moe2(x_ext)
            # x_ext, _, _ = self.extra_downsample_layers[1](x_ext)
            # print(x_ext[0].shape)
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[1]) if self.num_modals > 1 else x_ext[0]
            # x_ext, _, _ = self.extra_downsample_layers[0](x_ext)
            # x_f = self.fusion2(x_ext)
            # x_ext, x_f, loss_moe2 = self.allinone_moe2(x_ext)
            x_ext, x_f, loss_moe2 = self.moe2(x_ext)
            B, C, H, W = x_f.shape
            x_f = x_f.view(B, C, -1).permute(0, 2, 1)
            for blk in self.shared_extra_block2:
                x_f = blk(x_f, H, W)

            x_ext[0] = x_ext[0].view(B, C, -1).permute(0, 2, 1)
            x_ext[1] = x_ext[1].view(B, C, -1).permute(0, 2, 1)
            x_ext[2] = x_ext[2].view(B, C, -1).permute(0, 2, 1)

            for blk1 in self.diff1_extra_block2:
                x_ext[0] = blk1(x_ext[0], H, W)
            for blk2 in self.diff2_extra_block2:
                x_ext[1] = blk2(x_ext[1], H, W)
            for blk3 in self.diff3_extra_block2:
                x_ext[2] = blk3(x_ext[2], H, W)
            # for blk4 in self.extra_block2_diff4:
            #     x_ext[3] = blk4(x_ext[3])

            x_ext[0] = self.diff1_extra_norm2(x_ext[0]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext[1] = self.diff2_extra_norm2(x_ext[1]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext[2] = self.diff3_extra_norm2(x_ext[2]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            # x_ext[3] = self.extra_norm2_diff4(x_ext[3])
            x2_f = self.shared_extra_norm2(x_f).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x2_cam, x2_f = self.FRMs[1](x2_cam, x2_f)
            x_fused = self.FFMs[1](x2_cam, x2_f)

            x_ext_attn = self.attn_gate2(x_ext, x_fused)
            expert_combine_output = self.concat_conv2(x_ext_attn)
            final_fused = self.final_conv2(expert_combine_output, x_fused)
            outs.append(final_fused)
            # outs.append(x_fused)
            # x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x2_f for x_ in x_ext] if self.num_modals > 1 else [x2_f]
        else:
            outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)
        for blk in self.block3:
            x_cam = blk(x_cam, H, W)
        x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            # x_ext, loss_moe3 = self.moe3(x_ext)
            # x_ext, _, _ = self.extra_downsample_layers[2](x_ext)
            # print(x_ext[0].shape)
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[2]) if self.num_modals > 1 else x_ext[0]
            # x_ext, _, _ = self.extra_downsample_layers[0](x_ext)
            # x_f = self.fusion3(x_ext)
            # x_ext, x_f, loss_moe3 = self.allinone_moe3(x_ext)
            x_ext, x_f, loss_moe3 = self.moe3(x_ext)
            B, C, H, W = x_f.shape
            x_f = x_f.view(B, C, -1).permute(0, 2, 1)
            for blk in self.shared_extra_block3:
                x_f = blk(x_f, H, W)

            x_ext[0] = x_ext[0].view(B, C, -1).permute(0, 2, 1)
            x_ext[1] = x_ext[1].view(B, C, -1).permute(0, 2, 1)
            x_ext[2] = x_ext[2].view(B, C, -1).permute(0, 2, 1)

            for blk1 in self.diff1_extra_block3:
                x_ext[0] = blk1(x_ext[0], H, W)
            for blk2 in self.diff2_extra_block3:
                x_ext[1] = blk2(x_ext[1], H, W)
            for blk3 in self.diff3_extra_block3:
                x_ext[2] = blk3(x_ext[2], H, W)
            # for blk4 in self.extra_block3_diff4:
            #     x_ext[3] = blk4(x_ext[3])

            x_ext[0] = self.diff1_extra_norm3(x_ext[0]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext[1] = self.diff2_extra_norm3(x_ext[1]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext[2] = self.diff3_extra_norm3(x_ext[2]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            # x_ext[3] = self.extra_norm3_diff4(x_ext[3])
            x3_f = self.shared_extra_norm3(x_f).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x3_cam, x3_f = self.FRMs[2](x3_cam, x3_f)
            x_fused = self.FFMs[2](x3_cam, x3_f)

            x_ext_attn = self.attn_gate3(x_ext, x_fused)
            expert_combine_output = self.concat_conv3(x_ext_attn)
            final_fused = self.final_conv3(expert_combine_output, x_fused)
            outs.append(final_fused)
            # outs.append(x_fused)
            # x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x3_f for x_ in x_ext] if self.num_modals > 1 else [x3_f]
        else:
            outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x_cam = blk(x_cam, H, W)
        x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            # x_ext, loss_moe4 = self.moe4(x_ext)
            # print(x_ext[0].shape)
            # x_ext, _, _ = self.extra_downsample_layers[3](x_ext)
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[3]) if self.num_modals > 1 else x_ext[0]
            # x_ext, _, _ = self.extra_downsample_layers[0](x_ext)
            # x_f = self.fusion4(x_ext)
            # x_ext, x_f, loss_moe4 = self.allinone_moe4(x_ext)
            x_ext, x_f, loss_moe4 = self.moe4(x_ext)
            B, C, H, W = x_f.shape
            x_f = x_f.view(B, C, -1).permute(0, 2, 1)
            for blk in self.shared_extra_block4:
                x_f = blk(x_f, H, W)

            x_ext[0] = x_ext[0].view(B, C, -1).permute(0, 2, 1)
            x_ext[1] = x_ext[1].view(B, C, -1).permute(0, 2, 1)
            x_ext[2] = x_ext[2].view(B, C, -1).permute(0, 2, 1)

            for blk1 in self.diff1_extra_block4:
                x_ext[0] = blk1(x_ext[0], H, W)
            for blk2 in self.diff2_extra_block4:
                x_ext[1] = blk2(x_ext[1], H, W)
            for blk3 in self.diff3_extra_block4:
                x_ext[2] = blk3(x_ext[2], H, W)
            # for blk4 in self.extra_block4_diff4:
            #     x_ext[3] = blk4(x_ext[3])

            x_ext[0] = self.diff1_extra_norm4(x_ext[0]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext[1] = self.diff2_extra_norm4(x_ext[1]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_ext[2] = self.diff3_extra_norm4(x_ext[2]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            # x_ext[3] = self.extra_norm4_diff4(x_ext[3])
            x4_f = self.shared_extra_norm4(x_f).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            # print("4", x4_cam.shape, x4_f.shape)
            x4_cam, x4_f = self.FRMs[3](x4_cam, x4_f)
            x_fused = self.FFMs[3](x4_cam, x4_f)

            x_ext_attn = self.attn_gate4(x_ext, x_fused)
            expert_combine_output = self.concat_conv4(x_ext_attn)
            final_fused = self.final_conv4(expert_combine_output, x_fused)
            outs.append(final_fused)
            # outs.append(x_fused)
        else:
            outs.append(x4_cam)

        return outs, loss_moe1, loss_moe2, loss_moe3, loss_moe4  ######
        # return outs


if __name__ == '__main__':
    modals = ['rgb', 'depth', 'event', 'lidar']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = [torch.ones(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024), (torch.ones(2, 3, 1024, 1024) * 2),
         (torch.ones(2, 3, 1024, 1024) * 3)]
    # print(int(x[0].shape[2]/4))
    # raise Exception
    model = CMNeXt(int(x[0].shape[2] / 4), 'B2', modals)

    # total = 0
    # for name, param in model.named_parameters():
    #     if "fusion3" in name:
    #         if param.requires_grad:
    #             layer_params_m = param.numel() / 1e6
    #             total += layer_params_m
    #             print(f"层 {name} 参数量: {param.numel()} ({layer_params_m:.6f} M)")
    # print(f" ({total:.6f} M)")
    #
    # raise Exception

    outs = model(x)
    for y in outs[0]:
        print(y.shape)
    print(outs[1:])

