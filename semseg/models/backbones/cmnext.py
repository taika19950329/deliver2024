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
from semseg.models.modules.biapter import Bi_direct_adapter, Shared_direct_adapter
from semseg.utils.utils import nchw_to_nlc, nlc_to_nchw
from semseg.models.modules.BasicBlockNewNew import TF_3D
from semseg.models.modules.particleApt import BiParticFusion

import torch.nn as nn
import torch.autograd
from timm.models.layers import DropPath as timDrop
from mmcv.cnn import build_norm_layer
from torch.autograd import Variable


class MSPDWConv(nn.Module):
    def __init__(self, dim=768):
        super(MSPDWConv, self).__init__()
        self.mspdwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.mspdwconv(x)
        return x


class MSPMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.msp_fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.msp_dwconv = MSPDWConv(hidden_features)
        self.msp_act = act_layer()
        self.msp_fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.msp_drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.msp_fc1(x)

        x = self.msp_dwconv(x)
        x = self.msp_act(x)
        x = self.msp_drop(x)
        x = self.msp_fc2(x)
        x = self.msp_drop(x)

        return x


class MSPoolAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        pools = [3, 7, 11]
        self.msp_conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.msp_pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0] // 2, count_include_pad=False)
        self.msp_pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1] // 2, count_include_pad=False)
        self.msp_pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2] // 2, count_include_pad=False)
        self.msp_conv4 = nn.Conv2d(dim, dim, 1)
        self.msp_sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        x_in = self.msp_conv0(x)
        x_1 = self.msp_pool1(x_in)
        x_2 = self.msp_pool2(x_in)
        x_3 = self.msp_pool3(x_in)
        x_out = self.msp_sigmoid(self.msp_conv4(x_in + x_1 + x_2 + x_3)) * u
        return x_out + u


class MSPABlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.msp_norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.msp_attn = MSPoolAttention(dim)
        self.msp_drop_path = timDrop(drop_path) if drop_path > 0. else nn.Identity()
        self.msp_norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.msp_mlp = MSPMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.msp_layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.msp_layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.msp_is_channel_mix = True
        if self.msp_is_channel_mix:
            self.msp_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.msp_c_nets = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        x = x + self.msp_drop_path(
            self.msp_layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.msp_attn(self.msp_norm1(x)))  # 多尺度特征

        if self.msp_is_channel_mix:
            x_c = self.msp_avg_pool(x)  # H W 做全局的平均池化
            x_c = self.msp_c_nets(x_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 得到每个通道的权重
            x_c = x_c.expand_as(x)
            x_c_mix = x_c * x  # 重新标定，对通道重要性进行建模
            x_mlp = self.msp_drop_path(
                self.msp_layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.msp_mlp(self.msp_norm2(x)))
            x = x_c_mix + x_mlp
        else:
            x = x + self.msp_drop_path(
                self.msp_layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.msp_mlp(self.msp_norm2(x)))
        return x


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


class DualBlock(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., mlp_ratio=4., drop=0., act_layer=nn.GELU,
                 norm_cfg=dict(type='BN', requires_grad=True), is_fan=False):
        super().__init__()
        # MHSA
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not is_fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))

        # MSPA
        self.msp_norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.msp_attn = MSPoolAttention(dim)
        self.msp_drop_path = timDrop(dpr) if dpr > 0. else nn.Identity()
        self.msp_norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.msp_mlp = MSPMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.msp_layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.msp_layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.msp_is_channel_mix = True
        if self.msp_is_channel_mix:
            self.msp_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.msp_c_nets = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid())

        # fusion
        # self.fuse1 = BimodalFusion(dim)
        # self.fuse2 = BimodalFusion(dim)
        # self.fuse1 = FRM(dim=dim, reduction=1)
        # self.fuse1 = GRUKalmanFusion(dim, 8)
        self.fuse1 = BiParticFusion(dim, 8)

    def forward(self, x: Tensor, y: Tensor, B, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        y = y + self.msp_drop_path(
            self.msp_layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.msp_attn(self.msp_norm1(y)))  # 多尺度特征

        y = y.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        fuse = self.fuse1(x, y)
        x = x + fuse
        y = y + fuse
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        if self.msp_is_channel_mix:
            y_c = self.msp_avg_pool(y)  # H W 做全局的平均池化
            y_c = self.msp_c_nets(y_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 得到每个通道的权重
            y_c = y_c.expand_as(y)
            y_c_mix = y_c * y  # 重新标定，对通道重要性进行建模
            y_mlp = self.msp_drop_path(
                self.msp_layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.msp_mlp(self.msp_norm2(y)))
            y = y_c_mix + y_mlp
        else:
            y = y + self.msp_drop_path(
                self.msp_layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.msp_mlp(self.msp_norm2(y)))

        y = y.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        fuse = self.fuse1(x, y)
        x = x + fuse
        y = y + fuse
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x, y


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


class GRUKalmanFusion(nn.Module):
    def __init__(self, inplanes, hide_channel):
        super(GRUKalmanFusion, self).__init__()

        # GRU-like gates

        self.gateproj1 = GRUF2GateF1(inplanes, hide_channel)
        self.gateproj2 = GRUF2GateF1(inplanes, hide_channel)

        self.proj1 = nn.Linear(hide_channel, hide_channel)
        self.fcmean1 = nn.Linear(hide_channel, hide_channel)
        self.fcvar1 = nn.Linear(hide_channel, hide_channel)

        self.proj2 = nn.Linear(hide_channel, hide_channel)
        self.fcmean2 = nn.Linear(hide_channel, hide_channel)
        self.fcvar2 = nn.Linear(hide_channel, hide_channel)

        self.fuse_mean = nn.Linear(hide_channel, hide_channel)
        self.fuse_var = nn.Linear(hide_channel, hide_channel)

        self.proj_back = nn.Linear(hide_channel, inplanes)
        # Quality estimation modules (one for each feature)
        self.quality_estimator = nn.Linear(2 * hide_channel, 2)

    def quality(self, feature1, feature2):
        # Get quality scores for each feature
        quality_scores = self.quality_estimator(
            torch.cat([torch.mean(feature1, dim=1).unsqueeze(1), torch.mean(feature2, dim=1).unsqueeze(1)], dim=2))
        # Apply softmax along the channel dimension to get normalized weights
        weights = F.softmax(quality_scores, dim=2)

        # Split weights for each feature
        weight1 = weights[:, :, 0:1]
        weight2 = weights[:, :, 1:2]

        # Weighted fusion of the two features
        fused_feature = weight1 * feature1 + weight2 * feature2  # Shape: (B, C, H, W)

        return fused_feature

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, feature_1, feature_2):
        """
        Args:
            feature_1: torch.Tensor of shape (b, n, c) - First feature map (flattened spatial)
            feature_2: torch.Tensor of shape (b, n, c) - Second feature map (flattened spatial)

        Returns:
            Fused feature map with GRU gating and Kalman update
        """

        # Step 1: GRU-inspired gating
        feat1 = self.gateproj1(feature_1, feature_2)
        feat2 = self.gateproj2(feature_2, feature_1)

        # Step 2: Kalman update
        # Initialize mean and variance estimates from gated fusion output

        means = []
        vars = []

        h = F.relu(self.proj1(feat1))
        mean_estimate, variance_estimate = self.fcmean1(h), self.fcvar1(h)
        means.append(mean_estimate)
        vars.append(variance_estimate)

        h = F.relu(self.proj2(feat2))
        mean_estimate, variance_estimate = self.fcmean2(h), self.fcvar2(h)
        means.append(mean_estimate)
        vars.append(variance_estimate)

        epsilon = 1e-6

        # Kalman filter update

        mu_w, sigma_fused_w = weighted_fusion_initialization(means, vars)

        mean = self.fuse_mean(mu_w)
        var = self.fuse_var(sigma_fused_w)

        noise_threshold = mean

        fused_mean, fused_variance = kalman_fusion_feature_map(means, vars, mu_w, sigma_fused_w, noise_threshold)
        # import pdb;
        # pdb.set_trace()

        del means, vars, mu_w, sigma_fused_w
        fused_variance = torch.log(fused_variance + 1e-6)
        # proportion = 0.5
        fused_variance = self.quality(fused_variance, var)
        fused = self.reparametrize(fused_mean, fused_variance)

        # fused = self.reparametrize(fused_mean, proportion*fused_variance + (1-proportion) * var)

        output = self.proj_back(fused)

        return output


class GRUF2GateF1(nn.Module):
    def __init__(self, inplanes, hide_channel):
        super(GRUF2GateF1, self).__init__()

        # GRU-like gates

        self.proj1 = nn.Linear(inplanes, hide_channel)
        self.proj2 = nn.Linear(inplanes, hide_channel)

        self.reset_gate = nn.Linear(hide_channel * 2, hide_channel)
        self.update_gate = nn.Linear(hide_channel * 2, hide_channel)

        # Candidate fusion state
        self.candidate_fusion = nn.Linear(hide_channel * 2, hide_channel)

    def forward(self, feature_1, feature_2):
        """
        Args:
            feature_1: torch.Tensor of shape (b, n, c) - First feature map (flattened spatial)
            feature_2: torch.Tensor of shape (b, n, c) - Second feature map (flattened spatial)

        Returns:
            Fused feature map with GRU gating and Kalman update
        """

        # Step 1: GRU-inspired gating
        feat1 = self.proj1(feature_1)
        feat2 = self.proj2(feature_2)

        combined_features = torch.cat((feat1, feat2), dim=-1)  # Shape: (b, n, 2*c)

        reset_gate = torch.sigmoid(self.reset_gate(combined_features))  # Shape: (b, n, c)
        update_gate = torch.sigmoid(self.update_gate(combined_features))  # Shape: (b, n, c)

        reset_feature_1 = reset_gate * feat1  # Apply reset gate to feature_1
        candidate_fusion_input = torch.cat((reset_feature_1, feat2), dim=-1)  # Shape: (b, n, 2*c)
        candidate = torch.tanh(self.candidate_fusion(candidate_fusion_input))  # Shape: (b, n, c)

        gated_fusion = update_gate * candidate + (1 - update_gate) * feat1  # Gated output (b, n, c)

        return gated_fusion  # Final fused output


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

        # Quality estimation modules (one for each feature)
        self.quality_estimator = nn.Conv2d(2 * r, 2, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def quality(self, feature1, feature2):
        # Get quality scores for each feature
        quality_scores = self.quality_estimator(
            torch.cat([self.avgpool(feature1), self.avgpool(feature2)], dim=1))  # Shape: (B, 2, H, W)
        # Apply softmax along the channel dimension to get normalized weights
        weights = F.softmax(quality_scores, dim=1)  # Shape: (B, 2, H, W)

        # Split weights for each feature
        weight1 = weights[:, 0:1, :, :]  # Shape: (B, 1, H, W)
        weight2 = weights[:, 1:2, :, :]  # Shape: (B, 1, H, W)

        # Weighted fusion of the two features
        fused_feature = weight1 * feature1 + weight2 * feature2  # Shape: (B, C, H, W)

        return fused_feature

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
        # import pdb;
        # pdb.set_trace()

        del means, vars, mu_w, sigma_fused_w
        fused_variance = torch.log(fused_variance + 1e-6)
        # proportion = 0.5
        fused_variance = self.quality(fused_variance, var)
        fused = self.reparametrize(fused_mean, fused_variance)

        # fused = self.reparametrize(fused_mean, proportion*fused_variance + (1-proportion) * var)

        output = self.proj(fused)

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
    # 'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    # 'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    # 'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class CMNeXt(nn.Module):
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in cmnext_settings.keys(), f"Model name should be in {list(cmnext_settings.keys())}"
        embed_dims, depths = cmnext_settings[model_name]
        extra_depths = depths
        self.modals = modals[1:] if len(modals) > 1 else []
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        norm_cfg = dict(type='BN', requires_grad=True)

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7 // 2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3 // 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3 // 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3 // 2)

        # self.prompt_disentangle1 = Disentangle(embed_dims[0])
        # self.prompt_disentangle2 = Disentangle(embed_dims[1])
        # self.prompt_disentangle3 = Disentangle(embed_dims[2])
        # self.prompt_disentangle4 = Disentangle(embed_dims[3])

        if self.num_modals > 0:
            self.fusion1 = TF_3D(embedding_dim=64, patch_dim=8, nhead=4, method="TF")
            self.fusion2 = TF_3D(embedding_dim=128, patch_dim=8, nhead=4, method="TF")
            self.fusion3 = TF_3D(embedding_dim=320, patch_dim=8, nhead=4, method="TF")
            self.fusion4 = TF_3D(embedding_dim=512, patch_dim=8, nhead=4, method="TF")

        if self.num_modals > 0:
            self.extra_downsample_layers = nn.ModuleList([
                PatchEmbedParallel(3, embed_dims[0], 7, 4, 7 // 2, self.num_modals),
                *[PatchEmbedParallel(embed_dims[i], embed_dims[i + 1], 3, 2, 3 // 2, self.num_modals) for i in range(3)]
            ])
        if self.num_modals > 1:
            self.extra_score_predictor = nn.ModuleList(
                [PredictorConv(embed_dims[i], self.num_modals) for i in range(len(depths))])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList(
            [DualBlock(embed_dims[0], 1, 8, dpr[cur + i], mlp_ratio=8, norm_cfg=norm_cfg) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        if self.num_modals > 0:
            # self.extra_block1 = nn.ModuleList(
            #     [MSPABlock(embed_dims[0], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[0])])  # --- MSPABlock
            self.extra_norm1 = ConvLayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [DualBlock(embed_dims[1], 2, 4, dpr[cur + i], mlp_ratio=8, norm_cfg=norm_cfg) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        if self.num_modals > 0:
            # self.extra_block2 = nn.ModuleList(
            #     [MSPABlock(embed_dims[1], mlp_ratio=8, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[1])])
            self.extra_norm2 = ConvLayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [DualBlock(embed_dims[2], 5, 2, dpr[cur + i], mlp_ratio=4, norm_cfg=norm_cfg) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        if self.num_modals > 0:
            # self.extra_block3 = nn.ModuleList(
            #     [MSPABlock(embed_dims[2], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[2])])
            self.extra_norm3 = ConvLayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [DualBlock(embed_dims[3], 8, 1, dpr[cur + i], mlp_ratio=4, norm_cfg=norm_cfg) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        if self.num_modals > 0:
            # self.extra_block4 = nn.ModuleList(
            #     [MSPABlock(embed_dims[3], mlp_ratio=4, drop_path=dpr[cur + i], norm_cfg=norm_cfg) for i in
            #      range(extra_depths[3])])
            self.extra_norm4 = ConvLayerNorm(embed_dims[3])

        self.bi_fuse = nn.ModuleList(
            [Bi_direct_adapter(dim=embed_dims[i] // 2, input_dim=embed_dims[i]) for i in range(len(depths))])
        self.share_fuse = nn.ModuleList([Shared_direct_adapter(rank=embed_dims[i]) for i in range(len(depths))])

        if self.num_modals > 0:
            num_heads = [1, 2, 5, 8]
            # self.FRMs = nn.ModuleList([
            #     FRM(dim=embed_dims[0], reduction=1),
            #     FRM(dim=embed_dims[1], reduction=1),
            #     FRM(dim=embed_dims[2], reduction=1),
            #     FRM(dim=embed_dims[3], reduction=1)])
            self.FFMs = nn.ModuleList([
                FFM(dim=embed_dims[0], reduction=1, num_heads=num_heads[0], norm_layer=nn.BatchNorm2d),
                FFM(dim=embed_dims[1], reduction=1, num_heads=num_heads[1], norm_layer=nn.BatchNorm2d),
                FFM(dim=embed_dims[2], reduction=1, num_heads=num_heads[2], norm_layer=nn.BatchNorm2d),
                FFM(dim=embed_dims[3], reduction=1, num_heads=num_heads[3], norm_layer=nn.BatchNorm2d)])

        # 冻结参数debug
        # for layer in [self.block1, self.block2, self.block3, self.block4, self.norm1, self.norm2, self.norm3, self.norm4,
        #               self.FFMs, self.extra_norm1, self.extra_norm2, self.extra_norm3, self.extra_norm4,
        #               self.fusion1, self.fusion2, self.fusion3, self.fusion4,
        #               self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4]:
        #     for param in layer.parameters():
        #         param.requires_grad = False

    # def tokenselect(self, x_ext, module):
    #     x_scores = module(x_ext)
    #     for i in range(len(x_ext)):
    #         x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
    #     x_f = functools.reduce(torch.max, x_ext)
    #     return x_f
    def tokenselect(self, x_ext, module, fuse):
        x_scores = module(x_ext)
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
        # x_f = functools.reduce(torch.max, x_ext)
        output = fuse(x_ext)
        return output  # x_f

    def forward(self, x: list) -> list:
        x_cam = x[0]
        if self.num_modals > 0:
            x_ext = x[1:]
        B = x_cam.shape[0]
        outs = []
        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[0](x_ext)
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[0]) if self.num_modals > 1 else x_ext[0]
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[0],
            #                        self.prompt_disentangle1) if self.num_modals > 1 else x_ext[0]
            x_f = self.fusion1(x_ext)
            for blk in self.block1:
                x_cam, x_f = blk(x_cam, x_f, B, H, W)
            x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x1_f = self.extra_norm1(x_f)

            # x1_cam, x1_f = self.FRMs[0](x1_cam, x1_f)
            # INSTEAD OF FRM
            B, C, H, W = x1_cam.shape
            x1_cam_in = x1_cam.reshape(B, C, -1)
            x1_f_in = x1_f.reshape(B, C, -1)
            x1_cam_bi = self.bi_fuse[0](x1_cam_in)
            x1_f_bi = self.bi_fuse[0](x1_f_in)

            xy_s = self.share_fuse[0](x1_cam_in, x1_f_in)
            yx_s = self.share_fuse[0](x1_f_in, x1_cam_in)

            x1_cam = x1_cam + x1_f_bi.reshape(B, C, H, W) + xy_s.reshape(B, C, H, W)
            x1_f = x1_f + x1_cam_bi.reshape(B, C, H, W) + yx_s.reshape(B, C, H, W)

            x_fused = self.FFMs[0](x1_cam, x1_f)
            outs.append(x_fused)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x1_f for x_ in x_ext] if self.num_modals > 1 else [
                x1_f]
            del x1_cam_in, x1_f_in, x1_cam_bi, x1_f_bi
            torch.cuda.empty_cache()
        else:
            x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x1_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x1_cam)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[1](x_ext)
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[1]) if self.num_modals > 1 else x_ext[0]
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[1],
            #                        self.prompt_disentangle2) if self.num_modals > 1 else x_ext[0]
            x_f = self.fusion2(x_ext)
            for blk in self.block2:
                x_cam, x_f = blk(x_cam, x_f, B, H, W)
            x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x2_f = self.extra_norm2(x_f)

            # x2_cam, x2_f = self.FRMs[1](x2_cam, x2_f)
            # INSTEAD OF FRM
            B, C, H, W = x2_cam.shape
            x2_cam_in = x2_cam.reshape(B, C, -1)
            x2_f_in = x2_f.reshape(B, C, -1)
            x2_cam_bi = self.bi_fuse[1](x2_cam_in)
            x2_f_bi = self.bi_fuse[1](x2_f_in)

            xy_s = self.share_fuse[1](x2_cam_in, x2_f_in)
            yx_s = self.share_fuse[1](x2_f_in, x2_cam_in)

            x2_cam = x2_cam + x2_f_bi.reshape(B, C, H, W) + xy_s.reshape(B, C, H, W)
            x2_f = x2_f + x2_cam_bi.reshape(B, C, H, W) + yx_s.reshape(B, C, H, W)

            x_fused = self.FFMs[1](x2_cam, x2_f)
            outs.append(x_fused)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x2_f for x_ in x_ext] if self.num_modals > 1 else [
                x2_f]
            del x2_cam_in, x2_f_in, x2_cam_bi, x2_f_bi
            torch.cuda.empty_cache()
        else:
            x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[2](x_ext)
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[2]) if self.num_modals > 1 else x_ext[0]
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[2],
            #                        self.prompt_disentangle3) if self.num_modals > 1 else x_ext[0]
            x_f = self.fusion3(x_ext)
            for blk in self.block3:
                x_cam, x_f = blk(x_cam, x_f, B, H, W)
            x3_f = self.extra_norm3(x_f)
            x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)

            # x3_cam, x3_f = self.FRMs[2](x3_cam, x3_f)
            # INSTEAD OF FRM
            B, C, H, W = x3_cam.shape
            x3_cam_in = x3_cam.reshape(B, C, -1)
            x3_f_in = x3_f.reshape(B, C, -1)
            x3_cam_bi = self.bi_fuse[2](x3_cam_in)
            x3_f_bi = self.bi_fuse[2](x3_f_in)

            xy_s = self.share_fuse[2](x3_cam_in, x3_f_in)
            yx_s = self.share_fuse[2](x3_f_in, x3_cam_in)

            x3_cam = x3_cam + x3_f_bi.reshape(B, C, H, W) + xy_s.reshape(B, C, H, W)
            x3_f = x3_f + x3_cam_bi.reshape(B, C, H, W) + yx_s.reshape(B, C, H, W)

            x_fused = self.FFMs[2](x3_cam, x3_f)
            outs.append(x_fused)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x3_f for x_ in x_ext] if self.num_modals > 1 else [
                x3_f]
            del x3_cam_in, x3_f_in, x3_cam_bi, x3_f_bi
            torch.cuda.empty_cache()
        else:
            x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[3](x_ext)
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[3]) if self.num_modals > 1 else x_ext[0]
            # x_f = self.tokenselect(x_ext, self.extra_score_predictor[3],
            #                        self.prompt_disentangle4) if self.num_modals > 1 else x_ext[0]
            x_f = self.fusion4(x_ext)
            for blk in self.block4:
                x_cam, x_f = blk(x_cam, x_f, B, H, W)
            x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x4_f = self.extra_norm4(x_f)

            # x4_cam, x4_f = self.FRMs[3](x4_cam, x4_f)
            # INSTEAD OF FRM
            B, C, H, W = x4_cam.shape
            x4_cam_in = x4_cam.reshape(B, C, -1)
            x4_f_in = x4_f.reshape(B, C, -1)
            x4_cam_bi = self.bi_fuse[3](x4_cam_in)
            x4_f_bi = self.bi_fuse[3](x4_f_in)

            xy_s = self.share_fuse[3](x4_cam_in, x4_f_in)
            yx_s = self.share_fuse[3](x4_f_in, x4_cam_in)

            x4_cam = x4_cam + x4_f_bi.reshape(B, C, H, W) + xy_s.reshape(B, C, H, W)
            x4_f = x4_f + x4_cam_bi.reshape(B, C, H, W) + yx_s.reshape(B, C, H, W)
            # x4_cam, x4_f = self.FRMs[3](x4_cam, x4_f)

            x_fused = self.FFMs[3](x4_cam, x4_f)
            outs.append(x_fused)
            del x4_cam_in, x4_f_in, x4_cam_bi, x4_f_bi
            torch.cuda.empty_cache()
        else:
            x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x4_cam)

        return outs


if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    device = torch.device('cuda')
    x = [torch.zeros(1, 3, 1024, 1024).to(device), torch.ones(1, 3, 1024, 1024).to(device),
         (torch.ones(1, 3, 1024, 1024) * 2).to(device),
         (torch.ones(1, 3, 1024, 1024) * 3).to(device)]
    model = CMNeXt('B2', modals).to(device)
    outs = model(x)
    for y in outs:
        print(y.shape)