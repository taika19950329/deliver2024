import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ****************************************************************************************
# ------------------------------ 动态生成位置编码 (取代原PositionalEncoding) -------------
# ****************************************************************************************
class DynamicPositionalEncoding(nn.Module):
    """
    动态生成正余弦位置编码，可选缓存以避免重复计算。
    """
    def __init__(self, d_hid, use_cache=True):
        super().__init__()
        self.d_hid = d_hid
        self.use_cache = use_cache
        # 可选：缓存，不同序列长度只需计算一次
        if self.use_cache:
            self._pe_cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape = (B, N, d_hid)，N=patch_num，d_hid=embedding_dim
        :return: x + PE, 其中 PE.shape=(1, N, d_hid)
        """
        B, N, C = x.shape
        assert C == self.d_hid, f"输入通道 {C} 与 d_hid={self.d_hid} 不匹配!"

        if self.use_cache and (N in self._pe_cache):
            pos_table = self._pe_cache[N]
        else:
            pos_table = self._build_position_encoding(N, C, x.device)
            if self.use_cache:
                self._pe_cache[N] = pos_table

        return x + pos_table

    def _build_position_encoding(self, n_position, d_hid, device):
        """
        生成 (1, n_position, d_hid) 大小的正余弦位置编码
        """
        pos = torch.arange(n_position, dtype=torch.float, device=device).unsqueeze(1)  # (N,1)
        dim_i = torch.arange(d_hid, dtype=torch.float, device=device).unsqueeze(0)     # (1,C)

        div_term = torch.exp((2 * (dim_i // 2)) * (-np.log(10000.0) / d_hid))
        angle = pos * div_term  # => (N, C)

        pe = torch.zeros((n_position, d_hid), device=device)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe.unsqueeze(0)  # => (1, N, d_hid)


# ****************************************************************************************
# ------------------------------ DUpsampling2D & DUpsampling3D  --------------------------
# ****************************************************************************************
class DUpsampling2D(nn.Module):
    """
    原上半部分的 sub-pixel 上采样，用通道拆分来升高空间分辨率。
    假设 input shape = (B, Cin, H, W)，scale^2 * Cin = Cout
    """
    def __init__(self, inplanes, scale):
        super(DUpsampling2D, self).__init__()
        output_channel = inplanes * (scale ** 2)
        self.conv_2d = nn.Conv2d(inplanes, output_channel, kernel_size=1, stride=1, bias=False)
        self.scale = scale

    def forward(self, x):
        # (B, Cin, H, W)
        x = self.conv_2d(x)  # => (B, Cin*scale^2, H, W)
        B, C, H, W = x.size()
        # rearrange => sub-pixel:  reshape -> permute -> reshape
        # 1)  reshape => (B, scale^2, C/scale^2, H, W)
        x = x.view(B, self.scale*self.scale, C // (self.scale*self.scale), H, W)
        # 2)  permute => (B, C/(s^2), H, s^2, W) but we actually want to reorder dims
        #     这里把 s^2 拆成 (s, s)
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        # => (B, C/(s^2), H, s^2, W)
        # 3)  flatten s^2 => (s,s) in spatial
        #     view => (B, C/(s^2), H*s, W*s)
        x = x.view(B, C // (self.scale**2), H*self.scale, W*self.scale)
        return x


class DUpsampling3D(nn.Module):
    """
    原上半部分的 3D sub-pixel 上采样，同理将通道维度拆分到 (D,H,W) 中。
    仅当你真的需要 3D 体数据 & sub-pixel 方式时才会用到。
    """
    def __init__(self, inplanes, scale):
        super(DUpsampling3D, self).__init__()
        output_channel = inplanes * (scale ** 3)
        self.conv_3d = nn.Conv3d(inplanes, output_channel, kernel_size=1, stride=1, bias=False)
        self.scale = scale

    def forward(self, x):
        # (B, Cin, D, H, W)
        x = self.conv_3d(x)  # => (B, Cin*scale^3, D, H, W)
        B, C, D, H, W = x.size()

        # Step1: permute for easy reshaping
        # => (B, W, H, D, C)
        x = x.permute(0, 4, 3, 2, 1)
        # => reshape => (B, W, H, D*scale, C/scale)
        x = x.contiguous().view(B, W, H, D * self.scale, C // self.scale)
        # => permute => (B, D*scale, W, H, C/scale)
        x = x.permute(0, 3, 1, 2, 4)
        x = x.contiguous().view(B, D * self.scale, W, H * self.scale, C // (self.scale**2))
        x = x.permute(0, 1, 3, 2, 4)
        x = x.contiguous().view(B, D * self.scale, H * self.scale, W * self.scale, C // (self.scale**3))
        x = x.permute(0, 4, 1, 2, 3)
        return x


# ****************************************************************************************
# ------------------------------------ 原上半部分 UNet 系列  -----------------------------
# ****************************************************************************************
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm_type='instance', use_dropout=True, bias=True):
        super(UNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps
            conv_block = UNetConvBlock3D(in_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i+1), pool)
            in_features = out_features

    def forward(self, inputs):
        encoder_outputs = []
        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            if i == self.levels-1:
                continue
            if self.use_dropout:
                encoder_outputs.append(self.dropout(outputs))
            else:
                encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i+1))(outputs)
        return encoder_outputs, outputs


class UNetEncoder_hved(nn.Module):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm_type='instance', use_dropout=True, bias=True):
        super(UNetEncoder_hved, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout()

        in_features = in_channels
        for i in range(levels):
            out_features = (2**i) * feature_maps
            conv_block = UNetConvBlock3D_hved(in_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i+1), conv_block)

            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i+1), pool)
            in_features = out_features

    def forward(self, inputs):
        encoder_outputs = []
        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
            if i == self.levels-1:
                continue
            if self.use_dropout:
                encoder_outputs.append(self.dropout(outputs))
            else:
                encoder_outputs.append(outputs)
            outputs = getattr(self.features, 'pool%d' % (i+1))(outputs)
        return encoder_outputs, outputs


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True):
        super(UNetDecoder, self).__init__()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels-1):
            upconv = UNetUpSamplingBlock3D(2**(levels-i-1) * feature_maps,
                                           2**(levels-i-1) * feature_maps, deconv=False, bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)

            conv_block = UNetConvBlock3D(2**(levels-i-2) * feature_maps * 3,
                                         2**(levels-i-2) * feature_maps,
                                         norm_type=norm_type, bias=bias, flag='decoder')
            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.score = nn.Conv3d(feature_maps, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        encoder_outputs.reverse()
        outputs = inputs
        for i in range(self.levels-1):
            outputs = getattr(self.features, 'upconv%d' % (i+1))(outputs)
            outputs = torch.cat([encoder_outputs[i], outputs], dim=1)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
        encoder_outputs.reverse()
        return self.score(outputs)


class UNetDecoder_hved(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True):
        super(UNetDecoder_hved, self).__init__()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels-1):
            upconv = UNetUpSamplingBlock3D(2**(levels-i-1) * feature_maps,
                                           2**(levels-i-1) * feature_maps, deconv=False, bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)

            conv_block = UNetConvBlock3D_hved(2**(levels-i-2) * feature_maps * 3,
                                              2**(levels-i-2) * feature_maps,
                                              norm_type=norm_type, bias=bias, flag='decoder')
            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.score = nn.Conv3d(feature_maps, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        encoder_outputs.reverse()
        outputs = inputs
        for i in range(self.levels-1):
            outputs = getattr(self.features, 'upconv%d' % (i+1))(outputs)
            outputs = torch.cat([encoder_outputs[i], outputs], dim=1)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
        encoder_outputs.reverse()
        return self.score(outputs)


class UNetUpSamplingBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock3D, self).__init__()
        self.deconv = deconv
        if self.deconv:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, *inputs):
        if len(inputs) == 2:
            return self.forward_concat(inputs[0], inputs[1])
        else:
            return self.forward_standard(inputs[0])

    def forward_concat(self, inputs1, inputs2):
        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):
        return self.up(inputs)


class UNetConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding='SAME', norm_type='instance', bias=True, flag='encoder'):
        super(UNetConvBlock3D, self).__init__()
        if flag == 'encoder':
            self.conv1 = ConvNormRelu3D(in_channels, out_channels//2, kernel_size=kernel_size,
                                        padding=padding, norm_type=norm_type, bias=bias)
            self.conv2 = ConvNormRelu3D(out_channels//2, out_channels, kernel_size=kernel_size,
                                        padding=padding, norm_type=norm_type, bias=bias)
        else:
            self.conv1 = ConvNormRelu3D(in_channels, out_channels, kernel_size=kernel_size,
                                        padding=padding, norm_type=norm_type, bias=bias)
            self.conv2 = ConvNormRelu3D(out_channels, out_channels, kernel_size=kernel_size,
                                        padding=padding, norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetConvBlock3D_hved(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding='SAME', norm_type='instance', bias=True, flag='encoder'):
        super(UNetConvBlock3D_hved, self).__init__()
        if flag == 'encoder':
            self.conv1 = NormRelu3DConv(in_channels, out_channels//2, kernel_size=kernel_size,
                                        padding=padding, norm_type=norm_type, bias=bias)
            self.conv2 = NormRelu3DConv(out_channels//2, out_channels, kernel_size=kernel_size,
                                        padding=padding, norm_type=norm_type, bias=bias)
        else:
            self.conv1 = NormRelu3DConv(in_channels, out_channels, kernel_size=kernel_size,
                                        padding=padding, norm_type=norm_type, bias=bias)
            self.conv2 = NormRelu3DConv(out_channels, out_channels, kernel_size=kernel_size,
                                        padding=padding, norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class ConvNormRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):
        super(ConvNormRelu3D, self).__init__()
        norm = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d
        p = kernel_size // 2 if padding.upper() == 'SAME' else 0

        self.unit = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=p, stride=stride, bias=bias, dilation=dilation),
            norm(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, inputs):
        return self.unit(inputs)


class NormRelu3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):
        super(NormRelu3DConv, self).__init__()
        norm = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d
        p = kernel_size // 2 if padding.upper() == 'SAME' else 0

        self.unit = nn.Sequential(
            norm(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=p, stride=stride, bias=bias, dilation=dilation)
        )

    def forward(self, inputs):
        return self.unit(inputs)


# ****************************************************************************************
# --------------------------- 关键：改造后的 TF_3D，支持任意尺寸  ------------------------
# ****************************************************************************************
class TF_3D(nn.Module):
    """
    在「上半部分」的 TF_3D 基础上，做了如下改动：
      1) 移除对 volumn_size 的依赖，改用 patch_dim + nn.AdaptiveAvgPool2d 固定下采样；
      2) 用 DynamicPositionalEncoding 动态生成位置编码；
      3) reproject 时继续使用 DUpsampling2D (或3D)，而不是 F.interpolate；
      4) 假设输入 (H, W) 可以整除 patch_dim，否则 sub-pixel会上采样后形状不匹配。
    """
    def __init__(self,
                 embedding_dim=1024,
                 patch_dim=8,
                 nhead=4,
                 num_layers=8,
                 method='TF'):
        super(TF_3D, self).__init__()
        self.embedding_dim = embedding_dim
        self.d_model = self.embedding_dim
        self.patch_dim = patch_dim
        self.method = method

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=nhead,
                                                   batch_first=True,
                                                   dim_feedforward=self.d_model * 4)
        self.fusion_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 动态位置编码，替代原先写死 200 的表格
        self.pos_enc = DynamicPositionalEncoding(d_hid=self.d_model, use_cache=True)
        self.dropout = nn.Dropout(p=0.1)

        # 自适应池化到 (patch_dim, patch_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((patch_dim, patch_dim))

        # 上半部分有 "Token" 可选
        if method == 'Token':
            self.fusion_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

    def forward(self, all_content):
        """
        all_content: list[tensor], 每个是 (B, C, H, W)，其中 C==embedding_dim 或需自己1x1 conv对齐
        """
        n_modality = len(all_content)
        if n_modality == 0:
            raise ValueError("all_content 为空！至少要有1路特征")
        B = all_content[0].size(0)

        # 1) 下采样到 (patch_dim, patch_dim)，然后 flatten => token
        token_seq = []
        for feat in all_content:
            # 确保通道一致
            assert feat.shape[1] == self.embedding_dim, \
                f"特征通道 {feat.shape[1]} != TF_3D.embedding_dim={self.embedding_dim}"
            pooled = self.avgpool(feat)  # => (B, embedding_dim, patch_dim, patch_dim)
            # => (B, patch_dim*patch_dim, embedding_dim)
            pooled = pooled.flatten(start_dim=2).transpose(1, 2).contiguous()
            token_seq.append(pooled)

        # 多模态拼接 => (B, sum_of_patches, embedding_dim)
        tokens = torch.cat(token_seq, dim=1)

        if self.method == 'Token':
            # 如果有 learnable token
            fused_token = self.fusion_token.expand(B, -1, -1)  # (B,1,embedding_dim)
            tokens = torch.cat([fused_token, tokens], dim=1)

        # 2) Transformer
        x = self.fusion_block(self.dropout(self.pos_enc(tokens)))

        # 3) 如果插入了 token，需要拆掉
        if self.method == 'Token':
            x_patch = x[:, 1:, :]
        else:
            x_patch = x

        # 4) 拆分回各模态 + sub-pixel 还原
        patch_per_mod = self.patch_dim * self.patch_dim
        outputs_upsampled = []
        idx = 0

        for i in range(n_modality):
            chunk = x_patch[:, idx : idx + patch_per_mod, :]  # => (B, patch_per_mod, embedding_dim)
            idx += patch_per_mod

            # => (B, embedding_dim, patch_dim, patch_dim)
            chunk = chunk.transpose(1, 2).contiguous().view(B, self.embedding_dim, self.patch_dim, self.patch_dim)

            # 计算 sub-pixel 上采样倍数
            # 假设输入feat的 (H, W) 是 patch_dim 的整数倍
            H_orig, W_orig = all_content[i].shape[-2], all_content[i].shape[-1]
            scale_h = H_orig // self.patch_dim
            scale_w = W_orig // self.patch_dim
            if scale_h != scale_w:
                raise ValueError(f"不支持非等比上采样 => scale_h={scale_h}, scale_w={scale_w}")
            scale = scale_h

            # sub-pixel => DUpsampling2D
            # sub-pixel 后通道变 embedding_dim/(scale^2)
            # 与 all_content[i] 融合(通道数要匹配)
            upsampler = DUpsampling2D(self.embedding_dim, scale).to(chunk.device)
            upchunk = upsampler(chunk)  # => (B, embedding_dim/(scale^2), H_orig, W_orig) ?

            # 这里如果要和 all_content[i] 做逐元素相乘/加和，需要通道匹配
            # all_content[i] 是 (B, embedding_dim, H_orig, W_orig)
            # sub-pixel 后 => (B, embedding_dim/(s^2), H_orig, W_orig)
            # => 可能通道数量不一致 => 你可以再加一层 1x1 conv 将通道变回 embedding_dim
            if upchunk.shape[1] != all_content[i].shape[1]:
                # 用一个1x1 卷积再升回embedding_dim
                conv_1x1 = nn.Conv2d(upchunk.shape[1], self.embedding_dim, kernel_size=1, bias=False).to(chunk.device)
                upchunk = conv_1x1(upchunk)

            outputs_upsampled.append(upchunk)

        # 5) 做 softmax 加权融合(与上半部分类似)
        #   stack => (n_modality, B, C, H, W) => softmax => element-wise * all_content[i]
        stack = torch.stack(outputs_upsampled, dim=0)  # => (n_modality,B,C,H,W)
        attn_map = F.softmax(stack, dim=0)             # => 同shape
        fused_output = None
        for i in range(n_modality):
            if fused_output is None:
                fused_output = all_content[i] * attn_map[i]
            else:
                fused_output += all_content[i] * attn_map[i]

        return fused_output


# -----------------------------------------------------------------------------------------
#  简单测试函数：演示在不同输入大小时的调用
# -----------------------------------------------------------------------------------------
def main():
    # 构建改造后的 TF_3D
    # 让 embedding_dim=64, patch_dim=8
    # nhead=4, num_layers=2, method='TF'
    fusion = TF_3D(embedding_dim=64, patch_dim=8, nhead=4, num_layers=2, method='TF')

    # 假设两路输入特征 [B=1, C=64, H=32, W=32]
    # 32可以整除patch_dim=8 => scale=4
    feat1 = torch.randn(1, 64, 256, 256)
    feat2 = torch.randn(1, 64, 256, 256)

    fused = fusion([feat1, feat2])
    print("Fused output shape:", fused.shape)  # => [1, 64, 32, 32]

    # 再试另一个尺寸，如 [B=1, C=64, H=48, W=48]
    # 48 // patch_dim=8 => scale=6
    # sub-pixel会把 embedding_dim=64 => 64/(6*6)=64/36=1.(需再1x1 conv升回64)
    feat3 = torch.randn(1, 64, 128, 128)
    feat4 = torch.randn(1, 64, 128, 128)
    fused2 = fusion([feat3, feat4])
    print("Fused2 output shape:", fused2.shape)

    feat3 = torch.randn(1, 64, 64, 64)
    feat4 = torch.randn(1, 64, 64, 64)
    fused3 = fusion([feat3, feat4])
    print("Fused2 output shape:", fused3.shape)

    feat3 = torch.randn(1, 64, 320, 320)
    feat4 = torch.randn(1, 64, 320, 320)
    fused3 = fusion([feat3, feat4])
    print("Fused2 output shape:", fused3.shape)

    feat3 = torch.randn(1, 64, 448, 448)
    feat4 = torch.randn(1, 64, 448, 448)
    fused3 = fusion([feat3, feat4])
    print("Fused2 output shape:", fused3.shape)

if __name__ == '__main__':
    main()
