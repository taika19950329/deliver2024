import torch
import torch.nn as nn
from functools import partial

# Helper function to convert a single value to a tuple if it isn't already
def to_2tuple(value):
    if isinstance(value, (list, tuple)):
        return value
    return (value, value)

class Mlp_lowrank(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, hidden_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class PatchEmbedAndMlp(nn.Module):
    def __init__(self, in_channels, patch_size, mlp_hidden_features, act_layer=nn.GELU, norm_layer=None, bias=True, drop=0., use_conv=False):
        super().__init__()
        self.patch_size = patch_size
        self.mlp = Mlp_lowrank(
            in_features=in_channels * patch_size * patch_size,
            hidden_features=mlp_hidden_features,
            act_layer=act_layer,
            norm_layer=norm_layer,
            bias=bias,
            drop=drop,
            use_conv=use_conv
        )

    def forward(self, x):
        B, C, H, W = x.shape
        patch_size = self.patch_size
        new_H = H // patch_size
        new_W = W // patch_size

        # 重新调整形状以适应每个块的处理
        x = x.view(B, C, new_H, patch_size, new_W, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B * new_H * new_W, C * patch_size * patch_size)

        # 应用 MLP
        x = self.mlp(x)

        # 重塑回 (B, new_H, new_W, hidden_features)
        x = x.view(B, new_H, new_W, -1)

        # 调整形状为 (B, hidden_features, new_H, new_W)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

if "__main__"==__name__:
    input_tensor = torch.randn(2, 320, 64, 64)  # 假设输入张量为 (2, 64, 256, 256)
    in_channels = 320
    patch_size = 2  # 每个小块的大小 [4, 2, 2, 2]
    mlp_hidden_features = 256  # 目标输出特征数 [32, 64, 160, 256]

    model = PatchEmbedAndMlp(in_channels=in_channels, patch_size=patch_size, mlp_hidden_features=mlp_hidden_features, use_conv=False)

    output_tensor = model(input_tensor)

    print(output_tensor.shape)  # 输出应该是 (2, 64, 128, 128)
