import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Module):
    def __init__(
        self,
        r: int = 4,                 # lora rank
        alpha: int = 4,            # lora alpha
        in_dim: int = 64,
        dropout_p: float = 0.0,     # lora dropout
        test_mode: bool = False,    # 测试模式，用于控制 lora_B 是否为全零
    ):
        super(LoraLinear, self).__init__()

        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)

        # 定义 lora_A 和 lora_B 为 Parameter
        self.lora_A = nn.Parameter(torch.empty((r, in_dim)))
        self.lora_B = nn.Parameter(torch.empty((in_dim, r)))

        # 初始化 lora 矩阵
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        if test_mode:
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling = float(self.alpha) / float(self.r)     # lora 缩放系数
        lora_adjustment = F.linear(self.dropout(x), self.lora_A)
        lora_adjustment = F.linear(lora_adjustment, self.lora_B)
        return lora_adjustment * scaling


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, input_dim=768, r=4, alpha=4, dropout_p=0.1):
        super().__init__()

        # Adapter layers
        self.adapter_down = nn.Linear(input_dim, dim)
        self.adapter_up = nn.Linear(dim, input_dim)
        self.adapter_mid = nn.Linear(dim, dim)

        # LoRA down and up projection layers
        self.lora_down = LoraLinear(r=r, alpha=alpha, in_dim=input_dim, dropout_p=dropout_p)
        self.lora_up = LoraLinear(r=r, alpha=alpha, in_dim=input_dim, dropout_p=dropout_p)

        # Initialization
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # Activation and dropout
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        # Down projection
        x_down = self.adapter_down(x.permute(0, 2, 1))  # [b, h*w, dim]
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)

        # Up projection
        x_up = self.adapter_up(x_down)  # [b, h*w, input_dim]
        x_up = x_up  # Return to shape [b, c, h * w]

        # Down projection - LoRA
        x_down_lora = self.lora_down(x.permute(0, 2, 1))

        # Up projection - LoRA
        x_up_lora = self.lora_up(x_down_lora)

        # Combine LoRA and original adapter outputs
        x_up_combined = x_up + x_up_lora  # Element-wise addition
        x_up_combined = x_up_combined.permute(0, 2, 1)

        return x_up_combined



class Shared_direct_adapter(nn.Module):
    def __init__(self, rank, r=4, alpha=4, dropout_p=0.1):  # Default to nn.LayerNorm
        super(Shared_direct_adapter, self).__init__()
        # self.norm_x = nn.LayerNorm(rank)
        # self.norm_xi = nn.LayerNorm(rank)
        self.proj_x = nn.Linear(rank, rank)
        self.proj_xi = nn.Linear(rank, rank)
        self.act = QuickGELU()  # Use QuickGELU activation
        self.proj_out = nn.Linear(rank, rank)

        # LoRA projections for proj_x and proj_xi
        self.lora_proj_x = LoraLinear(r=r, alpha=alpha, in_dim=rank, dropout_p=dropout_p)
        self.lora_proj_xi = LoraLinear(r=r, alpha=alpha, in_dim=rank, dropout_p=dropout_p)

    def forward(self, x, xi):
        batch_size, channels, hw = x.shape
        x = x.permute(0, 2, 1).contiguous().view(batch_size * hw, channels)  # Flatten (batch_size * h * w, channels)
        xi = xi.permute(0, 2, 1).contiguous().view(batch_size * hw, channels)

        # Normalize and project inputs
        # x = self.norm_x(x)
        # xi = self.norm_xi(xi)

        # Reshape back to the original shape
        x = x.view(batch_size, channels, hw)  # Shape back to [batch_size, channels, h * w]
        xi = xi.view(batch_size, channels, hw)

        # Apply projections
        x_proj = self.proj_x(x.permute(0, 2, 1))
        xi_proj = self.proj_xi(xi.permute(0, 2, 1))

        # LoRA projections
        x_proj_lora = self.lora_proj_x(x.permute(0, 2, 1))
        xi_proj_lora = self.lora_proj_xi(xi.permute(0, 2, 1))

        # Combine standard and LoRA projections
        x_proj = x_proj + x_proj_lora
        xi_proj = xi_proj + xi_proj_lora

        # Activation on xi_proj and element-wise multiplication
        activated_xi_proj = self.act(xi_proj)

        combined = x_proj * activated_xi_proj
        combined = combined.permute(0, 2, 1)

        return combined



# 使用示例
if __name__ == "__main__":
    # 定义输入参数
    batch_size = 2
    input_dim = 64
    dim = input_dim // 2
    height = 256
    width = 256
    channels = input_dim

    # 创建一个随机输入张量，形状为 [b, c, h * w]
    x = torch.rand(batch_size, channels, height * width)
    y = torch.rand(batch_size, channels, height * width)

    # 初始化 Bi_direct_adapter
    bi_adapter = Bi_direct_adapter(dim=dim, input_dim=input_dim)
    shared_adapter = Shared_direct_adapter(rank=channels)

    # 前向传播
    output_x = bi_adapter(x)
    output_y = bi_adapter(y)

    output_xy = shared_adapter(x, y)
    output_yx = shared_adapter(y, x)

    # 输出结果的形状和内容
    print(output_x.shape, output_y.shape, output_xy.shape, output_yx.shape)
    # print("Output tensor:", output_tensor)
