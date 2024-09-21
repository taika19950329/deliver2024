import torch
import torch.nn.functional as F


def compute_infonce_loss(anchor, positive, negatives):
    """
    计算 InfoNCE 损失
    anchor: 锚点特征 (如 RGB 特征) [B, D]
    positive: 正样本特征 (如 Depth 特征) [B, D]
    negatives: 负样本特征 (batch 内其他样本) [B, N-1, D], N 是 batch size
    """
    # 计算正样本的相似度 (anchor 和正样本的点积)
    positive_sim = torch.sum(anchor * positive, dim=-1)  # [B]

    # 计算负样本的相似度 (anchor 和负样本的点积)
    negative_sim = torch.einsum('bd,bnd->bn', anchor, negatives)  # [B, N-1]

    # 拼接正负样本的相似度，正样本位于第 0 类
    logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)  # [B, 1 + (N-1)]

    # 使用交叉熵损失，正样本对应的标签应为 0
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)  # [B]

    # 计算 InfoNCE 损失
    loss = F.cross_entropy(logits, labels)
    return loss


# 定义输入数据示例
B, C, H, W = 8, 64, 32, 32  # Batch Size, Channels, Height, Width (假设特征提取后的尺寸)

rgb_input = torch.randn(B, C, H, W)  # RGB 特征输入 [B, C, H, W]
depth_input = torch.randn(B, C, H, W)  # Depth 特征输入 [B, C, H, W]

# 假设已经提取特征并保持尺寸对齐 [B, C, H, W]
# 展平特征为 [B, C*H*W]
rgb_feat = rgb_input.view(B, -1)  # [B, C*H*W]
depth_feat = depth_input.view(B, -1)  # [B, C*H*W]

# L2 归一化
rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
depth_feat = F.normalize(depth_feat, p=2, dim=1)

# 遍历 batch 中的每个样本
losses = []
for i in range(B):
    # 当前样本的 RGB 作为 anchor
    anchor = rgb_feat[i]  # [C*H*W]

    # 当前样本的 Depth 作为正样本
    positive = depth_feat[i]  # [C*H*W]

    # 其他样本的 Depth 作为负样本
    negatives = torch.cat([depth_feat[j].unsqueeze(0) for j in range(B) if j != i], dim=0)
    print(len(negatives))# [B-1, C*H*W]
    negatives = negatives.unsqueeze(0).repeat(1, 1, 1)  # [1, B-1, C*H*W]

    # 计算 InfoNCE 损失
    loss = compute_infonce_loss(anchor.unsqueeze(0), positive.unsqueeze(0), negatives)
    losses.append(loss)

# 平均损失
print(losses)
total_loss = torch.mean(torch.stack(losses))

print(f'RGB-Depth InfoNCE Loss: {total_loss.item()}')
