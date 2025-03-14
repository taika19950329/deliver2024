import torch
from torch import nn, Tensor
from torch.nn import functional as F
import functools
from torch.autograd import Variable
# from semseg.models.modules.biapter import Shared_direct_adapter as ShareApt




def weighted_fusion_initialization(variances, epsilon=1e-6):
    # Stack means and variances along a new dimension for modalities, resulting in shape [m, b, c, h, w]
    # means_stack = torch.stack(means, dim=0)
    variances_stack = torch.stack(variances, dim=0)

    # Clamp variances to prevent division by zero
    variances_stack = torch.clamp(variances_stack, min=epsilon)

    # Calculate inverse variance weights
    weights = 1 / variances_stack  # Shape [m, b, c, h, w]

    # Compute the weighted sum of means and the sum of weights
    # weighted_sum_means = (means_stack * weights).sum(dim=0)  # Shape [b, c, h, w]
    sum_weights = weights.sum(dim=0)  # Shape [b, c, h, w]

    # Calculate the initial fused mean and variance
    # mu_init = weighted_sum_means / sum_weights  # Shape [b, c, h, w]
    sigma_init_squared = 1 / sum_weights  # Initial variance, shape [b, c, h, w]

    # return mu_init, sigma_init_squared
    return sigma_init_squared


def particle_filter_fusion(means, variances, mu_fused, sigma_fused_squared, noise_threshold,
                                       num_particles=2):
    """
    使用粒子滤波对多个模态的特征进行融合。

    Args:
        means (list[torch.Tensor]): 每个模态的均值列表，形状为 (b, n, c)。
        variances (list[torch.Tensor]): 每个模态的方差列表，形状为 (b, n, c)。
        num_particles (int): 粒子数目，用于表示每个模态特征的采样粒子。

    Returns:
        torch.Tensor: 融合后的均值，形状为 (b, n, c)。
        torch.Tensor: 融合后的方差，形状为 (b, n, c)。
    """
    epsilon = 1e-6

    # 初始化粒子集
    B, N, C = means[0].shape

    # 初始化共享粒子集
    mean_combined = torch.clamp(torch.stack(means, dim=0).mean(dim=0), min=epsilon)  # 合并所有模态均值
    variance_combined = torch.clamp(torch.stack(variances, dim=0).mean(dim=0), min=epsilon)  # 合并所有模态方差

    if torch.all(mean_combined == 0) and torch.all(variance_combined == 0):
        return mean_combined, variance_combined

    # 初始化粒子和权重
    # print("variance_combined min:", variance_combined.min().item())
    # print("variance_combined max:", variance_combined.max().item())
    std_dev = torch.clamp(torch.sqrt(variance_combined + epsilon), min=epsilon)
    # print("std_dev min:", std_dev.min().item())
    # print("std_dev max:", std_dev.max().item())
    particles = mean_combined.unsqueeze(0) + std_dev.unsqueeze(0) * torch.randn((num_particles,) + mean_combined.shape).to(mean_combined.device)
    weights = torch.ones((num_particles,) + mean_combined.shape).to(mean_combined.device) / num_particles


    # 粒子滤波循环更新
    for i in range(1, len(means)):
        # 扩展 variances[i]，使其形状与 particles[i] 一致
        variance_expanded = torch.clamp(variances[i].unsqueeze(0).expand(num_particles, *variances[i].shape), min=epsilon)
        mean_expanded = torch.clamp(means[i].unsqueeze(0).expand(num_particles, *means[i].shape), min=epsilon)  # [num_particles, b, c, h, w]

        # 计算粒子权重基于当前模态的观测值e)
        # print("Particles min:", particles.min().item())
        # print("Particles max:", particles.max().item())
        # print("Mean expanded min:", mean_expanded.min().item())
        # print("Mean expanded max:", mean_expanded.max().item())
        # raise Exception
        diff = particles - mean_expanded  # 当前粒子与观测值的偏差
        # print("Diff min:", diff.min().item())
        # print("Diff max:", diff.max().item())
        weights_update = torch.clamp(torch.exp(-0.5 * torch.sum((diff ** 2) / variance_expanded, dim=-1, keepdim=True)), min=epsilon)

        # 更新权重
        weights *= weights_update
        weights /= torch.sum(weights, dim=0, keepdim=True)  # 对权重进行归一化
        weights = torch.clamp(weights, min=epsilon)  # 避免数值过小

        # 重采样
        # print("Min weight before update:", weights.min())
        indices = torch.multinomial(weights.permute(1, 2, 3, 0).reshape(-1, num_particles), num_particles, replacement=True)  # 重采样索引
        indices = indices.permute(1, 0).reshape(num_particles, B, N, C)  # 将索引调整为粒子形状
        particles = torch.gather(particles, 0, indices)  # 根据索引进行重采样

        # 计算融合结果：加权求和得到最终的均值和方差
    fused_mean = torch.sum(particles * weights, dim=0) / torch.sum(weights, dim=0)
    fused_variance = torch.sum(weights * (particles - fused_mean.unsqueeze(0)) ** 2, dim=0) / torch.sum(weights, dim=0)

    outliers = torch.abs(fused_mean - mu_fused) > noise_threshold * torch.sqrt(sigma_fused_squared)
    fused_mean = torch.where(outliers, mu_fused, fused_mean)
    fused_variance = torch.where(outliers, sigma_fused_squared + epsilon, fused_variance)


    return fused_mean, fused_variance


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


class BiParticFusion(nn.Module):
    def __init__(self, inplanes, hide_channel):
        super(BiParticFusion, self).__init__()

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
        feat1 = self.gateproj1(feature_1, feature_2)  # bi-apt
        feat2 = self.gateproj2(feature_2, feature_1)

        # print(feat1.shape)
        # raise Exception

        # feat1 = self.gateproj1(feature_1.permute(0, 2, 1), feature_2.permute(0, 2, 1)).permute(0, 2, 1)    # bi-apt
        # feat2 = self.gateproj2(feature_2.permute(0, 2, 1), feature_1.permute(0, 2, 1)).permute(0, 2, 1)   # bi-apt

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

        # Particle filter update

        sigma_fused_w = weighted_fusion_initialization(vars)
        mu_w = weighted_fusion_initialization(means)

        mean = self.fuse_mean(mu_w)
        var = self.fuse_var(sigma_fused_w)

        noise_threshold = mean


        # fused_mean, fused_variance = kalman_fusion_feature_map(means, vars, mu_w, sigma_fused_w, noise_threshold)
        fused_mean, fused_variance = particle_filter_fusion(means, vars, mu_w, sigma_fused_w, noise_threshold)
        # import pdb;
        # pdb.set_trace()

        del means, vars, sigma_fused_w
        fused_variance = torch.log(fused_variance + epsilon)
        # proportion = 0.5
        fused_variance = self.quality(fused_variance, var)
        fused = self.reparametrize(fused_mean, fused_variance)

        # fused = self.reparametrize(fused_mean, proportion*fused_variance + (1-proportion) * var)

        output = self.proj_back(fused)

        return output



if __name__ == "__main__":
    # 假设我们有两个模态的特征

    device = torch.device('cpu')
    b, c, h, w = 4, 64, 32, 32  # 假设批次大小为4，通道数为64，图像尺寸为32x32
    num_particles = 100  # 粒子数目

    # 生成模拟数据
    feature_1_mean = torch.randn(b, h*w, c).to(torch.float32).to(device)  # 模态1的均值
    feature_1_variance = torch.abs(torch.randn(b, h*w, c)).to(torch.float32).to(device)  # 模态1的方差

    feature_2_mean = torch.randn(b, h*w, c).to(torch.float32).to(device)  # 模态2的均值
    feature_2_variance = torch.abs(torch.randn(b, h*w, c)).to(torch.float32).to(device)  # 模态2的方差

    # 将模态的均值和方差放入列表
    means = [feature_1_mean, feature_2_mean]
    variances = [feature_1_variance, feature_2_variance]

    fuse1 = BiParticFusion(c, 8).to(device)
    fused = fuse1(feature_1_mean, feature_1_variance)
    print(fused.shape)
    print(fused)
    raise Exception

    # 调用粒子滤波融合函数
    fused_mean, fused_variance = particle_filter_fusion(means, variances, num_particles)

    # 打印融合后的均值和方差的形状
    print(f'Fused Mean Shape: {fused_mean.shape}')
    print(f'Fused Variance Shape: {fused_variance.shape}')