
# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


from functools import partial
from timm.models.layers import to_2tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions.normal import Normal
import numpy as np



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


class ConvProcessor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        """
        初始化卷积处理器。

        参数：
        - in_channels (int): 输入张量的通道数。
        - out_channels (int): 输出张量的通道数。
        - kernel_size (int or tuple): 卷积核的大小。默认为3。
        - padding (int or tuple): 填充的大小。默认为1。
        """
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        """
        对包含多个张量的列表进行卷积操作。

        参数：
        - tensor_list (list of torch.Tensor): 包含多个张量的列表。

        返回：
        - list of torch.Tensor: 包含卷积后张量的列表。
        """
        return self.conv_layer(x)


class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)   # padding=(ps[0]//2, ps[1]//2)
        self.norm = ConvLayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x


class conv_lowrank(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            c1,
            c2,
            patch_size,
            stride,
            padding
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.extra_downsample_layers = PatchEmbedParallel(self.c1, self.c2, self.patch_size, self.stride, self.padding)

    def forward(self, x):
        x = self.extra_downsample_layers(x)
        return x


class Mlp_lowrank(nn.Module):  ####V2.0
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            c1,
            c2,
            patch_size,
            stride,
            padding
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.extra_downsample_layers = PatchEmbedParallel(self.c1, self.c2, self.patch_size, self.stride, self.padding)

    def forward(self, x):
        x = self.extra_downsample_layers(x)
        return x


# class Mlp_lowrank(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
#     def __init__(
#             self,
#             in_features,
#             hidden_features=None,
#             act_layer=nn.GELU,
#             norm_layer=None,
#             bias=True,
#             drop=0.,
#             use_conv=False,
#     ):
#         super().__init__()
#         hidden_features = hidden_features or in_features
#         bias = to_2tuple(bias)
#         drop_probs = to_2tuple(drop)
#         linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
#
#         self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop_probs[0])
#         self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
#         self.fc2 = linear_layer(hidden_features, hidden_features, bias=bias[1])
#         self.drop2 = nn.Dropout(drop_probs[1])
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.norm(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x




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
            use_conv=False
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



class Mlp_lowrank_star(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
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




    def forward(self, x, y):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        y = self.act(y)
        x = x*y
        return x






class Mlp_lowrank_ZW(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features*2, hidden_features*2, bias=bias[1])

        # self.fc2 = linear_layer(hidden_features, hidden_features*2, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, y):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x_org = self.norm(x)

        x = torch.concat((x_org, y), dim=-1)
        x = self.fc2(x)
        x = self.drop2(x)
        # x = torch.concat((x, y), dim=-1)
        return x








class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices

        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, top_gedits, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(1).unsqueeze(2))
            # stitched = stitched.mul(top_gedits.unsqueeze(1))

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(2), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class MoE_lora(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, c1, c2, patch_size, stride, padding, num_experts, width, noisy_gating=True, k=2):
        super(MoE_lora, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.c11 = c1
        self.c21 = c2
        self.patch_size1 = patch_size
        self.width = width
        self.stride1 = stride
        self.padding1 = padding
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([Mlp_lowrank(self.c11, self.c21, self.patch_size1, self.stride1, self.padding1) for i in range(self.num_experts)])
        self.shared_expert = Mlp_lowrank(self.c11, self.c21, self.patch_size1, self.stride1, self.padding1)
        self.w_gate = nn.Parameter(torch.zeros(self.width, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.width, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        x_mean = torch.mean(x, dim=-1).mean(dim=1)
        # x_flat = x.reshape(x.shape[0], self.ch*self.width*self.height)
        # x_mean = torch.mean(x_mean, dim=-1)
        # z_mean = torch.mean(z, dim=-1)
        # gate_1 = self.shared_expert(x).mean(dim=-1).mean(dim=1)
        # print(gate_1.shape)

        clean_logits = x_mean @ self.w_gate

        if self.noisy_gating and train:
            raw_noise_stddev = x_mean @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization
        # self.logits = top_k_logits
        # self.act =  nn.GELU()

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, self.softmax(logits), top_k_logits



    def forward(self, xs, loss_coef=1e-2):
        total_loss = 0
        final_outputs = []
        for x in xs:

            gates, load, logits, top_k_logits = self.noisy_top_k_gating(x, self.training)

            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef
            total_loss += loss

            dispatcher = SparseDispatcher(self.num_experts, gates)
            shared_x = self.shared_expert(x)

            expert_inputs_x = dispatcher.dispatch(x)

            # gates = dispatcher.expert_to_gates()

            expert_outputs_x = [self.experts[i](expert_inputs_x[i]) for i in range(self.num_experts)]

            x_res = dispatcher.combine(expert_outputs_x, top_k_logits) + shared_x


            final_outputs.append(x_res)

        return final_outputs, total_loss


class MoE_lora_new(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, c1, c2, patch_size, stride, padding, num_experts, width, noisy_gating=True, k=2):
        super(MoE_lora_new, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.c11 = c1
        self.c21 = c2
        self.patch_size1 = patch_size
        self.width = width
        self.stride1 = stride
        self.padding1 = padding
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([Mlp_lowrank(self.c11, self.c21, self.patch_size1, self.stride1, self.padding1) for i in range(self.num_experts)])
        self.shared_expert = Mlp_lowrank(self.c11, self.c21, self.patch_size1, self.stride1, self.padding1)
        self.w_gate = nn.Parameter(torch.zeros(self.width, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.width, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        x_mean = torch.mean(x, dim=-1).mean(dim=1)
        # x_flat = x.reshape(x.shape[0], self.ch*self.width*self.height)
        # x_mean = torch.mean(x_mean, dim=-1)
        # z_mean = torch.mean(z, dim=-1)
        # gate_1 = self.shared_expert(x).mean(dim=-1).mean(dim=1)
        # print(gate_1.shape)
        _, height = x_mean.shape
        _, weight = self.w_gate.shape
        target_size = (1, height, weight)
        # print(self.w_gate.shape, target_size)
        # raise Exception# Specify the target size
        w_gate_reshaped = self.w_gate.unsqueeze(0).permute(0, 2, 1)
        w_gate_new = F.interpolate(w_gate_reshaped, size=(height, ), mode='linear',
                                   align_corners=False)
        # 去掉批次维度，恢复为 [height, weight] 的形状
        w_gate_new = w_gate_new.squeeze(0).permute(1, 0)
        clean_logits = x_mean @ w_gate_new

        if self.noisy_gating and train:
            raw_noise_stddev = x_mean @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization
        # self.logits = top_k_logits
        # self.act =  nn.GELU()

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, self.softmax(logits), top_k_logits

    def compute_symmetric_kl_loss(self, p, q):
        """
        Compute the symmetric KL divergence between two distributions.
        Args:
            p: Log probabilities of the first distribution.
            q: Probabilities of the second distribution.
        Returns:
            Symmetric KL divergence.
        """
        kl_pq = F.kl_div(p, q, reduction='batchmean')
        kl_qp = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        return kl_pq + kl_qp



    def forward(self, xs, loss_coef=1e-2):
        total_expert_loss = 0
        total_loss = 0
        final_shared_outputs = []
        final_diff_outputs = []
        for x in xs:

            gates, load, logits, top_k_logits = self.noisy_top_k_gating(x, self.training)

            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef
            total_expert_loss += loss

            dispatcher = SparseDispatcher(self.num_experts, gates)
            shared_x = self.shared_expert(x)

            expert_inputs_x = dispatcher.dispatch(x)

            # gates = dispatcher.expert_to_gates()

            expert_outputs_x = [self.experts[i](expert_inputs_x[i]) for i in range(self.num_experts)]

            x_res = dispatcher.combine(expert_outputs_x, top_k_logits)


            final_diff_outputs.append(x_res)
            final_shared_outputs.append(shared_x)

        shared_stacked_tensor = torch.stack(final_shared_outputs)
        shared_mean_tensor = torch.mean(shared_stacked_tensor, dim=0)

        # Compute uniformity loss (mean squared error between shared and individual features)
        uniformity_loss = 0
        for diff_output in final_diff_outputs:
            uniformity_loss += F.mse_loss(shared_mean_tensor, diff_output)

        # Compute distinctiveness loss (KL divergence between individual features)
        distinctiveness_loss = 0
        for i in range(len(final_diff_outputs)):
            for j in range(i + 1, len(final_diff_outputs)):
                p = F.log_softmax(final_diff_outputs[i].view(final_diff_outputs[i].size(0), -1), dim=-1)
                q = F.softmax(final_diff_outputs[j].view(final_diff_outputs[j].size(0), -1), dim=-1)
                distinctiveness_loss += self.compute_symmetric_kl_loss(p, q)

        # Ensure total_loss is non-negative and balanced
        total_loss += uniformity_loss + 0.1 * distinctiveness_loss + total_expert_loss



        return final_diff_outputs, shared_mean_tensor, total_loss



class AttentionWeightedSum(nn.Module):
    def __init__(self):
        super(AttentionWeightedSum, self).__init__()

    def forward(self, tensor_list, reference_tensor):
        # 初始化一个张量来存储加权结果
        weighted_tensors = []

        # 计算每个张量的注意力权重并应用
        for tensor in tensor_list:
            # 计算注意力权重（点积）
            attention_scores = torch.sum(reference_tensor * tensor, dim=(1, 2, 3), keepdim=True)

            # 归一化权重（使用 softmax）
            attention_weights = F.softmax(attention_scores, dim=0)

            # 应用权重
            weighted_tensor = tensor * attention_weights
            weighted_tensors.append(weighted_tensor)


        return weighted_tensors



class ConcatAndConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConcatAndConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, tensor_list):
        # 1. 在通道维度上拼接张量
        concatenated_tensor = torch.cat(tensor_list, dim=1)  # 在通道维度上拼接

        # 2. 应用卷积层
        output_tensor = self.conv(concatenated_tensor)

        return output_tensor


class FinalConvProcessor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        """
        初始化卷积处理器。

        参数：
        - in_channels (int): 输入张量的通道数。
        - out_channels (int): 输出张量的通道数。
        - kernel_size (int or tuple): 卷积核的大小。默认为3。
        - padding (int or tuple): 填充的大小。默认为1。
        """
        super(FinalConvProcessor, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, tensor1, tensor2):
        """
        对两个张量进行残差操作，然后进行卷积和归一化。

        参数：
        - tensor1 (torch.Tensor): 第一个输入张量。
        - tensor2 (torch.Tensor): 第二个输入张量。

        返回：
        - torch.Tensor: 卷积和归一化后的张量。
        """
        residual = tensor1 + tensor2  # 残差操作
        out = self.conv_layer(residual)  # 卷积操作
        out = self.batch_norm(out)  # 归一化操作
        return out




class AllInOne_lora(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    # def __init__(self, c1, c_inter, c2, patch_size1, patch_size2, stride, padding, num_experts, width, noisy_gating=True, k=2):
    def __init__(self, c1, c_inter, patch_size1, patch_size2, stride, padding, num_experts, width,
                 noisy_gating=True, k=2):
        super(AllInOne_lora, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.c11 = c1
        self.c_inter = c_inter
        # self.c21 = c2
        self.patch_size1 = patch_size1
        self.patch_size2 = patch_size2
        self.width = width
        self.stride1 = stride
        self.padding1 = padding
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList(
            [conv_lowrank(self.c11, self.c_inter, self.patch_size1, self.stride1, self.padding1) for i in
             range(self.num_experts)])
        self.shared_expert = PatchEmbedAndMlp(self.c11, self.patch_size2, self.c_inter)
        self.attn_gate = AttentionWeightedSum()
        self.concat_conv = ConcatAndConv(3*self.c_inter, self.c_inter)
        # self.back_convs = nn.ModuleList([ConvProcessor(self.c_inter, self.c21) for i in range(self.num_experts)])
        # self.final_conv = FinalConvProcessor(self.c_inter, self.c21)
        self.w_gate = nn.Parameter(torch.zeros(self.width, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.width, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        x_mean = torch.mean(x, dim=-1).mean(dim=1)
        # x_flat = x.reshape(x.shape[0], self.ch*self.width*self.height)
        # x_mean = torch.mean(x_mean, dim=-1)
        # z_mean = torch.mean(z, dim=-1)
        # gate_1 = self.shared_expert(x).mean(dim=-1).mean(dim=1)
        # print(gate_1.shape)

        clean_logits = x_mean @ self.w_gate

        if self.noisy_gating and train:
            raw_noise_stddev = x_mean @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization
        # self.logits = top_k_logits
        # self.act =  nn.GELU()

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, self.softmax(logits), top_k_logits

    def forward(self, xs, loss_coef=1e-2):
        total_loss = 0
        final_expert_outputs, final_shared_outputs = [], []
        for x in xs:
            gates, load, logits, top_k_logits = self.noisy_top_k_gating(x, self.training)

            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef
            total_loss += loss

            dispatcher = SparseDispatcher(self.num_experts, gates)
            shared_x = self.shared_expert(x)

            expert_inputs_x = dispatcher.dispatch(x)

            # gates = dispatcher.expert_to_gates()

            expert_outputs_x = [self.experts[i](expert_inputs_x[i]) for i in range(self.num_experts)]
            x_res = dispatcher.combine(expert_outputs_x, top_k_logits)

            final_expert_outputs.append(x_res)
            final_shared_outputs.append(shared_x)

        shared_stacked_tensor = torch.stack(final_shared_outputs)
        shared_mean_tensor = torch.mean(shared_stacked_tensor, dim=0)
        expert_combine_outputs = self.attn_gate(final_expert_outputs, shared_mean_tensor)
        expert_combine_output = self.concat_conv(expert_combine_outputs)

        final_ext_output = [torch.cat((x_, shared_mean_tensor), dim=1) for x_ in expert_combine_outputs]
        # final_ext_output = [self.back_convs[i](expert_combine_outputs[i]) for i in range(self.num_experts)]
        final_output = torch.cat((expert_combine_output, shared_mean_tensor), dim=1)
        # final_output = self.final_conv(expert_combine_output, shared_mean_tensor)

        return final_ext_output, final_output, total_loss



if "__main__"==__name__:
    # moe_instance = AllInOne_lora(3, 32, 7, 4, 4, 7//2, 6, 1024, True, 2)
    # moe_instance = AllInOne_lora(64, 64, 3, 2, 2, 3//2, 6, 256, True, 2)
    # moe_instance = AllInOne_lora(128, 160, 3, 2, 2, 3//2, 6, 128, True, 2)
    # moe_instance = AllInOne_lora(320, 256, 3, 2, 2, 3//2, 6, 64, True, 2)
    moe_instance = MoE_lora_new(3, 64, 7, 4, 7//2, 6, 1024, True, 2)

    # tensor_1 = torch.randn(1, 3, 1024, 1024)
    # tensor_2 = torch.randn(1, 3, 1024, 1024)
    # tensor_3 = torch.randn(1, 3, 1024, 1024)
    # tensor_4 = torch.randn(1, 3, 1024, 1024)

    tensor_1 = [torch.ones(2, 3, 1024, 1024), torch.ones(2, 3, 1024, 1024)*2, torch.ones(2, 3, 1024, 1024) *3]
    # tensor_1 = [torch.ones(2, 64, 256, 256), torch.ones(2, 64, 256, 256) * 2, torch.ones(2, 64, 256, 256) * 3]
    # tensor_1 = [torch.ones(2, 128, 128, 128), torch.ones(2, 128, 128, 128) * 2, torch.ones(2, 128, 128, 128) * 3]
    # tensor_1 = [torch.ones(2, 320, 64, 64), torch.ones(2, 320, 64, 64) * 2, torch.ones(2, 320, 64, 64) * 3]
    final_diff_outputs, shared_mean_tensor, total_loss = moe_instance(tensor_1)
    print(len(final_diff_outputs), shared_mean_tensor.shape, total_loss)





