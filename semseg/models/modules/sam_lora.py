from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    # reshape(B, N, 3, self.num_heads, self.head_dim) 重新塑形操作将线性层的输出重构为一个五维张量。
    # 其中，3 代表 QKV，self.num_heads 是注意力头的数量，
    # .permute(2, 0, 3, 1, 4) 重新排列这个五维张量的维度。这是为了使得查询、键和值在张量的第一维（即 0 维）上分离。
    # 以QKV维度表示张量
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    # 拆分
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        # x 25 14 14 768
        # qkv 25 14 14 2304
        qkv = self.qkv(x)
        # 25 14 14 768
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    '''
    确定应用LoRA的层：
    决定哪些层将应用LoRA微调是重要的，因为可能并不需要或不希望对所有层进行微调。这种选择可以基于模型的特定架构或预期的微调效果。

    初始化存储低秩矩阵的列表：
    创建 self.w_As 和 self.w_Bs 列表用于存储LoRA的低秩矩阵。这些矩阵是LoRA微调的核心，它们允许模型进行有效的参数更新。

    冻结原始模型的参数：
    通过将 requires_grad 设置为 False，冻结原始 Sam 模型的参数。这样做的目的是保留预训练模型的大部分知识，同时仅通过LoRA的低秩矩阵进行微调。

    遍历并修改指定层：
    这个过程是LoRA微调的核心。对于每个指定的层，替换原始的QKV线性层，引入新的线性层用于生成低秩更新。这种“手术式”修改使得模型在保持原有架构的基础上获得了额外的适应性。

    重置参数：
    self.reset_parameters() 可能用于初始化新加入的参数或重置现有参数，确保模型开始训练时处于良好状态。

    保存原始模型的引用：
    将原始 Sam 模型作为属性存储，使得微调后的模型可以在需要时访问原始模型的其他部分或属性。
    '''

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))
        self.w_As = []
        self.w_Bs = []

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        # 遍历图像编码器的每一层。返回索引和这个块本身
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # 如果当前层不在指定的LoRA层中，则跳过此层。
            if t_layer_i not in self.lora_layer:
                continue
            # 提取原始的QKV线性层。
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            # 创建两组线性层（w_a_linear_q/v 和 w_b_linear_q/v），用于生成低秩更新。
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            # 将这些线性层添加到列表 self.w_As 和 self.w_Bs。
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            # 用 _LoRA_qkv 类替换原始的QKV线性层，该类融合了原始的线性层和新的低秩更新。
            # 这个过程是LoRA微调的核心。对于每个指定的层，替换原始的QKV线性层，引入新的线性层用于生成低秩更新。
            # 这种“手术式”修改使得模型在保持原有架构的基础上获得了额外的适应性。
            # 更改取到的块本身里面的注意力里面的QKV，改成加了低秩矩阵的。
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model
        # self.blk = sam_model.image_encoder.blocks[0:3]

    def save_lora_parameters(self, filename: str) -> None:

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)
        # 保存lora的那部分权重
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam,
                                                                     torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
        # 合并lora权重和原始dict
        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)


class LoRA_Sam_submodel(nn.Module):
    '''
    确定应用LoRA的层：
    决定哪些层将应用LoRA微调是重要的，因为可能并不需要或不希望对所有层进行微调。这种选择可以基于模型的特定架构或预期的微调效果。

    初始化存储低秩矩阵的列表：
    创建 self.w_As 和 self.w_Bs 列表用于存储LoRA的低秩矩阵。这些矩阵是LoRA微调的核心，它们允许模型进行有效的参数更新。

    冻结原始模型的参数：
    通过将 requires_grad 设置为 False，冻结原始 Sam 模型的参数。这样做的目的是保留预训练模型的大部分知识，同时仅通过LoRA的低秩矩阵进行微调。

    遍历并修改指定层：
    这个过程是LoRA微调的核心。对于每个指定的层，替换原始的QKV线性层，引入新的线性层用于生成低秩更新。这种“手术式”修改使得模型在保持原有架构的基础上获得了额外的适应性。

    重置参数：
    self.reset_parameters() 可能用于初始化新加入的参数或重置现有参数，确保模型开始训练时处于良好状态。

    保存原始模型的引用：
    将原始 Sam 模型作为属性存储，使得微调后的模型可以在需要时访问原始模型的其他部分或属性。
    '''

    def __init__(self, sam_model: Sam, r: int, lora_layer=None, block_range=None):
        super(LoRA_Sam_submodel, self).__init__()
        assert r > 0

        # block_range定义了这个子模型要处理的block范围
        self.block_range = block_range if block_range is not None else (0, len(sam_model.image_encoder.blocks))
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(self.block_range[0], self.block_range[1]))
        self.w_As = []
        self.w_Bs = []

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # 如果是第一个子模型，包含patch_embed层
        if self.block_range[0] == 0:
            self.patch_embed = sam_model.image_encoder.patch_embed
        else:
            self.patch_embed = None

        if self.block_range[1] == 12:
            self.neck = sam_model.image_encoder.neck
        else:
            self.neck = None

        # 遍历图像编码器的每一层。返回索引和这个块本身
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # 如果当前层不在指定的LoRA层中，则跳过此层。
            print(t_layer_i,  t_layer_i not in self.lora_layer)
            if t_layer_i not in self.lora_layer:
                continue
            # 提取原始的QKV线性层。
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            # 创建两组线性层（w_a_linear_q/v 和 w_b_linear_q/v），用于生成低秩更新。
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            # 将这些线性层添加到列表 self.w_As 和 self.w_Bs。
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            # 用 _LoRA_qkv 类替换原始的QKV线性层，该类融合了原始的线性层和新的低秩更新。
            # 这个过程是LoRA微调的核心。对于每个指定的层，替换原始的QKV线性层，引入新的线性层用于生成低秩更新。
            # 这种“手术式”修改使得模型在保持原有架构的基础上获得了额外的适应性。
            # 更改取到的块本身里面的注意力里面的QKV，改成加了低秩矩阵的。
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.block = sam_model.image_encoder.blocks[self.block_range[0]:self.block_range[1]]

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x):
        if self.patch_embed:
            x = self.patch_embed(x)
        for blk in self.block:
            x = blk(x)
        x = x.permute(0, 3, 1, 2)
        if self.neck:
            x = self.neck(x)
        return x


if __name__ == "__main__":
    # sam = sam_model_registry["vit_b"](checkpoint="/home/yi/Documents/DELIVER/checkpoints/pretrained/sam/sam_vit_b_01ec64.pth")
    # lora_sam = LoRA_Sam(sam, 4)
    # print(lora_sam.sam.image_encoder)
    # output = lora_sam.sam.image_encoder(torch.rand(size=(2, 3, 1024, 1024)))
    # print(output.size())
    # raise  Exception
    sam = sam_model_registry["vit_b"](
        checkpoint="/home/yi/Documents/DELIVER/checkpoints/pretrained/sam/sam_vit_b_01ec64.pth")

    # 定义第一个子模型，处理patch_embed层和第0到3层的blocks
    lora_sam_1 = LoRA_Sam_submodel(sam, r=4, block_range=(0, 3))

    # 定义第二个子模型，仅处理第4到7层的blocks
    lora_sam_2 = LoRA_Sam_submodel(sam, r=4, block_range=(3, 6))
    lora_sam_3 = LoRA_Sam_submodel(sam, r=4, block_range=(6, 9))
    lora_sam_4 = LoRA_Sam_submodel(sam, r=4, block_range=(9, 12))

    # 处理输入数据
    input_tensor = torch.rand(size=(2, 3, 1024, 1024))
    output_1 = lora_sam_1(input_tensor)
    output_2 = lora_sam_2(output_1.permute(0, 2, 3, 1))
    output_3 = lora_sam_3(output_2.permute(0, 2, 3, 1))
    output_4 = lora_sam_4(output_3.permute(0, 2, 3, 1))

    print(output_1.size())
    print(output_2.size())
    print(output_3.size())
    print(output_4.size())

'''
# 以下是训练好模型之后的保存和加载模型权重进行推理的操作

# 1.训练好了之后保存模型
# 假设 lora_sam 是已经训练好的 LoRA_Sam 实例
lora_sam.save_lora_parameters('lora_sam_trained_model.pth')

# 2.加载模型用于验证
# 创建一个新的 LoRA_Sam 实例
# 假设 sam 是一个 Sam 模型的实例
# r 是 LoRA 的秩，lora_layer 是应用 LoRA 微调的层
new_lora_sam = LoRA_Sam(sam, r, lora_layer)
# 加载之前保存的模型参数
new_lora_sam.load_lora_parameters('lora_sam_trained_model.pth')
# 使用加载的模型进行预测或其他操作
# 例如，对输入数据进行前向传递
# 假设 input_data 是要处理的输入数据
output = new_lora_sam(input_data)
'''

