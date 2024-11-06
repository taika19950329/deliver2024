import torch
import torch.nn as nn
import torch.nn.functional as F


class BimodalFusion(nn.Module):
    def __init__(self, embed_dim):
        super(BimodalFusion, self).__init__()

        # 1x1 Convolutions for each modality
        self.conv_ir = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.conv_vs = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, F_ir, F_vs):
        # Input tensors F_ir, F_vs are assumed to be of shape [b, c, h, w]

        # Step 1: 1x1 convolution for each modality
        F_MIR = self.conv_ir(F_ir)  # [b, c, h, w]
        F_MVS = self.conv_vs(F_vs)  # [b, c, h, w]

        # Step 2: Compute Softmax attention weights
        Q1 = F.softmax(F_MIR, dim=1)  # Softmax along the channel dimension
        Q2 = F.softmax(F_MVS, dim=1)

        # Step 3: Element-wise multiplication by attention weights
        S_IR = Q1 * F_ir
        S_VS = Q2 * F_vs

        # Step 4: Adding weighted features to the original features
        Y1 = F_ir + S_IR
        Y2 = F_vs + S_VS

        # Step 5: Adding the two modalities
        F_M = Y1 + Y2  # Final fused feature [b, c, h, w]

        return F_M


if __name__ == '__main__':
    # Example usage:
    batch_size = 8
    height, width = 32, 32
    embed_dim = 256  # This is the channel dimension 'c'

    # Create example inputs of shape [b, c, h, w]
    F_ir = torch.rand(batch_size, embed_dim, height, width)  # Infrared modality input
    F_vs = torch.rand(batch_size, embed_dim, height, width)  # Visible spectrum modality input

    # Instantiate the fusion model
    fusion_model = BimodalFusion(embed_dim)

    # Forward pass
    fused_output = fusion_model(F_ir, F_vs)

    print(fused_output.shape)  # Output shape should be [b, c, h, w]
