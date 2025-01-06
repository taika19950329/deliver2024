import torch
import torch.nn as nn


class MetaBEVWithModalFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts, height, width, block_size=32):
        super(MetaBEVWithModalFusion, self).__init__()
        self.height = height
        self.width = width
        self.block_size = block_size

        # Cross-Modal Attention layers
        self.cross_modal_attention_depth = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_modal_attention_lidar = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_modal_attention_event = nn.MultiheadAttention(embed_dim, num_heads)

        # Self-Attention for Meta-BEV
        self.self_attention_meta = nn.MultiheadAttention(embed_dim, num_heads)

        # MoE gating mechanism
        self.moe_gate = nn.Linear(embed_dim, num_experts)
        self.expert_layers = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)]
        )

    def forward(self, B_depth, B_lidar, B_event):
        B, C, H, W = B_depth.shape
        block_size = self.block_size

        # Step 1: Compute Meta-BEV as the mean of Depth, LiDAR, and Event
        B_meta = (B_depth + B_lidar + B_event) / 3  # [B, C, H, W]

        # Step 2: Initialize output tensor
        fused_output = torch.zeros_like(B_meta)  # [B, C, H, W]

        # Step 3: Process blocks
        H_blocks, W_blocks = H // block_size, W // block_size

        for i in range(H_blocks):
            for j in range(W_blocks):
                # Extract blocks
                block_depth = B_depth[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_lidar = B_lidar[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_event = B_event[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_meta = B_meta[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

                # Flatten and permute for attention
                block_depth_flat = block_depth.flatten(2).permute(2, 0, 1)  # [block_size^2, B, C]
                block_lidar_flat = block_lidar.flatten(2).permute(2, 0, 1)
                block_event_flat = block_event.flatten(2).permute(2, 0, 1)
                block_meta_flat = block_meta.flatten(2).permute(2, 0, 1)

                # Cross-Modal Attention
                # Query: Meta-BEV block, Key/Value: Modality blocks
                attn_depth, _ = self.cross_modal_attention_depth(block_meta_flat, block_depth_flat, block_depth_flat)
                attn_lidar, _ = self.cross_modal_attention_lidar(block_meta_flat, block_lidar_flat, block_lidar_flat)
                attn_event, _ = self.cross_modal_attention_event(block_meta_flat, block_event_flat, block_event_flat)

                # Fuse attention outputs
                fused_block_flat = attn_depth + attn_lidar + attn_event  # [block_size^2, B, C]

                # Step 4: MoE Gating
                gate_scores = torch.softmax(self.moe_gate(fused_block_flat), dim=-1)  # [block_size^2, B, num_experts]
                fused_block = sum(
                    gate_scores[..., i].unsqueeze(-1) * self.expert_layers[i](fused_block_flat)
                    for i in range(len(self.expert_layers))
                )

                # Reshape back to [B, C, block_size, block_size]
                fused_block = fused_block.permute(1, 2, 0).view(B, C, block_size, block_size)
                fused_output[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = fused_block

        # Step 5: Self-Attention on Meta-BEV
        fused_output_flat = fused_output.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        fused_output_flat, _ = self.self_attention_meta(fused_output_flat, fused_output_flat, fused_output_flat)

        # Reshape back to [B, C, H, W]
        fused_output = fused_output_flat.permute(1, 2, 0).view(B, C, self.height, self.width)

        return fused_output


# Example usage
if __name__ == "__main__":
    # Define input tensor dimensions
    B, C, H, W = 1, 64, 256, 256
    num_heads = 4
    num_experts = 4
    block_size = 32

    # Instantiate the fusion model
    fusion_model = MetaBEVWithModalFusion(embed_dim=C, num_heads=num_heads, num_experts=num_experts, height=H, width=W, block_size=block_size)

    # Generate example inputs
    B_depth = torch.randn(B, C, H, W)  # Depth modality
    B_lidar = torch.randn(B, C, H, W)  # LiDAR modality
    B_event = torch.randn(B, C, H, W)  # Event modality

    # Forward pass
    fused_output = fusion_model(B_depth, B_lidar, B_event)

    # Output shape
    print(f"Fused output shape: {fused_output.shape}")  # Expected: [1, 64, 256, 256]
