"""Full assembly of the parts to form the complete network"""

"""
Author: Milesile
Repo: https://github.com/milesial/Pytorch-UNet/tree/master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, shape=(64, 64)):
        super().__init__()
        self.shape = shape
        self.height_embedding = nn.Embedding(shape[0], embed_dim)
        self.width_embedding = nn.Embedding(shape[1], embed_dim)

    def forward(self, device):
        shape = self.shape
        rows = torch.arange(shape[0], device=device)
        cols = torch.arange(shape[1], device=device)
        row_embed = self.height_embedding(rows)
        col_embed = self.width_embedding(cols)
        pos_embed = row_embed.unsqueeze(1) + col_embed.unsqueeze(0)

        return pos_embed.view(shape[0] * shape[1], -1)


class AttnFF(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.l1 = nn.Linear(self.in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class TransformerBlock(nn.Module):
    """
    A self-attention transformer block applied to the skip connection feature map (x2).
    """

    def __init__(self, in_channels, num_heads=8):

        super().__init__()

        # Ensure the channel dimension is divisible by the number of heads
        if in_channels % num_heads != 0:
            raise ValueError("in_channels must be divisible by num_heads")

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.hidden_channels_ff = 4 * in_channels

        self.norm_1 = nn.LayerNorm(in_channels)
        self.norm_2 = nn.LayerNorm(in_channels)

        # The attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=False,  # Expects (Seq, Batch, Emb)
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(in_channels, self.hidden_channels_ff),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels_ff, in_channels),
        )

    def forward(self, x):
        # Store the original input to add as a residual connection later
        x_residual = x

        # Normalize, apply position embeddings, and attention
        x_seq = self.norm_1(x)

        # Apply self-attention. query, key, and value are all the same.
        attn_output, _ = self.attention(query=x_seq, key=x_seq, value=x_seq)

        # add residual, normalize, and pass to FF
        x_seq = x_residual + attn_output
        x_residual_ff = x_seq  # save for residual connection post FF
        x_seq = self.norm_2(x_seq)
        ff_output = self.feed_forward(x_seq)

        # Add residual connection
        x_out = ff_output + x_residual_ff


        return x_out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, shape=(64, 64)):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UFormer(nn.Module):
    def __init__(self, n_channels, shape) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.shape = shape
        self.encoder = PositionalEncoder(n_channels, shape = shape)
        self.projector = nn.Linear(self.n_channels, self.n_channels)
        self.t1 = TransformerBlock(n_channels)
        self.t2 = TransformerBlock(n_channels)
        self.t3 = TransformerBlock(n_channels)
        self.t4 = TransformerBlock(n_channels)
        self.t5 = TransformerBlock(n_channels)
        self.t6 = TransformerBlock(n_channels)
        self.t7 = TransformerBlock(n_channels)
        self.t8 = TransformerBlock(n_channels)
        self.t9 = TransformerBlock(n_channels)
        self.t10 = TransformerBlock(n_channels)
        self.t11 = TransformerBlock(n_channels)
        self.t12 = TransformerBlock(n_channels)

    def _patches(self, x):
        """
        Converts the feature map from the CNN bottleneck into a sequence of patches.
        Args:
            x: Input feature map with shape (B, C, H, W)
        Returns:
            A tensor of patches with shape (Seq, B, C), where Seq = H * W.
        """
        B, C, H, W = x.shape

        # Flatten the spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        patches = x.view(B, C, H * W)
        
        # Permute to get the sequence format required by the Transformer:
        # (B, C, H*W) -> (H*W, B, C)
        patches = patches.permute(2, 0, 1)
        
        return patches

    def _unpatches(self, patches):
        """
        Converts a sequence of patches back into a feature map.
        Args:
            patches: A tensor of patches with shape (Seq, B, C), where Seq = H * W.
        Returns:
            An output feature map with shape (B, C, H, W).
        """
        Seq, B, C = patches.shape
        H, W = self.shape
        # Ensure the sequence length is correct
        assert Seq == H * W, "Sequence length must match H*W for unpatching."
        
        # Permute back to (B, C, Seq)
        x = patches.permute(1, 2, 0)
        
        # Reshape to (B, C, H, W)
        x = x.view(B, C, H, W)
        return x

    def forward(self, x):
        patches = self._patches(x)
        projection = self.projector(patches)
        embeddings = self.encoder(device=x.device).unsqueeze(1) # add a batch dimension
        out = projection + embeddings
        out = self.t1(out)
        out = self.t2(out)
        out = self.t3(out)
        out = self.t4(out)
        out = self.t5(out)
        out = self.t6(out)
        out = self.t7(out)
        out = self.t8(out)
        out = self.t9(out)
        out = self.t10(out)
        out = self.t12(out)
        return self._unpatches(out)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, input_shape=(128,128)):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) 
        
        final_shape = list(np.array(input_shape, dtype=np.uint8) // 16)
        self.transu = UFormer(1024 // factor, final_shape) 
        self.up1 = Up(1024, 512 // factor, bilinear, (8, 8))
        self.up2 = Up(512, 256 // factor, bilinear, (16, 16))
        self.up3 = Up(256, 128 // factor, bilinear, (32, 32))
        self.up4 = Up(128, 64, bilinear, (64, 64))
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.transu(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
