"""Full assembly of the parts to form the complete network"""

"""
Author: Milesile
Repo: https://github.com/milesial/Pytorch-UNet/tree/master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, shape=(64, 64)):
        super().__init__()
        self.height_embedding = nn.Embedding(shape[0], embed_dim)
        self.width_embedding = nn.Embedding(shape[1], embed_dim)

    def forward(self, shape, device):
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

    def __init__(self, in_channels, num_heads=8, shape=(64, 64)):
        super().__init__()
        # Ensure the channel dimension is divisible by the number of heads
        if in_channels % num_heads != 0:
            raise ValueError("in_channels must be divisible by num_heads")

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.hidden_channels_ff = 4 * in_channels
        self.pos_encoder = PositionalEncoding(in_channels, shape)

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
        # Input x has shape (B, C, H, W)
        B, C, H, W = x.shape

        # Reshape for attention: (B, C, H, W) -> (B, C, H*W) -> (H*W, B, C)
        # H*W is the sequence length, B is the batch size, C is the embedding dimension
        x_seq = x.view(B, C, H * W).permute(2, 0, 1)  # (Seq, Batch, Emb)

        # Store the original input to add as a residual connection later
        x_residual = x_seq

        # Normalize, apply position embeddings, and attention
        x_seq = self.norm_1(x_seq)
        pos_embedding = self.pos_encoder((H, W), x.device)
        x_seq = x_seq + pos_embedding.unsqueeze(1)

        # Apply self-attention. query, key, and value are all the same.
        attn_output, _ = self.attention(query=x_seq, key=x_seq, value=x_seq)

        # add residual, normalize, and pass to FF
        x_seq = x_residual + attn_output
        x_residual_ff = x_seq  # save for residual connection post FF
        x_seq = self.norm_2(x_seq)
        ff_output = self.feed_forward(x_seq)

        # Add residual connection
        x_seq = ff_output + x_residual_ff

        # Reshape back to image format: (H*W, B, C) -> (B, C, H*W) -> (B, C, H, W)
        x_out = x_seq.permute(1, 2, 0).view(B, C, H, W)

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

        self.attention = TransformerBlock(in_channels=out_channels, num_heads=8, shape=shape)
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
        x2_attn = self.attention(x2)
        x = torch.cat([x2_attn, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
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
        self.up1 = Up(1024, 512 // factor, bilinear, (8, 8))
        self.up2 = Up(512, 256 // factor, bilinear, (16, 16))
        self.up3 = Up(256, 128 // factor, bilinear, (32, 32))
        self.up4 = Up(128, 64, bilinear, (64, 64))
        self.outc = OutConv(64, n_classes)
        self.attention = TransformerBlock(in_channels=1024 // factor, shape=(4, 4))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
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
