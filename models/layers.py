"""
Specialized layer implementations for the Karma model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

class DepthwiseSeparableConv(nn.Module):
    """Depthwise + Pointwise Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionSepConvBlock(nn.Module):
    """Inception-style block that uses depthwise separable convs"""
    def __init__(self, in_channels, f1, f2, f3):
        super(InceptionSepConvBlock, self).__init__()
        # Branch1: two depthwise-separable 3x3 blocks
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, f1, kernel_size=1, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),

            nn.Conv2d(f1, f1, kernel_size=3, padding=1, groups=f1, bias=False),
            nn.Conv2d(f1, f1, kernel_size=1, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True)
        )
        # Branch2: two depthwise-separable 5x5 blocks
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),

            nn.Conv2d(f2, f2, kernel_size=5, padding=2, groups=f2, bias=False),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True)
        )
        # Branch3: MaxPool + pointwise
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, f3, kernel_size=1, bias=False),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        return torch.cat([branch1, branch2, branch3], dim=1)

class PatchEmbed(nn.Module):
    """Image/feature to Patch Embedding"""
    def __init__(
        self,
        img_size=8,
        patch_size=1,
        stride=1,
        in_chans=576,
        embed_dim=576
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        # 1x1 patch => each spatial location becomes 1 token
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
            bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C=576, 8, 8)
        B, C, H, W = x.shape
        out = self.proj(x)  # (B, embed_dim, H, W)
        _, embed_dim, H_new, W_new = out.shape
        # Flatten into tokens
        out = out.flatten(2).transpose(1, 2)  # (B, H_new*W_new, embed_dim)
        out = self.norm(out)
        return out, H_new, W_new

class DW_bn_relu(nn.Module):
    """Depthwise + BN + ReLU used inside KAN layer"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x
