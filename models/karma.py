"""
KARMA: Efficient Structural Defect Segmentation via Kolmogorov-Arnold Representation Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import DepthwiseSeparableConv, InceptionSepConvBlock, PatchEmbed
from models.blocks import KANBlock

class Karma(nn.Module):
    """
    KARMA: Efficient Structural Defect Segmentation via Kolmogorov-Arnold Representation Learning
    
    A hybrid architecture that combines:
    - Bottom-up path with Inception-style blocks
    - KAN integration for enhanced feature representation
    - Top-down feature pyramid for multi-scale predictions
    
    Args:
        num_classes (int): Number of segmentation classes
        no_kan (bool): If True, use normal linear layers instead of KAN
    """
    def __init__(self, num_classes=7, no_kan=False):
        super().__init__()
        # Bottom-up (same as your code, smaller expansions)
        self.conv1 = InceptionSepConvBlock(3, 16, 16, 16)   # 48
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = InceptionSepConvBlock(48, 32, 32, 32)  # 96
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = InceptionSepConvBlock(96, 64, 64, 64)  # 192
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = InceptionSepConvBlock(192, 128, 128, 128)  # 384
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = InceptionSepConvBlock(384, 192, 192, 192)  # 576

        # Patchify c5 -> KAN -> unpatchify
        self.patch_embed = PatchEmbed(
            img_size=8,
            patch_size=1,
            stride=1,
            in_chans=576,
            embed_dim=576
        )
        self.kan_block = KANBlock(dim=576, drop=0.0, drop_path=0.0, no_kan=no_kan)
        self.post_kan_conv = nn.Conv2d(576, 64, kernel_size=1, bias=False)

        # Top-down
        self.p4 = DepthwiseSeparableConv(384, 64, kernel_size=1, padding=0)
        self.p3 = DepthwiseSeparableConv(192, 64, kernel_size=1, padding=0)
        self.p2 = DepthwiseSeparableConv(96, 64, kernel_size=1, padding=0)

        # Prediction heads
        self.output_p5 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.output_p4 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.output_p3 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.output_p2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool1(c1))
        c3 = self.conv3(self.pool2(c2))
        c4 = self.conv4(self.pool3(c3))
        c5 = self.conv5(self.pool4(c4))  # (B,576,8,8)

        # Patchify + KAN
        B, C, H, W = c5.shape
        tokens, newH, newW = self.patch_embed(c5)
        tokens = self.kan_block(tokens, newH, newW)
        # unpatchify
        out_kan = tokens.reshape(B, newH, newW, C).permute(0,3,1,2).contiguous()

        # reduce channels -> 64
        p5 = self.post_kan_conv(out_kan)

        # Top-down FPN
        p4 = self.p4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.p3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.p2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')

        # Heads
        out_p5 = self.output_p5(p5)
        out_p4 = self.output_p4(p4)
        out_p3 = self.output_p3(p3)
        out_p2 = self.output_p2(p2)

        # Merge final
        out = (
            F.interpolate(out_p2, scale_factor=2, mode='nearest') +
            F.interpolate(out_p3, scale_factor=4, mode='nearest') +
            F.interpolate(out_p4, scale_factor=8, mode='nearest') +
            F.interpolate(out_p5, scale_factor=16, mode='nearest')
        )
        return self.final_conv(out)
