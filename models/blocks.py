"""
Neural network blocks used in the Karma architecture.
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from models.layers import DW_bn_relu

try:
    from tikan import KANLinear
except ImportError:
    print("Warning: tikan module not found. KANBlock will use normal linear layers.")

class KANLayer(nn.Module):
    """Use KANLinear from tikan, plus depthwise conv steps"""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        grid_size=3,
        spline_order=2,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1,1],
        drop=0.,
        no_kan=False
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if not no_kan:
            try:
                # Use real KANLinear
                self.fc1 = KANLinear(
                    in_features,
                    hidden_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
                self.fc2 = KANLinear(
                    hidden_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            except NameError:
                print("KANLinear not available, falling back to normal linear layers")
                self.fc1 = nn.Linear(in_features, hidden_features)
                self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            # Fall back to normal linear if needed
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # fc1
        x = self.fc1(x.reshape(B*N, C))
        x = x.reshape(B, N, -1)
        x = self.dwconv_1(x, H, W)
        # fc2
        B, N, C2 = x.shape
        x = self.fc2(x.reshape(B*N, C2))
        x = x.reshape(B, N, -1)
        x = self.dwconv_2(x, H, W)
        return x

class KANBlock(nn.Module):
    """A KAN block that normalizes + uses KANLayer + residual"""
    def __init__(self, dim, drop=0., drop_path=0., no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.layer = KANLayer(dim, dim, dim, no_kan=no_kan)

    def forward(self, x, H, W):
        shortcut = x
        x = self.norm(x)
        x = self.layer(x, H, W)
        return shortcut + self.drop_path(x)
