"""
TiKAN: Tiny Kolmogorov-Arnold Network module.

This module implements KANLinear, a parameter-efficient version of KAN through 
low-rank factorization based on the Kolmogorov-Arnold representation theorem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.u = nn.Parameter(torch.Tensor(in_features, rank))
        self.v = nn.Parameter(torch.Tensor(rank, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.u)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.v.t() @ self.u.t(), self.bias)


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=3, spline_order=1,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1],
                 share_spline_weights=True, groups=4, rank=10, factor_rank=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        self.base_activation = base_activation()
        self.share_spline_weights = share_spline_weights
        
        self.base_weight = LowRankLinear(in_features, out_features, rank)
        
        spline_features = grid_size + spline_order
        if share_spline_weights:
            self.spline_weight_u = nn.Parameter(torch.Tensor(out_features, factor_rank))
            self.spline_weight_v = nn.Parameter(torch.Tensor(factor_rank, spline_features))
        else:
            self.spline_weight_u = nn.Parameter(torch.Tensor(out_features, in_features, factor_rank))
            self.spline_weight_v = nn.Parameter(torch.Tensor(factor_rank, spline_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.base_weight.reset_parameters()
        nn.init.kaiming_uniform_(self.spline_weight_u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight_v, a=math.sqrt(5))

    def forward(self, x):
        # Apply base linear transformation with activation
        base_output = self.base_activation(self.base_weight(x)) * self.scale_base
        
        # Generate spline input
        spline_input = torch.linspace(self.grid_range[0], self.grid_range[1], 
                                    self.grid_size + self.spline_order, device=x.device)
        
        # Add small noise for regularization
        if self.training and self.scale_noise > 0:
            noise = torch.randn_like(spline_input) * self.scale_noise
            spline_input = spline_input + noise
        
        # Apply spline transformation
        if self.share_spline_weights:
            spline_weight = self.spline_weight_u @ self.spline_weight_v
            spline_output = F.linear(spline_input.repeat(x.size(0), 1), spline_weight)
        else:
            spline_weight = torch.bmm(self.spline_weight_u, 
                                    self.spline_weight_v.unsqueeze(0).expand(self.out_features, -1, -1))
            spline_output = torch.sum(F.linear(spline_input.repeat(x.size(0), self.in_features, 1),
                                             spline_weight.view(-1, self.grid_size + self.spline_order)) 
                                    * x.unsqueeze(-1), dim=1)
        
        # Combine base and spline outputs
        return base_output + spline_output * self.scale_spline

    def prune(self, threshold=1e-3):
        with torch.no_grad():
            self.base_weight.u.data[abs(self.base_weight.u.data) < threshold] = 0
            self.base_weight.v.data[abs(self.base_weight.v.data) < threshold] = 0
            self.spline_weight_u.data[abs(self.spline_weight_u.data) < threshold] = 0
            self.spline_weight_v.data[abs(self.spline_weight_v.data) < threshold] = 0


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
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
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        loss = 0
        for layer in self.layers:
            if hasattr(layer, 'regularization_loss'):
                loss += layer.regularization_loss(regularize_activation, regularize_entropy)
        return loss
