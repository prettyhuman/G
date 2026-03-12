"""
GraphVAE.py  (Enhanced for CI-GNN-v2)
======================================
- Deeper encoder: 3-layer GCN with skip connections
- Separate projection heads for alpha / beta
- Clamped reparameterization to prevent NaN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, device):
        super().__init__()
        self.device = device
        self.conv1 = GCNConv(input_dim,  hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.bn3   = nn.BatchNorm1d(hidden_dim)
        # Skip projection
        self.skip  = nn.Linear(input_dim, hidden_dim)

        half = hidden_dim // 2
        self.alpha_mu = nn.Linear(hidden_dim, half)
        self.alpha_lv = nn.Linear(hidden_dim, half)
        self.beta_mu  = nn.Linear(hidden_dim, half)
        self.beta_lv  = nn.Linear(hidden_dim, half)

    def forward(self, x, edge_index, batch):
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))
        # Skip from input
        h  = h3 + self.skip(x)
        g  = global_add_pool(h, batch)          # [B, hidden]
        return (self.alpha_mu(g), self.alpha_lv(g),
                self.beta_mu(g),  self.beta_lv(g), h)


class GraphDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    log_var = torch.clamp(log_var, -4.0, 4.0)
    return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
