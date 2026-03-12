"""
GIN-based graph classifier (φ in the paper).
Operates on the causal subgraph G_sub produced from alpha.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


class GINNet(nn.Module):
    """
    Graph Isomorphism Network classifier.

    Args:
        input_dim  : node feature dimension
        output_dim : number of classes
        args       : argument namespace (uses args.GIN_hidden_dim, args.GIN_num_layers,
                     args.readout)
        device     : torch device
    """

    def __init__(self, input_dim: int, output_dim: int, args, device):
        super().__init__()
        self.device = device
        hidden = args.GIN_hidden_dim
        n_layers = args.GIN_num_layers
        self.readout = getattr(args, "readout", "sum")

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(n_layers):
            in_ch = input_dim if i == 0 else hidden
            self.convs.append(GINConv(_mlp(in_ch, hidden)))
            self.bns.append(nn.BatchNorm1d(hidden))

        # Final classifier head
        self.fc1 = nn.Linear(hidden * n_layers, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x, edge_index, batch):
        xs = []
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index)))
            xs.append(h)

        # Concat pooled representations from every layer
        out = torch.cat(
            [self._pool(xi, batch) for xi in xs], dim=-1
        )                                                # [B, hidden * n_layers]

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)                              # [B, num_classes]
        return out

    def _pool(self, x, batch):
        if self.readout == "sum":
            return global_add_pool(x, batch)
        elif self.readout == "mean":
            return global_mean_pool(x, batch)
        else:
            return global_max_pool(x, batch)
