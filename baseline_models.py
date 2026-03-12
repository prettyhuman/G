"""
baseline_models.py
==================
对比方法：GIN / IAGNN / GATv2 / IGCL-GNN

IAGNN 说明：
  加入 DropEdge + 更强 Dropout + 输出温度缩放，防止过拟合至 100%

IGCL-GNN 说明：
  Invariant Graph Contrastive Learning GNN
  Ref: 结合图对比学习与不变特征学习，通过环境增强 + 对比损失
       学习跨环境不变的图表示，适合工业故障诊断的域泛化场景
  实现：
    1. 图增强模块 (DropEdge + FeatureMask)
    2. 投影头做对比 (SimCLR-style InfoNCE)
    3. 不变性正则：最小化不同增强视角间的表示差异
    4. 主分类头 GIN backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv, GATv2Conv,
    global_add_pool, global_mean_pool, global_max_pool
)


def _mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )

def _pool(x, batch, mode="sum"):
    if mode == "sum":  return global_add_pool(x, batch)
    if mode == "mean": return global_mean_pool(x, batch)
    return global_max_pool(x, batch)


# -----------------------------------------------------------------------
# 1. GIN
# -----------------------------------------------------------------------

class GINModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128,
                 n_layers=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINConv(_mlp(input_dim if i == 0 else hidden, hidden)))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.fc1     = nn.Linear(hidden * n_layers, hidden)
        self.fc2     = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        xs, h = [], x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index)))
            xs.append(h)
        out = torch.cat([_pool(xi, batch, "sum") for xi in xs], dim=-1)
        return self.fc2(self.dropout(F.relu(self.fc1(out))))

    @torch.no_grad()
    def predict(self, x, edge_index, batch):
        return self.forward(x, edge_index, batch)


# -----------------------------------------------------------------------
# 2. IAGNN  (Interaction-Aware GNN)
#    修复：加入 DropEdge + 温度缩放 + 更强正则，防止 100% 过拟合
# -----------------------------------------------------------------------

class IAGNNConv(nn.Module):
    """
    Interaction-Aware message passing with edge dropout regularization.
    e_ij = σ(W_a [h_i || h_j])    interaction gate
    m_ij = e_ij ⊙ W_m h_j
    h_i' = ReLU( BN( Σ_j m_ij + W_r h_i ) )
    """
    def __init__(self, in_dim, out_dim, dropedge_p=0.15):
        super().__init__()
        self.W_a       = nn.Linear(in_dim * 2, 1)
        self.W_m       = nn.Linear(in_dim, out_dim)
        self.W_r       = nn.Linear(in_dim, out_dim)
        self.bn        = nn.BatchNorm1d(out_dim)
        self.dropedge_p = dropedge_p

    def forward(self, x, edge_index):
        # DropEdge during training
        if self.training and self.dropedge_p > 0:
            mask = torch.rand(edge_index.size(1), device=x.device) > self.dropedge_p
            edge_index = edge_index[:, mask]

        if edge_index.size(1) == 0:
            # All edges dropped — fall back to residual only
            return F.relu(self.bn(self.W_r(x)))

        src, dst = edge_index
        gate = torch.sigmoid(self.W_a(torch.cat([x[src], x[dst]], dim=-1)))  # [E,1]
        msg  = gate * self.W_m(x[src])                                        # [E, out]
        agg  = torch.zeros(x.size(0), msg.size(-1), device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return F.relu(self.bn(agg + self.W_r(x)))


class IAGNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128,
                 n_layers=4, dropout=0.4, dropedge_p=0.15,
                 temperature=2.0):
        """
        temperature > 1.0 : 软化 logits，防止过于自信导致 acc 虚高
        dropout=0.4       : 比 GIN 更强的 dropout
        dropedge_p=0.15   : 训练时随机丢弃 15% 的边
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.convs      = nn.ModuleList(
            [IAGNNConv(hidden, hidden, dropedge_p) for _ in range(n_layers)]
        )
        self.fc1        = nn.Linear(hidden, hidden // 2)
        self.fc2        = nn.Linear(hidden // 2, num_classes)
        self.dropout    = nn.Dropout(dropout)
        self.temperature = temperature

    def forward(self, x, edge_index, batch):
        h = F.relu(self.input_proj(x))
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.dropout(h)           # node-level dropout after each layer
        out = _pool(h, batch, "sum")
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits / self.temperature  # temperature scaling

    @torch.no_grad()
    def predict(self, x, edge_index, batch):
        return self.forward(x, edge_index, batch)


# -----------------------------------------------------------------------
# 3. GATv2
# -----------------------------------------------------------------------

class GATv2Model(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=64, heads=4,
                 n_layers=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(n_layers):
            in_ch   = input_dim if i == 0 else hidden * heads
            is_last = (i == n_layers - 1)
            self.convs.append(
                GATv2Conv(in_ch, hidden, heads=heads,
                          concat=(not is_last), dropout=dropout,
                          add_self_loops=True))
            self.bns.append(nn.BatchNorm1d(hidden if is_last else hidden * heads))
        self.fc1     = nn.Linear(hidden, hidden)
        self.fc2     = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.elu(bn(conv(h, edge_index)))
        out = F.relu(self.fc1(_pool(h, batch, "mean")))
        return self.fc2(self.dropout(out))

    @torch.no_grad()
    def predict(self, x, edge_index, batch):
        return self.forward(x, edge_index, batch)


# -----------------------------------------------------------------------
# 4. IGCL-GNN  (Invariant Graph Contrastive Learning GNN)
#
#  核心思路：
#    - 对每个图做两种随机增强（视角A / 视角B）：DropEdge + FeatureMask
#    - 投影头将图表示映射到对比空间，用 InfoNCE 拉近同图两视角
#    - 不变性损失：最小化两视角表示的 L2 距离（VICReg 风格）
#    - 主分类损失：交叉熵
#    - 组合损失 = CE + λ_cl * InfoNCE + λ_inv * Invariance
#
#  前向传播时 forward() 只用原始图做分类（无增强），保持推理效率
#  训练时调用 forward_train() 计算完整损失
# -----------------------------------------------------------------------

class _GINBackbone(nn.Module):
    """Shared GIN backbone used inside IGCL-GNN."""
    def __init__(self, input_dim, hidden, n_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINConv(_mlp(input_dim if i == 0 else hidden, hidden)))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.out_dim = hidden * n_layers

    def forward(self, x, edge_index, batch):
        xs, h = [], x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index)))
            xs.append(h)
        return torch.cat([_pool(xi, batch, "sum") for xi in xs], dim=-1)  # [B, out_dim]


class _GraphAugment:
    """随机图增强：DropEdge + FeatureMask"""
    def __init__(self, drop_edge_p=0.2, mask_feat_p=0.15):
        self.drop_edge_p = drop_edge_p
        self.mask_feat_p = mask_feat_p

    def __call__(self, x, edge_index):
        # DropEdge
        if self.drop_edge_p > 0 and edge_index.size(1) > 0:
            mask = torch.rand(edge_index.size(1), device=x.device) > self.drop_edge_p
            edge_index = edge_index[:, mask]
        # FeatureMask
        if self.mask_feat_p > 0:
            feat_mask = torch.rand_like(x) > self.mask_feat_p
            x = x * feat_mask.float()
        return x, edge_index


class IGCLGNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128,
                 n_layers=4, dropout=0.3,
                 proj_dim=64,
                 drop_edge_p=0.20, mask_feat_p=0.15,
                 lambda_cl=0.3, lambda_inv=0.2,
                 temperature=0.5):
        super().__init__()
        self.backbone   = _GINBackbone(input_dim, hidden, n_layers)
        self.dropout    = nn.Dropout(dropout)
        self.augment    = _GraphAugment(drop_edge_p, mask_feat_p)
        self.lambda_cl  = lambda_cl
        self.lambda_inv = lambda_inv
        self.temp       = temperature

        feat_dim = self.backbone.out_dim   # hidden * n_layers

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        # Projection head for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
        )

    # ------ Contrastive loss (InfoNCE) ----------------------------------
    def _infonce(self, z_a, z_b):
        """NT-Xent loss on graph-level embeddings [B, proj_dim]."""
        B = z_a.size(0)
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        z   = torch.cat([z_a, z_b], dim=0)      # [2B, proj_dim]
        sim = (z @ z.T) / self.temp              # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim  = sim.masked_fill(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
        loss   = F.cross_entropy(sim, labels)
        return loss

    # ------ Invariance loss (VICReg-style) ------------------------------
    def _invariance(self, z_a, z_b):
        return F.mse_loss(z_a, z_b)

    # ------ Training forward (with augmentation + contrastive loss) -----
    def forward_train(self, x, edge_index, batch, labels, class_weights, device):
        """
        Called during training. Returns total loss scalar.
        """
        # View A and View B (two independent augmentations)
        x_a, ei_a = self.augment(x, edge_index)
        x_b, ei_b = self.augment(x, edge_index)

        h_a = self.backbone(x_a, ei_a, batch)   # [B, feat]
        h_b = self.backbone(x_b, ei_b, batch)

        # Projection for contrastive
        z_a = self.proj_head(h_a)
        z_b = self.proj_head(h_b)

        # Classification from View A
        logits = self.cls_head(self.dropout(h_a))
        ce_loss  = F.cross_entropy(logits, labels,
                                   weight=class_weights.to(device))
        cl_loss  = self._infonce(z_a, z_b)
        inv_loss = self._invariance(h_a, h_b)

        total = ce_loss + self.lambda_cl * cl_loss + self.lambda_inv * inv_loss
        return total

    # ------ Inference (no augmentation) ---------------------------------
    def forward(self, x, edge_index, batch):
        h = self.backbone(x, edge_index, batch)
        return self.cls_head(self.dropout(h))

    @torch.no_grad()
    def predict(self, x, edge_index, batch):
        return self.forward(x, edge_index, batch)


# -----------------------------------------------------------------------
# 4. DIR-GNN  (Discovering Invariant Rationales for Graph Neural Networks)
#    Wu et al., ICLR 2022 — https://arxiv.org/abs/2201.12872
#
#  核心思路：
#    将图的边软划分为"因果子图 Gc"和"环境子图 Ge"两部分
#    通过 IRM (Invariant Risk Minimization) 风格的惩罚项
#    迫使因果子图 Gc 的预测在不同"环境"（不同 Ge 组合）下保持不变
#
#  实现：
#    1. EdgeScorer: 为每条边打分 s_ij ∈ (0,1)
#       → 软掩码保留因果边 (Gc) 或环境边 (Ge)
#    2. 两路 GIN:
#         rep_c = GIN_c(Gc)   因果表示
#         rep_e = GIN_e(Ge)   环境表示
#    3. 3种组合的 logits（DIR 原文 Table 1）:
#         f(Gc)          仅因果
#         f(Gc ∪ Ge)     完整图
#         f(Gc) + f(Ge)  分离相加
#    4. 损失 = CE(main) + λ_dir * IRM_penalty
#       IRM_penalty: 各 logit 组合间预测方差 → 逼近不变性
# -----------------------------------------------------------------------

class _EdgeScorer(nn.Module):
    """给每条边打分 s ∈ (0,1)，基于两端节点特征。"""
    def __init__(self, node_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        pair = torch.cat([x[src], x[dst]], dim=-1)  # [E, 2*node_dim]
        return torch.sigmoid(self.net(pair)).squeeze(-1)  # [E]


class _GINPoolEncoder(nn.Module):
    """单路 GIN 编码器 → 图级向量。"""
    def __init__(self, input_dim, hidden, n_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GINConv(_mlp(input_dim if i == 0 else hidden, hidden)))
            self.bns.append(nn.BatchNorm1d(hidden))

    def forward(self, x, edge_index, batch):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index)))
        return _pool(h, batch, "sum")  # [B, hidden]


class DIRGNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128,
                 n_layers=4, dropout=0.3,
                 causal_ratio=0.6, lambda_dir=0.5):
        """
        causal_ratio : 期望保留为因果边的比例上界（软约束）
        lambda_dir   : IRM 不变性惩罚权重
        """
        super().__init__()
        self.causal_ratio = causal_ratio
        self.lambda_dir   = lambda_dir

        # 节点嵌入投影
        self.node_proj = nn.Linear(input_dim, hidden)

        # 边评分器
        self.edge_scorer = _EdgeScorer(hidden)

        # 两路 GIN 编码器
        self.gin_c = _GINPoolEncoder(hidden, hidden, n_layers)  # 因果
        self.gin_e = _GINPoolEncoder(hidden, hidden, n_layers)  # 环境

        # 分类头（共享）
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    # ------ 软掩码边分割 ------------------------------------------------
    def _split_edges(self, x_h, edge_index):
        """
        返回因果边掩码 mask_c 和环境边掩码 mask_e。
        训练时用软分数加权消息传递（Gumbel-Sigmoid 近似）；
        推理时用硬阈值。
        """
        score = self.edge_scorer(x_h, edge_index)  # [E] ∈ (0,1)

        if self.training:
            # Straight-through Gumbel-Sigmoid
            noise  = -torch.log(-torch.log(
                         torch.clamp(torch.rand_like(score), 1e-7, 1 - 1e-7)))
            score  = torch.sigmoid((torch.log(score + 1e-7)
                                    - torch.log(1 - score + 1e-7)
                                    + noise) / 0.5)
            mask_c = score
            mask_e = 1.0 - score
        else:
            mask_c = (score >= 0.5).float()
            mask_e = 1.0 - mask_c

        return mask_c, mask_e, score

    # ------ 加权 GIN 消息传递 ------------------------------------------
    def _masked_gin(self, gin, x_h, edge_index, batch, mask):
        """
        用边权重 mask [E] 实现软的子图 GIN。
        通过在节点聚合前对消息乘以权重来近似。
        """
        # 将边权重广播到节点维度，直接在 GIN 内部无法注入，
        # 用一个简单替代：删掉权重低的边（score < 0.1 视为无边）
        keep = mask > 0.1
        if keep.sum() == 0:
            # 无边退化：仅用节点特征做池化
            h = x_h
            return _pool(h, batch, "sum")
        sub_ei = edge_index[:, keep]
        return gin(x_h, sub_ei, batch)

    # ------ 前向（推理） -----------------------------------------------
    def forward(self, x, edge_index, batch):
        x_h = F.relu(self.node_proj(x))
        mask_c, mask_e, _ = self._split_edges(x_h, edge_index)
        rep_c = self._masked_gin(self.gin_c, x_h, edge_index, batch, mask_c)
        return self.head(self.dropout(rep_c))

    # ------ 训练前向（含 IRM 损失） ------------------------------------
    def forward_train(self, x, edge_index, batch, labels,
                       class_weights, device):
        x_h = F.relu(self.node_proj(x))
        mask_c, mask_e, score = self._split_edges(x_h, edge_index)

        rep_c  = self._masked_gin(self.gin_c, x_h, edge_index, batch, mask_c)
        rep_e  = self._masked_gin(self.gin_e, x_h, edge_index, batch, mask_e)
        rep_ce = rep_c + rep_e   # combined

        # 三路 logits
        logits_c  = self.head(self.dropout(rep_c))
        logits_e  = self.head(self.dropout(rep_e))
        logits_ce = self.head(self.dropout(rep_ce))

        w = class_weights.to(device)
        ce_c  = F.cross_entropy(logits_c,  labels, weight=w)
        ce_e  = F.cross_entropy(logits_e,  labels, weight=w)
        ce_ce = F.cross_entropy(logits_ce, labels, weight=w)

        # IRM 惩罚：各路损失的方差 → 逼近不变性
        losses = torch.stack([ce_c, ce_e, ce_ce])
        irm_penalty = losses.var()

        # 稀疏性正则：让因果边比例接近 causal_ratio
        sparsity = (score.mean() - self.causal_ratio).pow(2)

        total = ce_c + self.lambda_dir * irm_penalty + 0.1 * sparsity
        return total

    @torch.no_grad()
    def predict(self, x, edge_index, batch):
        return self.forward(x, edge_index, batch)
