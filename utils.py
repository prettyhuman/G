"""
utils.py – 数据加载、划分、评估
"""

import os
import pickle
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix

from data_generator import generate_dataset, CLASS_NAMES, CLASS_COUNTS
from graph_builder import build_dataset


# -----------------------------------------------------------------------
# 加载（带缓存）
# -----------------------------------------------------------------------

def load_dataset(cache_dir: str = "data/nickel",
                 batch_size: int = 64,
                 seed: int = 42,
                 window_size: int = 64,
                 corr_threshold: float = 0.5,
                 force_regen: bool = False):
    """
    生成/加载图数据集，返回 train/val/test DataLoader 及元信息。

    Returns:
        train_loader, val_loader, test_loader, input_dim, num_classes
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir,
        f"graphs_w{window_size}_c{corr_threshold}_s{seed}.pkl")

    if os.path.exists(cache_path) and not force_regen:
        print(f"从缓存加载数据: {cache_path}")
        with open(cache_path, "rb") as f:
            graphs = pickle.load(f)
    else:
        features_list, labels_list = generate_dataset(window_size, seed)
        print("构建图结构...")
        graphs = build_dataset(features_list, labels_list, corr_threshold)
        with open(cache_path, "wb") as f:
            pickle.dump(graphs, f)
        print(f"数据已缓存到: {cache_path}")

    # 80/10/10 分层划分（保持各类比例）
    rng = np.random.default_rng(seed)

    # 按类别分层
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, g in enumerate(graphs):
        class_indices[int(g.y.item())].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in class_indices.items():
        idxs = rng.permutation(idxs).tolist()
        n     = len(idxs)
        n_test = max(1, int(0.10 * n))
        n_val  = max(1, int(0.10 * n))
        train_idx += idxs[:n - n_test - n_val]
        val_idx   += idxs[n - n_test - n_val: n - n_test]
        test_idx  += idxs[n - n_test:]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]

    # 类别权重（处理不平衡）
    class_counts = np.array([CLASS_COUNTS[c] for c in range(6)], dtype=np.float32)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * 6   # 归一化

    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_graphs)} 个图")
    print(f"  验证集: {len(val_graphs)} 个图")
    print(f"  测试集: {len(test_graphs)} 个图")
    print(f"  节点数: {graphs[0].num_nodes}")
    print(f"  节点特征维度: {graphs[0].x.shape[1]}")
    avg_edges = np.mean([g.edge_index.shape[1] for g in graphs[:100]])
    print(f"  平均边数: {avg_edges:.1f}")

    input_dim   = graphs[0].x.shape[1]  # 8
    num_classes = 6

    train_loader = DataLoader(train_graphs, batch_size=batch_size,
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size,
                              shuffle=False, drop_last=False)

    return (train_loader, val_loader, test_loader,
            input_dim, num_classes, class_weights)


# -----------------------------------------------------------------------
# 评估 (Accuracy + Macro-F1 + MCC + 混淆矩阵)
# -----------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, verbose=False):
    """
    Returns dict: {acc, f1, mcc, confusion_matrix}
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        batch  = batch.to(device)
        logits = model.predict(batch.x, batch.edge_index, batch.batch)
        preds  = logits.argmax(dim=-1).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = 100.0 * np.mean(all_preds == all_labels)
    f1  = 100.0 * f1_score(all_labels, all_preds,
                            average="macro", zero_division=0)
    mcc = 100.0 * matthews_corrcoef(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds, labels=list(range(6)))

    if verbose:
        print(f"  Acc={acc:.2f}%  F1={f1:.2f}%  MCC={mcc:.2f}%")
        print("  混淆矩阵 (行=真实, 列=预测):")
        header = "  " + "".join(f"{CLASS_NAMES[i][:4]:>8}" for i in range(6))
        print(header)
        for i, row in enumerate(cm):
            print(f"  {CLASS_NAMES[i][:6]:<8}" +
                  "".join(f"{v:>8}" for v in row))

    return {"acc": acc, "f1": f1, "mcc": mcc, "cm": cm}
