"""
explain.py
==========
CI-GNN 可解释性模块 — 镍顶吹炉故障诊断

节点重要性计算: 遮蔽法 (Occlusion-based, 无需梯度)
  对每个节点 i，将其特征置零后重新预测，
  重要性 = 原始预测置信度 - 遮蔽后置信度
  值越高说明该传感器越关键。

完全不依赖 .backward()，不受 no_grad 上下文影响。
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator import VARIABLE_NAMES, CLASS_NAMES


# -----------------------------------------------------------------------
# 遮蔽法节点重要性
# -----------------------------------------------------------------------

@torch.no_grad()
def compute_node_importance(gce, batch, device):
    """
    遮蔽法: 逐一将每个节点特征置零，观察预测置信度下降幅度。

    Returns:
        importance : [B, N_VARS]  numpy array (越大越重要)
        preds      : [B]          predicted labels (numpy)
    """
    gce.eval()

    x          = batch.x.to(device)          # [N_total, feat_dim]
    edge_index = batch.edge_index.to(device)
    b          = batch.batch.to(device)
    n_graphs   = int(b.max().item()) + 1
    n_vars     = 20

    # 原始预测置信度
    logits_orig = gce.predict(x, edge_index, b)          # [B, num_classes]
    probs_orig  = F.softmax(logits_orig, dim=-1)          # [B, num_classes]
    preds       = logits_orig.argmax(dim=-1).cpu().numpy()

    # 每张图预测类的置信度
    base_conf = probs_orig[torch.arange(n_graphs), preds].cpu().numpy()  # [B]

    importance = np.zeros((n_graphs, n_vars), dtype=np.float32)

    # 按图+节点逐一遮蔽
    for node_idx in range(n_vars):
        x_masked = x.clone()
        # 找出所有图中第 node_idx 个节点（每张图节点数固定=20）
        global_node_ids = []
        for g in range(n_graphs):
            g_mask  = (b == g).nonzero(as_tuple=True)[0]
            if len(g_mask) > node_idx:
                global_node_ids.append(g_mask[node_idx].item())

        if len(global_node_ids) == 0:
            continue

        # 置零
        x_masked[global_node_ids] = 0.0

        logits_masked = gce.predict(x_masked, edge_index, b)
        probs_masked  = F.softmax(logits_masked, dim=-1)
        masked_conf   = probs_masked[torch.arange(n_graphs), preds].cpu().numpy()

        drop = base_conf - masked_conf    # [B]  正值=该节点重要
        for g in range(n_graphs):
            if g < len(global_node_ids):
                importance[g, node_idx] = max(float(drop[g]), 0.0)

    return importance, preds


# -----------------------------------------------------------------------
# 每类故障平均重要性汇总
# -----------------------------------------------------------------------

def class_importance_summary(gce, loader, device, top_k=5):
    """
    对整个数据集，按真实类别聚合平均节点重要性。
    打印每类故障最重要的 top_k 个传感器。
    """
    print("\n" + "=" * 60)
    print("CI-GNN 可解释性分析 — 各故障关键传感器 (遮蔽法)")
    print("=" * 60)

    class_imp   = {c: [] for c in range(6)}
    class_corr  = {c: 0  for c in range(6)}
    class_total = {c: 0  for c in range(6)}

    for batch in loader:
        labels = batch.y.cpu().numpy()
        imp, preds = compute_node_importance(gce, batch, device)

        for i, (label, pred) in enumerate(zip(labels, preds)):
            label = int(label)
            if i < imp.shape[0]:
                class_imp[label].append(imp[i])
            if pred == label:
                class_corr[label] += 1
            class_total[label] += 1

    results = {}
    for c in range(6):
        if len(class_imp[c]) == 0:
            continue

        avg_imp = np.mean(class_imp[c], axis=0)   # [N_VARS]
        top_idx = np.argsort(avg_imp)[::-1][:top_k]
        acc_c   = 100.0 * class_corr[c] / max(class_total[c], 1)

        print(f"\n  类别 {c}: {CLASS_NAMES[c]}  (类内准确率: {acc_c:.1f}%)")
        print(f"  {'排名':<4} {'传感器':<22} {'重要性(置信度下降)':>18}")
        print(f"  {'-'*46}")
        for rank, idx in enumerate(top_idx):
            print(f"  {rank+1:<4} {VARIABLE_NAMES[idx]:<22} {avg_imp[idx]:>18.4f}")

        results[c] = {"avg_importance": avg_imp, "top_idx": top_idx}

    return results


# -----------------------------------------------------------------------
# 可视化: 重要性热力图
# -----------------------------------------------------------------------

def plot_importance_heatmap(results: dict,
                             save_path: str = "fault_importance.png"):
    n_classes = 6
    n_vars    = 20

    matrix = np.zeros((n_classes, n_vars), dtype=np.float32)
    for c, res in results.items():
        imp = res["avg_importance"]
        rng = imp.max() - imp.min() + 1e-8
        matrix[c] = (imp - imp.min()) / rng

    fig, ax = plt.subplots(figsize=(18, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(VARIABLE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([CLASS_NAMES[c] for c in range(n_classes)], fontsize=9)

    for i in range(n_classes):
        for j in range(n_vars):
            val   = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="归一化重要性")
    ax.set_title("CI-GNN 因果子图分析 — 各故障关键传感器热力图\n"
                 "(镍顶吹炉故障诊断, 遮蔽法)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n热力图已保存: {save_path}")


# -----------------------------------------------------------------------
# 可视化: 混淆矩阵
# -----------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray,
                           save_path: str = "confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")

    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [CLASS_NAMES[i] for i in range(n)]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("预测类别", fontsize=10)
    ax.set_ylabel("真实类别", fontsize=10)
    ax.set_title("混淆矩阵 — 镍顶吹炉故障诊断 (CI-GNN)", fontsize=11)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm / (row_sums + 1e-8)
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")