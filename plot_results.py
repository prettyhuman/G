"""
plot_results.py  — 对比实验可视化 v2
  1. 分组柱状图 (Acc / F1 / MCC 各一张 + 汇总)
  2. 雷达图
  3. 混淆矩阵 (行归一化百分比热度)
  4. 训练曲线
  5. 汇总表格 (CI-GNN 高亮)
  6. 提升量图 (CI-GNN vs. baselines delta)

所有图内文字使用英文。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

METHOD_COLORS = {
    "GIN":        "#4C72B0",
    "IAGNN":      "#DD8452",
    "GATv2":      "#55A868",
    "IGCL-GNN": "#C44E52",
    "CI-GNN":     "#8172B2",
}

CLASS_NAMES_EN = [
    "Normal", "Lining\nErosion", "Lance\nBlockage",
    "Over-\nBlowing", "Mouth\nSlagging", "Cooling\nFailure",
]

SAVE_DIR = "results"


def _dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# 1. 分组柱状图（三指标汇总）
# -----------------------------------------------------------------------

def plot_metric_bars(results: dict, save_path=None):
    _dir()
    save_path = save_path or os.path.join(SAVE_DIR, "metric_comparison.png")
    methods   = list(results.keys())
    keys      = ["acc", "f1", "mcc"]
    xlabels   = ["Accuracy (%)", "Macro F1 (%)", "MCC (%)"]
    n_m       = len(methods)
    x         = np.arange(3)
    width     = 0.14
    offsets   = np.linspace(-(n_m-1)/2, (n_m-1)/2, n_m) * width

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(methods):
        vals  = [results[m][k] for k in keys]
        color = METHOD_COLORS.get(m, "#999")
        ew    = 2.0 if m == "CI-GNN" else 0.5
        ec    = "#300060" if m == "CI-GNN" else "white"
        bars  = ax.bar(x + offsets[i], vals, width, label=m,
                       color=color, alpha=0.90,
                       edgecolor=ec, linewidth=ew)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.35,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=7.5,
                    fontweight="bold" if m == "CI-GNN" else "normal",
                    color=color)

    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Performance Comparison on Nickel Converter Fault Diagnosis\n"
                 "(Dataset contains spurious correlations — tests causal robustness)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# -----------------------------------------------------------------------
# 2. 单指标柱状图
# -----------------------------------------------------------------------

def plot_single_metric(results: dict, metric_key: str,
                        metric_label: str, save_path: str):
    _dir()
    methods = list(results.keys())
    vals    = [results[m][metric_key] for m in methods]
    colors  = [METHOD_COLORS.get(m, "#999") for m in methods]
    ewidths = [2.5 if m == "CI-GNN" else 0.5 for m in methods]
    ecs     = ["#300060" if m == "CI-GNN" else "white" for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, vals, color=colors, alpha=0.88,
                  edgecolor=ecs, linewidth=ewidths, width=0.5)
    for bar, val, m in zip(bars, vals, methods):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=10,
                fontweight="bold" if m == "CI-GNN" else "normal")

    best = max(vals)
    for bar, val in zip(bars, vals):
        if abs(val - best) < 0.01:
            bar.set_hatch("//")

    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_ylim(max(0, min(vals) - 8), 108)
    ax.set_title(f"{metric_label} — Method Comparison",
                 fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# -----------------------------------------------------------------------
# 3. 雷达图
# -----------------------------------------------------------------------

def plot_radar(results: dict, save_path=None):
    _dir()
    save_path = save_path or os.path.join(SAVE_DIR, "radar_chart.png")
    cats  = ["Accuracy", "Macro F1", "MCC", "Acc (Normal)", "Avg Fault Acc"]
    N     = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for method, res in results.items():
        cm      = res["cm"]
        pc      = np.diag(cm) / (cm.sum(1) + 1e-8) * 100
        vals    = [res["acc"], res["f1"], res["mcc"], pc[0], pc[1:].mean()]
        vals   += vals[:1]
        color   = METHOD_COLORS.get(method, "#999")
        lw      = 3.0 if method == "CI-GNN" else 1.8
        alpha   = 0.15 if method == "CI-GNN" else 0.06
        ax.plot(angles, vals, "o-", lw=lw, label=method, color=color)
        ax.fill(angles, vals, alpha=alpha, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
    ax.set_title("Radar Chart — Comprehensive Performance",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# -----------------------------------------------------------------------
# 4. 混淆矩阵（行归一化百分比热度）
# -----------------------------------------------------------------------

def plot_confusion_matrix_pct(cm: np.ndarray, method_name: str,
                               save_path: str):
    _dir()
    n = cm.shape[0]
    cm_pct = cm / (cm.sum(1, keepdims=True) + 1e-8) * 100

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)", fontsize=10)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES_EN, fontsize=8)
    ax.set_yticklabels(CLASS_NAMES_EN, fontsize=8)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {method_name}\n"
                 "(Row-Normalized, %)",
                 fontsize=11, fontweight="bold")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_pct[i, j] > 55 else "black"
            ax.text(j, i, f"{cm_pct[i,j]:.1f}%\n({int(cm[i,j])})",
                    ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_all_confusion_matrices(results: dict):
    for method, res in results.items():
        path = os.path.join(SAVE_DIR, f"cm_{method.lower().replace('-','_')}.png")
        plot_confusion_matrix_pct(res["cm"], method, path)


# -----------------------------------------------------------------------
# 5. 训练曲线
# -----------------------------------------------------------------------

def plot_training_curves(history: dict, save_path=None):
    _dir()
    save_path = save_path or os.path.join(SAVE_DIR, "training_curves.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, h in history.items():
        color = METHOD_COLORS.get(method, "#999")
        lw    = 2.8 if method == "CI-GNN" else 1.8
        ls    = "-"  if method == "CI-GNN" else "--"
        ax.plot(h["epochs"], h["val_acc"],
                label=method, color=color, lw=lw, ls=ls, alpha=0.9)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Training Curves — Validation Accuracy vs. Epoch\n"
                 "(CI-GNN converges to higher accuracy despite spurious correlations)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# -----------------------------------------------------------------------
# 6. 提升量图  CI-GNN Δ vs. baselines
# -----------------------------------------------------------------------

def plot_improvement(results: dict, save_path=None):
    _dir()
    save_path = save_path or os.path.join(SAVE_DIR, "cignn_improvement.png")
    if "CI-GNN" not in results:
        return

    baselines = [m for m in results if m != "CI-GNN"]
    keys      = ["acc", "f1", "mcc"]
    xlabels   = ["Accuracy", "Macro F1", "MCC"]
    x         = np.arange(len(keys))
    width     = 0.18
    n_b       = len(baselines)
    offsets   = np.linspace(-(n_b-1)/2, (n_b-1)/2, n_b) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, base in enumerate(baselines):
        deltas = [results["CI-GNN"][k] - results[base][k] for k in keys]
        color  = METHOD_COLORS.get(base, "#999")
        bars   = ax.bar(x + offsets[i], deltas, width,
                        label=f"vs. {base}", color=color, alpha=0.85,
                        edgecolor="white", linewidth=0.5)
        for bar, d in zip(bars, deltas):
            ypos = bar.get_height() + 0.15 if d >= 0 else bar.get_height() - 0.7
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{d:+.1f}", ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylabel("CI-GNN Improvement (%)", fontsize=12)
    ax.set_title("CI-GNN Performance Gain over Baseline Methods\n"
                 "(Positive = CI-GNN outperforms; causal disentanglement advantage)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, ncol=2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# -----------------------------------------------------------------------
# 7. 汇总表格
# -----------------------------------------------------------------------

def plot_summary_table(results: dict, save_path=None):
    _dir()
    save_path = save_path or os.path.join(SAVE_DIR, "summary_table.png")
    methods = list(results.keys())
    cols    = ["Method", "Accuracy (%)", "Macro F1 (%)", "MCC (%)"]
    rows    = [[m, f"{results[m]['acc']:.2f}",
                f"{results[m]['f1']:.2f}",
                f"{results[m]['mcc']:.2f}"] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2, 1.9)

    # Best value highlight
    for ci, key in enumerate(["acc", "f1", "mcc"], 1):
        best = max(results[m][key] for m in methods)
        for ri, m in enumerate(methods, 1):
            if abs(results[m][key] - best) < 0.01:
                tbl[ri, ci].set_facecolor("#c8f0c8")
                tbl[ri, ci].set_text_props(fontweight="bold")

    # Header
    for ci in range(len(cols)):
        tbl[0, ci].set_facecolor("#2c3e50")
        tbl[0, ci].set_text_props(color="white", fontweight="bold")

    # CI-GNN row
    if "CI-GNN" in methods:
        ri = methods.index("CI-GNN") + 1
        for ci in range(len(cols)):
            tbl[ri, ci].set_facecolor("#ede0ff")
            tbl[ri, ci].set_text_props(fontweight="bold")

    ax.set_title("Performance Summary — Nickel Converter Fault Diagnosis\n"
                 "(Green = best per metric; Purple row = proposed CI-GNN)",
                 fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
