"""
compare_main.py — 对比实验
==========================
方法: GIN / IAGNN / GATv2 / DIR-GNN / CI-GNN
CI-GNN 训练直接调用 main.train_cignn()，不重复实现。

运行: python compare_main.py
"""

import argparse, os, random
import numpy as np
import torch
import torch.nn.functional as F

# CI-GNN 从 main.py 导入，复用全部逻辑
from main            import train_cignn, get_cignn_args, set_seed
from utils           import load_dataset
from train_utils     import evaluate_model
from baseline_models import GINModel, IAGNNModel, GATv2Model, DIRGNNModel
from plot_results    import (
    plot_metric_bars, plot_single_metric, plot_radar,
    plot_all_confusion_matrices, plot_training_curves,
    plot_summary_table, plot_improvement, SAVE_DIR,
)


# -----------------------------------------------------------------------
# 参数（只含对比实验自己的参数，CI-GNN 参数由 get_cignn_args 负责）
# -----------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default="data/nickel_v2")
    p.add_argument("--window_size",    type=int,   default=64)
    p.add_argument("--corr_threshold", type=float, default=0.5)
    p.add_argument("--batch_size",     type=int,   default=128)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--force_regen",    action="store_true")
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--hidden",         type=int,   default=128)
    p.add_argument("--n_layers",       type=int,   default=4)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--verbose",        type=int,   default=20)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# -----------------------------------------------------------------------
# 通用训练追踪器（baselines 用）
# -----------------------------------------------------------------------

class CurveTracker:
    def __init__(self): self.data = {}

    def track(self, model, train_loader, val_loader, device,
              epochs, lr, wd, class_weights, verbose, name,
              use_forward_train=False):
        """
        use_forward_train=True : 调用 model.forward_train()（DIR-GNN）
        use_forward_train=False: 调用 model(x, ei, batch) 标准前向
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        best_f1, best_st = 0.0, None
        ep_rec, ac_rec   = [], []

        for epoch in range(1, epochs + 1):
            model.train()
            for b in train_loader:
                b = b.to(device); opt.zero_grad()
                if use_forward_train:
                    loss = model.forward_train(
                        b.x, b.edge_index, b.batch,
                        b.y, class_weights, device)
                else:
                    loss = F.cross_entropy(
                        model(b.x, b.edge_index, b.batch),
                        b.y, weight=class_weights.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            if epoch % verbose == 0 or epoch == epochs:
                res = evaluate_model(model, val_loader, device)
                ep_rec.append(epoch); ac_rec.append(res["acc"])
                print(f"  [{name}] {epoch:4d}/{epochs}  "
                      f"Acc={res['acc']:.1f}%  "
                      f"F1={res['f1']:.1f}%  MCC={res['mcc']:.1f}%")
                if res["f1"] > best_f1:
                    best_f1 = res["f1"]
                    best_st = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}

        if best_st:
            model.load_state_dict(
                {k: v.to(device) for k, v in best_st.items()})
        self.data[name] = {"epochs": ep_rec, "val_acc": ac_rec}
        return model


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    args   = get_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 65)
    print("  Nickel Converter Fault Diagnosis — Comparison Experiment")
    print("  Methods: GIN / IAGNN / GATv2 / DIR-GNN / CI-GNN (proposed)")
    print("=" * 65)
    print(f"Device: {device}\n")

    # ---- 数据 ----------------------------------------------------------
    (train_loader, val_loader, test_loader,
     input_dim, num_classes, class_weights) = load_dataset(
        args.data_root, args.batch_size, args.seed,
        args.window_size, args.corr_threshold, args.force_regen)

    tracker = CurveTracker()
    results = {}
    SEP = "=" * 65

    # ---- GIN -----------------------------------------------------------
    print(SEP); print("Training: GIN"); print(SEP)
    gin = GINModel(input_dim, num_classes,
                   args.hidden, args.n_layers, args.dropout).to(device)
    gin = tracker.track(gin, train_loader, val_loader, device,
                        args.epochs, args.lr, 1e-5,
                        class_weights, args.verbose, "GIN")
    results["GIN"] = evaluate_model(gin, test_loader, device)
    print(f"  GIN  → Acc={results['GIN']['acc']:.2f}%  "
          f"F1={results['GIN']['f1']:.2f}%  MCC={results['GIN']['mcc']:.2f}%\n")

    # ---- IAGNN ---------------------------------------------------------
    print(SEP); print("Training: IAGNN"); print(SEP)
    iagnn = IAGNNModel(input_dim, num_classes, args.hidden, args.n_layers,
                       dropout=0.4, dropedge_p=0.15,
                       temperature=2.0).to(device)
    iagnn = tracker.track(iagnn, train_loader, val_loader, device,
                           args.epochs, args.lr, 2e-4,
                           class_weights, args.verbose, "IAGNN")
    results["IAGNN"] = evaluate_model(iagnn, test_loader, device)
    print(f"  IAGNN  → Acc={results['IAGNN']['acc']:.2f}%  "
          f"F1={results['IAGNN']['f1']:.2f}%  MCC={results['IAGNN']['mcc']:.2f}%\n")

    # ---- GATv2 ---------------------------------------------------------
    print(SEP); print("Training: GATv2"); print(SEP)
    gatv2 = GATv2Model(input_dim, num_classes, hidden=64, heads=4,
                        n_layers=args.n_layers,
                        dropout=args.dropout).to(device)
    gatv2 = tracker.track(gatv2, train_loader, val_loader, device,
                           args.epochs, args.lr, 1e-5,
                           class_weights, args.verbose, "GATv2")
    results["GATv2"] = evaluate_model(gatv2, test_loader, device)
    print(f"  GATv2  → Acc={results['GATv2']['acc']:.2f}%  "
          f"F1={results['GATv2']['f1']:.2f}%  MCC={results['GATv2']['mcc']:.2f}%\n")

    # ---- DIR-GNN -------------------------------------------------------
    print(SEP); print("Training: DIR-GNN"); print(SEP)
    dirgnn = DIRGNNModel(input_dim, num_classes,
                          hidden=args.hidden, n_layers=args.n_layers,
                          dropout=args.dropout,
                          causal_ratio=0.6, lambda_dir=0.5).to(device)
    dirgnn = tracker.track(dirgnn, train_loader, val_loader, device,
                            args.epochs, args.lr, 1e-5,
                            class_weights, args.verbose, "DIR-GNN",
                            use_forward_train=True)
    results["DIR-GNN"] = evaluate_model(dirgnn, test_loader, device)
    print(f"  DIR-GNN  → Acc={results['DIR-GNN']['acc']:.2f}%  "
          f"F1={results['DIR-GNN']['f1']:.2f}%  MCC={results['DIR-GNN']['mcc']:.2f}%\n")

    # ---- CI-GNN（直接调用 main.train_cignn，无重复代码）---------------
    print(SEP); print("Training: CI-GNN (Proposed) — via main.train_cignn()"); print(SEP)

    # 用对比实验的数据目录和 seed 覆盖 CI-GNN 默认参数
    cignn_args = get_cignn_args(override={
        "data_root":       args.data_root,
        "seed":            args.seed,
        "batch_size":      args.batch_size,
        "window_size":     args.window_size,
        "corr_threshold":  args.corr_threshold,
        "verbose":         args.verbose,
        "device":          str(device),
    })

    ci_curve = {}   # train_cignn 会写入 epochs/val_acc
    gce = train_cignn(
        cignn_args,
        train_loader, val_loader,
        input_dim, num_classes, class_weights,
        device,
        curve_tracker=ci_curve,
    )
    tracker.data["CI-GNN"] = ci_curve
    results["CI-GNN"] = evaluate_model(gce, test_loader, device)
    print(f"  CI-GNN  → Acc={results['CI-GNN']['acc']:.2f}%  "
          f"F1={results['CI-GNN']['f1']:.2f}%  MCC={results['CI-GNN']['mcc']:.2f}%\n")

    # ---- 汇总 ----------------------------------------------------------
    print(SEP); print("FINAL RESULTS (Test Set)"); print(SEP)
    print(f"  {'Method':<14} {'Accuracy':>10} {'Macro F1':>10} {'MCC':>10}")
    print(f"  {'-'*50}")
    for m, r in results.items():
        tag = "  ← proposed" if m == "CI-GNN" else ""
        print(f"  {m:<14} {r['acc']:>9.2f}%  "
              f"{r['f1']:>9.2f}%  {r['mcc']:>9.2f}%{tag}")

    # ---- 绘图 ----------------------------------------------------------
    print(f"\nGenerating plots → {SAVE_DIR}/")
    plot_metric_bars(results)
    plot_single_metric(results, "acc", "Accuracy (%)",
                        os.path.join(SAVE_DIR, "accuracy_bar.png"))
    plot_single_metric(results, "f1",  "Macro F1 (%)",
                        os.path.join(SAVE_DIR, "f1_bar.png"))
    plot_single_metric(results, "mcc", "MCC (%)",
                        os.path.join(SAVE_DIR, "mcc_bar.png"))
    plot_radar(results)
    plot_all_confusion_matrices(results)
    plot_training_curves(tracker.data)
    plot_summary_table(results)
    plot_improvement(results)
    print(f"\nAll plots saved to: {os.path.abspath(SAVE_DIR)}/")


if __name__ == "__main__":
    main()
