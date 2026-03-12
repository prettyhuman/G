"""
main.py — CI-GNN 镍顶吹炉故障诊断
====================================
单独运行:
    python main.py
被 compare_main.py 调用:
    from main import train_cignn, get_cignn_args
"""

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from GraphVAE       import GraphEncoder, GraphDecoder
from GIN_classifier import GINNet
from GCE            import GenerativeCausalExplainer
from causaleffect   import CMIEstimator, hsic as hsic_fn
from utils          import load_dataset, evaluate
from explain        import (class_importance_summary,
                             plot_importance_heatmap,
                             plot_confusion_matrix)


# -----------------------------------------------------------------------
# 参数
# -----------------------------------------------------------------------

def get_cignn_args(override: dict = None):
    """
    返回 CI-GNN 默认 args。
    compare_main.py 可传入 override dict 覆盖部分字段，
    避免与 argparse 的 sys.argv 冲突。
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_root",         default="data/nickel")
    p.add_argument("--window_size",       type=int,   default=64)
    p.add_argument("--corr_threshold",    type=float, default=0.5)
    p.add_argument("--batch_size",        type=int,   default=128)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--force_regen",       action="store_true")
    p.add_argument("--GVAE_hidden_dim",   type=int,   default=128)
    p.add_argument("--stage1_epochs",     type=int,   default=150)
    p.add_argument("--stage1_lr",         type=float, default=1e-3)
    p.add_argument("--recon_threshold",   type=float, default=0.01)
    p.add_argument("--GIN_hidden_dim",    type=int,   default=128)
    p.add_argument("--GIN_num_layers",    type=int,   default=4)
    p.add_argument("--readout",           default="sum")
    p.add_argument("--dropout",           type=float, default=0.3)
    p.add_argument("--stage2_epochs",     type=int,   default=250)
    p.add_argument("--stage2_lr",         type=float, default=3e-4)
    p.add_argument("--lambda_cmi",        type=float, default=0.5)
    p.add_argument("--lambda_hsic",       type=float, default=0.15)
    p.add_argument("--use_class_weight",  action="store_true", default=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose",           type=int,   default=10)
    p.add_argument("--no_explain",        action="store_true")

    # parse_known_args：忽略 compare_main 传入的其他参数
    args, _ = p.parse_known_args()

    if override:
        for k, v in override.items():
            setattr(args, k, v)

    args.Nalpha = args.GVAE_hidden_dim // 2
    args.Nbeta  = args.GVAE_hidden_dim // 2
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# 核心训练函数（供 main() 和 compare_main.py 共同调用）
# -----------------------------------------------------------------------

def train_cignn(args, train_loader, val_loader,
                input_dim, num_classes, class_weights, device,
                curve_tracker=None):
    """
    完整训练 CI-GNN 并返回最优 gce 模型。

    Parameters
    ----------
    curve_tracker : dict | None
        若传入，则写入 {"epochs": [...], "val_acc": [...]}，
        供 compare_main.py 绘制训练曲线。

    Returns
    -------
    gce : GenerativeCausalExplainer  (已加载最优权重)
    """
    # ---- 构建模型 -------------------------------------------------------
    encoder        = GraphEncoder(input_dim, args.GVAE_hidden_dim, device).to(device)
    decoder        = GraphDecoder(args.Nalpha + args.Nbeta, input_dim).to(device)
    causal_decoder = GraphDecoder(args.Nalpha, input_dim).to(device)
    classifier     = GINNet(input_dim, num_classes, args, device).to(device)
    gce = GenerativeCausalExplainer(
        classifier, decoder, encoder, causal_decoder, device).to(device)
    cmi_estimator = CMIEstimator(args.Nalpha, args.Nbeta, num_classes).to(device)

    n_params = (sum(p.numel() for p in gce.parameters())
                + sum(p.numel() for p in cmi_estimator.parameters()))
    print(f"  [CI-GNN] Parameters: {n_params:,}")

    # ---- Stage 1: GraphVAE ----------------------------------------------
    print("\n  [CI-GNN] Stage 1: GraphVAE pre-training")
    s1  = list(encoder.parameters()) + list(decoder.parameters())
    o1  = torch.optim.Adam(s1, lr=args.stage1_lr, weight_decay=1e-5)
    s1s = torch.optim.lr_scheduler.ReduceLROnPlateau(
              o1, patience=15, factor=0.5, min_lr=1e-5)

    for epoch in range(1, args.stage1_epochs + 1):
        gce.train()
        r_sum = nb = 0.0
        for b in train_loader:
            b = b.to(device); o1.zero_grad()
            loss, r = gce.stage1_loss(b.x, b.edge_index, b.batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s1, 1.0)
            o1.step()
            r_sum += r; nb += 1
        avg_r = r_sum / nb
        s1s.step(avg_r)
        if args.verbose and epoch % args.verbose == 0:
            print(f"    S1 Epoch {epoch:4d}/{args.stage1_epochs}  recon={avg_r:.4f}")
        if avg_r < args.recon_threshold:
            print(f"    Stage1 early stop @ epoch {epoch}")
            break

    # ---- Stage 2: Causal classifier -------------------------------------
    print("\n  [CI-GNN] Stage 2: Causal classifier training")
    s2  = (list(encoder.parameters()) + list(causal_decoder.parameters())
           + list(classifier.parameters()) + list(cmi_estimator.parameters()))
    o2  = torch.optim.Adam(s2, lr=args.stage2_lr, weight_decay=1e-5)
    s2s = torch.optim.lr_scheduler.CosineAnnealingLR(
              o2, T_max=args.stage2_epochs, eta_min=1e-6)

    best_f1, best_state = 0.0, None
    ep_rec, ac_rec = [], []

    for epoch in range(1, args.stage2_epochs + 1):
        gce.train(); cmi_estimator.train()
        tot = cls_s = cmi_s = hsic_s = nb = 0.0
        for b in train_loader:
            b = b.to(device); o2.zero_grad()
            alpha_mu, _, beta_mu, _, _ = gce.encoder(b.x, b.edge_index, b.batch)
            x_sub  = gce.causal_decoder(alpha_mu[b.batch])
            logits = gce.classifier(x_sub, b.edge_index, b.batch)
            cls   = F.cross_entropy(logits, b.y,
                                    weight=class_weights.to(device)
                                    if args.use_class_weight else None)
            cmi   = cmi_estimator(alpha_mu, beta_mu, b.y)
            hsic  = hsic_fn(alpha_mu, beta_mu)
            loss  = cls + args.lambda_cmi * cmi + args.lambda_hsic * hsic
            if not torch.isfinite(loss): loss = cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s2, 1.0)
            o2.step()
            tot += loss.item(); cls_s += cls.item()
            cmi_s += cmi.item(); hsic_s += hsic.item(); nb += 1
        s2s.step()

        if args.verbose and (epoch % args.verbose == 0
                              or epoch == args.stage2_epochs):
            res = evaluate(gce, val_loader, device)
            ep_rec.append(epoch); ac_rec.append(res["acc"])
            print(f"    S2 Epoch {epoch:4d}/{args.stage2_epochs}  "
                  f"loss={tot/nb:.4f}  cls={cls_s/nb:.4f}  "
                  f"| Val Acc={res['acc']:.1f}%  "
                  f"F1={res['f1']:.1f}%  MCC={res['mcc']:.1f}%")
            if res["f1"] > best_f1:
                best_f1 = res["f1"]
                best_state = {k: v.cpu().clone()
                              for k, v in gce.state_dict().items()}

    if best_state:
        gce.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if curve_tracker is not None:
        curve_tracker["epochs"]  = ep_rec
        curve_tracker["val_acc"] = ac_rec

    return gce


# -----------------------------------------------------------------------
# 单独运行入口
# -----------------------------------------------------------------------

def main():
    args   = get_cignn_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    print("=" * 60)
    print("  CI-GNN — 镍顶吹炉故障诊断系统")
    print("=" * 60)
    print(f"设备: {device}")

    (train_loader, val_loader, test_loader,
     input_dim, num_classes, class_weights) = load_dataset(
        args.data_root, args.batch_size, args.seed,
        args.window_size, args.corr_threshold, args.force_regen)

    gce = train_cignn(args, train_loader, val_loader,
                      input_dim, num_classes, class_weights, device)

    # ---- 最终评估 -------------------------------------------------------
    print("\n" + "=" * 60)
    print("最终评估结果")
    print("=" * 60)
    for split, loader in [("训练集", train_loader),
                           ("验证集", val_loader),
                           ("测试集", test_loader)]:
        res = evaluate(gce, loader, device)
        print(f"  {split}  Acc={res['acc']:.2f}%  "
              f"F1={res['f1']:.2f}%  MCC={res['mcc']:.2f}%")

    test_res = evaluate(gce, test_loader, device, verbose=True)
    torch.save(gce.state_dict(), "ci_gnn_nickel.pt")
    print("\n模型已保存 → ci_gnn_nickel.pt")

    # ---- 可解释性 -------------------------------------------------------
    if not args.no_explain:
        print("\n" + "=" * 60)
        print("可解释性分析")
        print("=" * 60)
        results = class_importance_summary(gce, test_loader, device, top_k=5)
        plot_importance_heatmap(results, "fault_importance.png")
        plot_confusion_matrix(test_res["cm"], "confusion_matrix.png")


if __name__ == "__main__":
    main()
