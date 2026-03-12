"""
train_utils.py
==============
统一训练/评估接口，适用于所有对比方法。
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix


# -----------------------------------------------------------------------
# 通用训练器（适用于 GIN / IAGNN / GATv2 / Graphormer）
# -----------------------------------------------------------------------

def train_standard(model, train_loader, val_loader, device,
                   epochs=200, lr=5e-4, weight_decay=1e-5,
                   class_weights=None, verbose=10,
                   model_name="Model"):
    """
    标准 Adam + CosineAnnealing 训练。
    返回在验证集上最优的模型状态字典。
    """
    opt   = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=1e-6)

    best_f1    = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = n_b = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            if class_weights is not None:
                loss = F.cross_entropy(logits, batch.y,
                                       weight=class_weights.to(device))
            else:
                loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item(); n_b += 1
        sched.step()

        if verbose and epoch % verbose == 0:
            res = evaluate_model(model, val_loader, device)
            print(f"  [{model_name}] Epoch {epoch:4d}/{epochs}  "
                  f"loss={tot_loss/n_b:.4f}  "
                  f"Val Acc={res['acc']:.1f}%  "
                  f"F1={res['f1']:.1f}%  MCC={res['mcc']:.1f}%")
            if res["f1"] > best_f1:
                best_f1    = res["f1"]
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device)
                               for k, v in best_state.items()})
    return model


# -----------------------------------------------------------------------
# CI-GNN 两阶段训练器
# -----------------------------------------------------------------------

def train_cignn(gce, cmi_estimator, train_loader, val_loader, device,
                class_weights, args):
    """封装 CI-GNN 的 Stage1 + Stage2 训练，返回训练好的 gce。"""
    from causaleffect import hsic as hsic_fn

    # Stage 1
    s1 = list(gce.encoder.parameters()) + list(gce.decoder.parameters())
    o1 = torch.optim.Adam(s1, lr=args.stage1_lr, weight_decay=1e-5)
    s1_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                   o1, patience=15, factor=0.5, min_lr=1e-5)

    print(f"\n  [CI-GNN] Stage 1: GraphVAE pre-training")
    for epoch in range(1, args.stage1_epochs + 1):
        gce.train()
        tot = recon_sum = n_b = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            o1.zero_grad()
            loss, recon = gce.stage1_loss(batch.x, batch.edge_index, batch.batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s1, 1.0)
            o1.step()
            tot += loss.item(); recon_sum += recon; n_b += 1
        avg_recon = recon_sum / n_b
        s1_sched.step(avg_recon)
        if args.verbose and epoch % args.verbose == 0:
            print(f"  [CI-GNN] S1 Epoch {epoch:4d}  recon={avg_recon:.4f}")
        if avg_recon < args.recon_threshold:
            print(f"  [CI-GNN] Stage1 early stop @ epoch {epoch}")
            break

    # Stage 2
    s2 = (list(gce.encoder.parameters())
          + list(gce.causal_decoder.parameters())
          + list(gce.classifier.parameters())
          + list(cmi_estimator.parameters()))
    o2 = torch.optim.Adam(s2, lr=args.stage2_lr, weight_decay=1e-5)
    s2_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                   o2, T_max=args.stage2_epochs, eta_min=1e-6)

    best_f1    = 0.0
    best_state = None

    print(f"\n  [CI-GNN] Stage 2: Causal classifier training")
    for epoch in range(1, args.stage2_epochs + 1):
        gce.train(); cmi_estimator.train()
        tot = n_b = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            o2.zero_grad()
            alpha_mu, _, beta_mu, _, _ = gce.encoder(
                batch.x, batch.edge_index, batch.batch)
            x_sub  = gce.causal_decoder(alpha_mu[batch.batch])
            logits = gce.classifier(x_sub, batch.edge_index, batch.batch)
            cls_loss  = F.cross_entropy(logits, batch.y,
                                        weight=class_weights.to(device))
            cmi_loss  = cmi_estimator(alpha_mu, beta_mu, batch.y)
            hsic_loss = hsic_fn(alpha_mu, beta_mu)
            loss = cls_loss + args.lambda_cmi * cmi_loss \
                 + args.lambda_hsic * hsic_loss
            if not torch.isfinite(loss):
                loss = cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s2, 1.0)
            o2.step()
            tot += loss.item(); n_b += 1
        s2_sched.step()

        if args.verbose and epoch % args.verbose == 0:
            res = evaluate_model(gce, val_loader, device)
            print(f"  [CI-GNN] S2 Epoch {epoch:4d}  loss={tot/n_b:.4f}  "
                  f"Val Acc={res['acc']:.1f}%  F1={res['f1']:.1f}%  "
                  f"MCC={res['mcc']:.1f}%")
            if res["f1"] > best_f1:
                best_f1    = res["f1"]
                best_state = {k: v.cpu().clone()
                              for k, v in gce.state_dict().items()}

    if best_state is not None:
        gce.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return gce


# -----------------------------------------------------------------------
# 统一评估
# -----------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model.predict(batch.x, batch.edge_index, batch.batch)
        preds  = logits.argmax(-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch.y.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    acc  = 100.0 * np.mean(y_true == y_pred)
    f1   = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc  = 100.0 * matthews_corrcoef(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    return {"acc": acc, "f1": f1, "mcc": mcc, "cm": cm}
