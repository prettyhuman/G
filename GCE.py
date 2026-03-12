"""
GenerativeCausalExplainer (GCE)

Ties together:
  1. GraphVAE  – encode G → Z=[alpha;beta], decode Z → Ĝ
  2. Causal subgraph generator – decode alpha → G_sub node features
  3. GIN classifier φ – classify G_sub → Y
  4. CMI + HSIC regularisers

Two-stage training (as in the paper):
  Stage 1 – Train GraphVAE (reconstruction + KL)
  Stage 2 – Train causal subgraph generator + classifier + CMI + HSIC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from GraphVAE import reparameterize
from causaleffect import CMIEstimator, hsic


class GenerativeCausalExplainer(nn.Module):

    def __init__(self, classifier, decoder, encoder, causal_decoder, device):
        super().__init__()
        self.classifier     = classifier
        self.decoder        = decoder
        self.encoder        = encoder
        self.causal_decoder = causal_decoder
        self.device         = device

    # ------------------------------------------------------------------
    # Stage 1: GraphVAE pre-training
    # ------------------------------------------------------------------

    def stage1_loss(self, x, edge_index, batch):
        alpha_mu, alpha_lv, beta_mu, beta_lv, _ = \
            self.encoder(x, edge_index, batch)

        alpha  = reparameterize(alpha_mu, alpha_lv)
        beta   = reparameterize(beta_mu,  beta_lv)
        z      = torch.cat([alpha, beta], dim=-1)       # [B, Nalpha+Nbeta]
        z_node = z[batch]                               # [N, Nalpha+Nbeta]
        x_hat  = self.decoder(z_node)                   # [N, input_dim]

        recon_loss = F.mse_loss(x_hat, x)

        # KL with clamped log_var (already clamped inside reparameterize)
        alpha_lv_c = torch.clamp(alpha_lv, -4.0, 4.0)
        beta_lv_c  = torch.clamp(beta_lv,  -4.0, 4.0)
        kl = -0.5 * torch.mean(
            1 + alpha_lv_c - alpha_mu.pow(2) - alpha_lv_c.exp()
        ) + -0.5 * torch.mean(
            1 + beta_lv_c  - beta_mu.pow(2)  - beta_lv_c.exp()
        )

        return recon_loss + kl, recon_loss.item()

    # ------------------------------------------------------------------
    # Stage 2: Causal subgraph + classifier training
    # ------------------------------------------------------------------

    def stage2_loss(self, x, edge_index, batch, labels,
                    cmi_estimator: CMIEstimator,
                    lambda_cmi: float = 0.3,
                    lambda_hsic: float = 0.1):
        alpha_mu, alpha_lv, beta_mu, beta_lv, _ = \
            self.encoder(x, edge_index, batch)

        # Use mean for stability (no sampling noise during stage 2)
        alpha = alpha_mu                                # [B, Nalpha]
        beta  = beta_mu                                 # [B, Nbeta]

        # Causal subgraph node features from alpha
        alpha_node = alpha[batch]                       # [N, Nalpha]
        x_sub      = self.causal_decoder(alpha_node)    # [N, input_dim]

        # Classify
        logits   = self.classifier(x_sub, edge_index, batch)
        cls_loss = F.cross_entropy(logits, labels)

        # CMI regulariser
        cmi_loss = cmi_estimator(alpha, beta, labels)

        # HSIC independence
        hsic_loss = hsic(alpha, beta)

        total_loss = cls_loss + lambda_cmi * cmi_loss + lambda_hsic * hsic_loss

        # Guard against NaN (should not happen after fixes, but keep as safety)
        if not torch.isfinite(total_loss):
            total_loss = cls_loss   # fall back to pure classification loss

        return total_loss, logits, cls_loss.item(), cmi_loss.item(), hsic_loss.item()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x, edge_index, batch):
        alpha_mu, _, _, _, _ = self.encoder(x, edge_index, batch)
        alpha_node = alpha_mu[batch]
        x_sub      = self.causal_decoder(alpha_node)
        logits     = self.classifier(x_sub, edge_index, batch)
        return logits
