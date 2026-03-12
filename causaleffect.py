"""
Causal Effect Estimator.

CMI surrogate  : I(alpha; Y | beta)  — stable, non-adversarial formulation
HSIC           : independence penalty alpha ⊥ beta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CMI estimator  — stable version
# ---------------------------------------------------------------------------

class CMIEstimator(nn.Module):
    """
    Stable CMI surrogate.

    Original formulation  ce_joint - ce_beta  is adversarial and causes NaN.

    Replacement: train a *joint* head to predict Y from (alpha, beta) and a
    *alpha-only* head to predict Y from alpha alone.  Loss = ce_joint + ce_alpha.
    This encourages alpha to carry class-relevant information without the
    unstable adversarial subtraction.
    """

    def __init__(self, alpha_dim: int, beta_dim: int, num_classes: int):
        super().__init__()
        self.joint_head = nn.Sequential(
            nn.Linear(alpha_dim + beta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(alpha_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        joint_logits = self.joint_head(torch.cat([alpha, beta], dim=-1))
        alpha_logits = self.alpha_head(alpha)

        ce_joint = F.cross_entropy(joint_logits, labels)
        ce_alpha = F.cross_entropy(alpha_logits, labels)

        # Both terms minimised: alpha must predict Y, and (alpha,beta) must too.
        # This is a stable lower bound on I(alpha;Y|beta).
        return ce_joint + ce_alpha


# ---------------------------------------------------------------------------
# HSIC independence criterion  alpha ⊥ beta  — numerically stable version
# ---------------------------------------------------------------------------

def rbf_kernel(X: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian RBF kernel with numerically stable distance computation."""
    # X: [N, D]
    # L2-normalise rows so distances are bounded in [0, 2]
    X_norm = F.normalize(X, p=2, dim=-1)
    diff   = X_norm.unsqueeze(1) - X_norm.unsqueeze(0)          # [N, N, D]
    sq     = diff.pow(2).sum(-1).clamp(min=0.0)                  # [N, N]
    return torch.exp(-sq / (2.0 * sigma ** 2))


def hsic(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Biased HSIC estimator.  Returns 0 if batch is too small.
    Inputs are L2-normalised inside rbf_kernel to prevent overflow.
    """
    n = X.size(0)
    if n < 4:
        return X.new_zeros(1).squeeze()

    Kx = rbf_kernel(X, sigma)
    Ky = rbf_kernel(Y, sigma)

    H  = torch.eye(n, device=X.device) - (1.0 / n)
    Kxc = H @ Kx @ H
    Kyc = H @ Ky @ H

    hsic_val = (Kxc * Kyc).sum() / (n - 1) ** 2
    return hsic_val.clamp(min=0.0)
