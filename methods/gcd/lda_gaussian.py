"""Tied full-covariance Gaussian model in concept-logit space (xGCD draft Eq 3-4, 12).

Each class (known or novel) is a Gaussian N(ell | mu_k, Sigma) with a single shared,
full covariance Sigma estimated once from the labelled known classes. A shared Sigma
(LDA, not per-component QDA) keeps every component well-posed in the high-dim concept
space even a one-sample novel cluster inherits a well-conditioned metric.

    Sigma = 1/(n - K^l) * sum_k sum_{x in D_k^l} (ell_i - mu_k)(ell_i - mu_k)^T
            + gamma * (tr(within)/C) * I                                    (Eq 4)

The ridge term floors the eigenvalues so Sigma^{-1} is well-defined; scaling by
tr(within)/C keeps gamma dimensionless (a fixed fraction of the average variance).
"""
from typing import Optional

import torch
from loguru import logger


class LDAGaussian:
    def __init__(self, ridge_gamma: float = 0.1):
        self.ridge_gamma = ridge_gamma
        self.means: Optional[torch.Tensor] = None       # [K, C] known-class means mu_k
        self.cov: Optional[torch.Tensor] = None          # [C, C] tied Sigma
        self.precision: Optional[torch.Tensor] = None    # [C, C] Sigma^{-1}
        self.logdet: Optional[torch.Tensor] = None       # scalar log|Sigma|
        self.C: Optional[int] = None
        self.classes: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------- fit
    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "LDAGaussian":
        """Estimate mu_k (per known class) and the tied Sigma from labelled logits.

        logits: [N, C] concept logits of labelled known-class images.
        labels: [N]    contiguous class ids 0..K^l-1.
        """
        logits = logits.float()
        device = logits.device
        N, C = logits.shape
        classes = torch.unique(labels).sort().values
        K = len(classes)

        means = torch.stack([logits[labels == c].mean(0) for c in classes])  # [K, C]

        # pooled within-class scatter
        S = torch.zeros(C, C, device=device, dtype=logits.dtype)
        for i, c in enumerate(classes):
            Xc = logits[labels == c] - means[i]
            S += Xc.T @ Xc
        within = S / max(N - K, 1)                       # [C, C]
        ridge = self.ridge_gamma * (torch.trace(within) / C)
        cov = within + ridge * torch.eye(C, device=device, dtype=logits.dtype)

        # symmetrize for numerical safety, then invert via Cholesky
        cov = 0.5 * (cov + cov.T)
        L = torch.linalg.cholesky(cov)                   # Sigma = L L^T
        precision = torch.cholesky_inverse(L)
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()

        self.means, self.cov, self.precision = means, cov, precision
        self.chol_L = L                                  # for whitening
        self.logdet, self.C, self.classes = logdet, C, classes
        logger.info(
            f"LDAGaussian fit: K={K} known classes, C={C}, "
            f"cond(Sigma)~{float(torch.linalg.cond(cov)):.1f}, ridge_gamma={self.ridge_gamma}"
        )
        return self

    # ---------------------------------------------------------- distances
    def mahalanobis_sq(self, X: torch.Tensor, means: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Squared Mahalanobis distance d^2_Sigma(x, mu) under the tied Sigma (Eq 6, 12).

        X:     [N, C]
        means: [K, C] (defaults to the fitted known-class means)
        returns [N, K].
        """
        means = self.means if means is None else means
        diff = X.unsqueeze(1) - means.unsqueeze(0)                 # [N, K, C]
        d2 = torch.einsum("nkc,cd,nkd->nk", diff, self.precision.to(X.dtype), diff)
        return d2.clamp_min(0.0)

    def whiten(self, X: torch.Tensor) -> torch.Tensor:
        """Map to the whitened space where Euclidean distance == Mahalanobis distance.

        With Sigma = L L^T, x~ = L^{-1} x gives ||x~_a - x~_b||^2 = d^2_Sigma(a, b).
        Differentiable in X (L is a fixed metric). Used for the per-batch M-step losses.
        """
        L = self.chol_L.detach().to(X.dtype)
        # solve L X~^T = X^T  ->  X~ = (L^{-1} X^T)^T
        return torch.linalg.solve_triangular(L, X.T, upper=False).T

    def gaussian_logprob(self, X: torch.Tensor, means: torch.Tensor,
                         cov_scale: float = 1.0) -> torch.Tensor:
        """log N(x | mean, cov_scale * Sigma) for each (x, mean) pair.

        Used by the DPMM (Stage B), where component covariances are scalar multiples of
        the shared Sigma. X:[N,C], means:[K,C] -> [N,K].
        """
        C = self.C
        maha = self.mahalanobis_sq(X, means) / cov_scale          # [N, K]
        logdet = self.logdet + C * torch.log(torch.tensor(float(cov_scale)))
        return -0.5 * (C * torch.log(torch.tensor(2.0 * torch.pi)) + logdet + maha)

    def to(self, device) -> "LDAGaussian":
        for a in ("means", "cov", "precision", "chol_L", "logdet", "classes"):
            v = getattr(self, a)
            if torch.is_tensor(v):
                setattr(self, a, v.to(device))
        return self
