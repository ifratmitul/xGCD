"""Phase-3 linear classifier head over concept logits (xGCD draft, parametric refinement).

A single Linear(C_concepts -> K_total) on top of the frozen-discovery concept logits ell.
Initialised from the prototypes via the LDA-classifier identity so that, at init,
argmax_k head(ell) == nearest-prototype(ell) under the shared-Sigma Mahalanobis metric:

    minimise (ell - mu_k)^T Sigma^-1 (ell - mu_k)
    == maximise  (Sigma^-1 mu_k) . ell  -  1/2 mu_k^T Sigma^-1 mu_k

so a linear head with
    W[k] = Sigma^-1 mu_k          b[k] = -1/2 mu_k^T Sigma^-1 mu_k
reproduces the K=... nearest-prototype assignment that scored All=0.87 — epoch 0 is then a
result in its own right ("gate + init, no training"), and CE refines from there.
"""
import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(num_concepts, num_classes)

    def forward(self, ell: torch.Tensor) -> torch.Tensor:
        return self.fc(ell)                      # [N, K] class logits

    @torch.no_grad()
    def init_from_prototypes(self, prototypes: torch.Tensor, precision: torch.Tensor):
        """Set W = Sigma^-1 mu, b = -1/2 mu^T Sigma^-1 mu (rows = classes).

        prototypes: [K, C] known means then graduated novel means.
        precision:  [C, C] = Sigma^-1 (the fitted LDA precision).
        """
        P = precision.to(prototypes.dtype).to(prototypes.device)
        W = prototypes @ P                       # [K, C]  (row k = Sigma^-1 mu_k, P symmetric)
        b = -0.5 * (W * prototypes).sum(dim=1)   # [K]     (-1/2 mu_k^T Sigma^-1 mu_k)
        self.fc.weight.copy_(W)
        self.fc.bias.copy_(b)
