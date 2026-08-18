"""M-step losses in concept-logit space (xGCD draft Eq 12-16).

All similarities use the Mahalanobis metric induced by the shared LDA covariance,
    d^2_Sigma(a, b) = (a - b)^T Sigma^{-1} (a - b),                    (Eq 12)
so representation learning shares the geometry of the E-step's Gaussian mixture.
Sigma^{-1} (`precision`) is taken from the 
E-step and held FIXED here — gradients flow
only through the concept logits (the representation), never through the metric.

  L_CL  : self-supervised contrastive over two augmented views       (Eq 13)
  L_PCL : prototypical contrastive against the fixed E-step prototypes (Eq 14)
  fidelity BCE (alpha_f * L_BCE) is the Stage-1 concept loss, reused in the M-step.
"""
import torch
import torch.nn.functional as F


def mahalanobis_sq_pairs(A: torch.Tensor, B: torch.Tensor,
                         precision: torch.Tensor = None) -> torch.Tensor:
    """Pairwise squared distances. A:[N,C], B:[M,C] -> [N,M].

    precision=None: plain squared Euclidean (use when A, B are already whitened via
    LDAGaussian.whiten -> Euclidean == Mahalanobis, the fast per-batch path).
    precision given: squared Mahalanobis under that (constant) metric.
    """
    if precision is None:
        return torch.cdist(A, B, p=2).pow(2)
    precision = precision.detach().to(A.dtype)
    diff = A.unsqueeze(1) - B.unsqueeze(0)                     # [N, M, C]
    return torch.einsum("nmc,cd,nmd->nm", diff, precision, diff).clamp_min(0.0)


def mahalanobis_contrastive_loss(z: torch.Tensor, z_prime: torch.Tensor,
                                 precision: torch.Tensor = None, temperature: float = 1.0,
                                 symmetric: bool = True, normalize_by_c: bool = True) -> torch.Tensor:
    """L_CL (Eq 13): pull the two views of an image together, push different images
    apart, under the Mahalanobis metric. z, z_prime: [B, C] logits of the two views.
    Pass whitened logits with precision=None for the fast Euclidean path.

    normalize_by_c=True divides d^2 by C: in whitened space d^2 ~ O(C), so without it the
    softmax logits are ~O(C)=O(70) and saturate to argmax -> L_CL dies (~0 gradient).
    Set False to reproduce the pre-fix behaviour (for ablations)."""
    d2 = mahalanobis_sq_pairs(z, z_prime, precision)          # [B, B]
    denom = (z.size(1) if normalize_by_c else 1.0) * temperature
    logits = -0.5 * d2 / denom
    labels = torch.arange(z.size(0), device=z.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.t(), labels))
    return loss


def prototypical_loss(z: torch.Tensor, prototypes: torch.Tensor, assignments: torch.Tensor,
                      precision: torch.Tensor = None, temperature: float = 1.0,
                      weights: torch.Tensor = None, normalize_by_c: bool = True) -> torch.Tensor:
    """L_PCL (Eq 14): minimise the Mahalanobis NLL of each logit under its assigned
    Gaussian component — a softmax over prototypes.

    z:[B,C], prototypes:[K,C] (fixed mu_k), assignments:[B] (target component id).
    weights:[B] optional per-sample weight (e.g. down-weight unlabelled-assigned-known
    samples to stop them contaminating the known prototypes). normalize_by_c: see L_CL.
    assignments==-1 are ignored (unassigned samples still contribute to L_CL).
    """
    d2 = mahalanobis_sq_pairs(z, prototypes, precision)       # [B, K]
    denom = (z.size(1) if normalize_by_c else 1.0) * temperature
    logits = -0.5 * d2 / denom
    valid = assignments != -1
    if int(valid.sum()) == 0:
        return logits.sum() * 0.0   # all ignored -> 0 loss (keeps graph connected)
    per = F.cross_entropy(logits, assignments.clamp(min=0), reduction="none")  # [B]
    mask = valid.float()
    if weights is not None:
        mask = mask * weights
    denom = mask.sum()
    if denom == 0:
        return logits.sum() * 0.0
    return (per * mask).sum() / denom


def fidelity_bce(concept_logits: torch.Tensor, targets: torch.Tensor,
                 pos_weight: torch.Tensor = None) -> torch.Tensor:
    """Concept-fidelity term alpha_f * L_BCE (Eq 16), anchoring the named concept axes.
    Applied to labelled images only (caller masks). concept_logits, targets: [B, C]."""
    return F.binary_cross_entropy_with_logits(
        concept_logits, targets,
        pos_weight=pos_weight.to(concept_logits.device) if pos_weight is not None else None,
    )
