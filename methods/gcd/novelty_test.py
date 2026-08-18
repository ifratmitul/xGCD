"""Stage A — novelty detection by hypothesis testing (xGCD draft Eq 5-9).

For an unlabelled concept logit ell_i we test, against each known cluster k,
    H0: ell_i ~ N(mu_k, Sigma)   vs   H1: ell_i not ~ N(mu_k, Sigma).
Under H0 the squared Mahalanobis distance d_k(ell_i) follows a chi-squared with C
degrees of freedom, so we accept membership at significance alpha when

    d_k(ell_i) <= tau,   tau = chi2_{C, 1 - alpha}   (Eq 8)

A sample is *known* if it is compatible with at least one known cluster (assigned to
the nearest one); otherwise it joins the novel pool U_nov for the DPMM (Stage B).

tau is not a tuned distance: it is fixed by the significance level alpha and the known
dimension C, giving the decision a calibrated statistical meaning.
"""
from dataclasses import dataclass

import torch
from loguru import logger
from scipy.stats import chi2


def chi2_threshold(C: int, alpha: float) -> float:
    """tau = (1 - alpha) quantile of chi^2 with C degrees of freedom (Eq 8)."""
    return float(chi2.ppf(1.0 - alpha, df=C))


@dataclass
class NoveltyResult:
    assignments: torch.Tensor   # [N] known cluster id in 0..K^l-1, or -1 if novel
    is_known: torch.Tensor      # [N] bool
    novel_idx: torch.Tensor     # indices (into the input) of novel-pool samples
    min_dist: torch.Tensor      # [N] min_k d_k(ell_i)
    tau: float                  # chi-squared threshold used


def empirical_tau(lab_logits: torch.Tensor, lab_labels: torch.Tensor, lda,
                  alpha: float) -> float:
    """(1-alpha) quantile of labelled OWN-class Mahalanobis distances.

    Data-driven alternative to the chi^2 threshold: because mu_k and Sigma are estimated
    (and ridge-inflated), the own-class distances do not follow chi^2_C exactly, so
    calibrating tau to the actual labelled distribution is better than assuming chi^2.
    NOTE: computed on the same labelled points used to fit mu_k/Sigma, so it is mildly
    optimistic (own-class distances are a touch smaller than held-out would be).
    """
    d = lda.mahalanobis_sq(lab_logits)                                  # [N, K]
    d_own = d.gather(1, lab_labels.view(-1, 1).long()).squeeze(1)       # [N]
    return float(torch.quantile(d_own, 1.0 - alpha))


def novelty_test(logits: torch.Tensor, lda, alpha: float = 0.05,
                 tau: float = None) -> NoveltyResult:
    """Route each unlabelled logit to a known cluster or the novel pool (Eq 9).

    logits: [N, C] unlabelled concept logits.
    lda:    a fitted LDAGaussian (provides mu_k, Sigma^{-1}, C).
    tau:    membership threshold; if None, uses the chi^2_{C,1-alpha} quantile.
    """
    C = lda.C
    if tau is None:
        tau = chi2_threshold(C, alpha)

    d = lda.mahalanobis_sq(logits)                 # [N, K^l]
    min_dist, nearest = d.min(dim=1)               # [N], [N]
    is_known = min_dist <= tau

    assignments = torch.where(is_known, nearest,
                              torch.full_like(nearest, -1))
    novel_idx = torch.nonzero(~is_known, as_tuple=False).squeeze(1)

    n = len(logits)
    logger.info(
        f"Stage A novelty test: alpha={alpha}, tau={tau:.1f} | "
        f"known={int(is_known.sum())}/{n} ({100*float(is_known.float().mean()):.1f}%), "
        f"novel_pool={len(novel_idx)}"
    )
    return NoveltyResult(assignments, is_known, novel_idx, min_dist, tau)
