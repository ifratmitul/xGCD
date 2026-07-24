"""The xGCD E-step: known geometry -> novelty test -> DPMM (draft Alg 1, E-step).

Combines the three pieces into a single call that produces, for the current
representation, the prototype set and cluster assignments the M-step trains against:

  1. Fit the tied-Sigma LDA on labelled known-class logits  -> mu_k, Sigma       (Eq 3-4)
  2. Stage A: chi^2 novelty test on unlabelled logits        -> known / novel pool (Eq 5-9)
  3. Stage B: DPMM collapsed Gibbs on the novel pool         -> K^n, novel means  (Eq 10-11)

Prototypes = [known mu_k (0..K^l-1)] ++ [novel means (K^l..K^l+K^n-1)].
Assignments for every unlabelled sample: its known cluster id, or K^l + its novel id.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger

from methods.gcd.lda_gaussian import LDAGaussian
from methods.gcd.novelty_test import novelty_test
from methods.gcd.dpmm import DPMM


@dataclass
class EStepResult:
    lda: LDAGaussian            # fitted model (Sigma^{-1} = the M-step metric)
    prototypes: torch.Tensor    # [K^l + K^n, C]  known means then novel means
    assignments: torch.Tensor   # [N_unlab] cluster id in 0..K^l+K^n-1
    k_known: int
    k_novel: int
    novel_idx: torch.Tensor     # indices into the unlabelled set that were deemed novel
    tau: float


def run_estep(labelled_logits: torch.Tensor, labelled_labels: torch.Tensor,
              unlab_logits: torch.Tensor, args) -> EStepResult:
    alpha = getattr(args, "novelty_alpha", 0.05)
    ridge = getattr(args, "lda_ridge_gamma", 0.1)
    dpmm_alpha = getattr(args, "dpmm_alpha", 1.0)
    dpmm_beta = getattr(args, "dpmm_beta", 1.0)
    dpmm_sweeps = getattr(args, "dpmm_sweeps", 30)
    dpmm_max_pts = getattr(args, "dpmm_max_points", 8000)
    dpmm_min_cluster = getattr(args, "dpmm_min_cluster_size", 10)
    min_novel = getattr(args, "dpmm_min_pool", 10)

    # 1. known-class geometry
    lda = LDAGaussian(ridge_gamma=ridge).fit(labelled_logits, labelled_labels)
    k_known = len(lda.classes)

    # 2. Stage A novelty test
    nov = novelty_test(unlab_logits, lda, alpha=alpha)
    assignments = nov.assignments.clone()          # known ids in 0..K^l-1, -1 for novel

    # 3. Stage B DPMM on the novel pool
    k_novel = 0
    novel_means = torch.zeros(0, lda.C, device=lda.means.device)
    if len(nov.novel_idx) >= min_novel:
        pool = unlab_logits[nov.novel_idx]
        m0 = unlab_logits.mean(0)                  # empirical-Bayes base-measure mean
        dp = DPMM(alpha=dpmm_alpha, beta=dpmm_beta, n_sweeps=dpmm_sweeps,
                  max_points=dpmm_max_pts, min_cluster_size=dpmm_min_cluster).fit(pool, lda, m0=m0)
        k_novel = dp.n_components
        novel_means = dp.means.to(assignments.device if assignments.is_floating_point()
                                  else lda.means.device)
        # novel samples get ids K^l + their DPMM cluster id
        assignments[nov.novel_idx] = k_known + dp.assignments.to(assignments.device)
    else:
        logger.warning(f"Novel pool too small ({len(nov.novel_idx)} < {min_novel}); K^n=0 this round")

    prototypes = torch.cat([lda.means, novel_means.to(lda.means.device)], dim=0)
    logger.info(f"E-step: K^l={k_known}, K^n={k_novel}, total prototypes={prototypes.shape[0]}")

    return EStepResult(lda=lda, prototypes=prototypes, assignments=assignments,
                       k_known=k_known, k_novel=k_novel, novel_idx=nov.novel_idx, tau=nov.tau)
