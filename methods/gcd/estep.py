"""The xGCD E-step: known geometry -> novelty test -> DPMM (draft Alg 1, E-step).

Combines the three pieces into a single call that produces, for the current
representation, the prototype set and cluster assignments the M-step trains against:

  1. Fit the tied-Sigma LDA on labelled known-class logits  -> mu_k, Sigma       (Eq 3-4)
  2. Stage A: chi^2 novelty test on unlabelled logits        -> known / novel pool (Eq 5-9)
  3. Stage B: DPMM collapsed Gibbs on the novel pool         -> K^n, novel means  (Eq 10-11)

Prototypes = [known mu_k (0..K^l-1)] ++ [novel means (K^l..K^l+K^n-1)].
Assignments for every unlabelled sample: its known cluster id, or K^l + its novel id.
"""
import copy
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


def _check_chi2_normalization(labelled_logits: torch.Tensor, labelled_labels: torch.Tensor,
                              lda: LDAGaussian, alpha: float, tau_chi2: float) -> None:
    """Sanity-check the assumption behind the chi^2 novelty test.

    The test assumes a labelled sample's squared Mahalanobis distance to its OWN class
    mean is ~ chi^2_C. If the concepts are truly whitened by Sigma, then:
      * mean/median own-distance ~ C,
      * the empirical (1-alpha) quantile ~ tau_chi2   (ratio ~ 1.0),
      * the chi^2 bar rejects ~alpha of knowns.
    Large deviations mean tau=chi2(C,alpha) is mis-calibrated (concepts not chi^2-
    normalised, or Sigma over-/under-regularised) and an empirical tau is safer.

    Uses the labelled logits already computed for the LDA fit, so it is nearly free.
    """
    with torch.no_grad():
        d = lda.mahalanobis_sq(labelled_logits)                          # [N, K^l]
        classes = lda.classes.to(labelled_labels.device)
        own_col = (labelled_labels.view(-1, 1) == classes.view(1, -1)).int().argmax(1)
        own = d[torch.arange(d.shape[0], device=d.device), own_col.to(d.device)]  # [N]

        C = lda.C
        mean_d, med_d = float(own.mean()), float(own.median())
        tau_emp = float(torch.quantile(own, 1.0 - alpha))
        ratio = tau_emp / tau_chi2 if tau_chi2 > 0 else float("nan")
        chi2_reject = float((own > tau_chi2).float().mean())

        logger.info(
            f"[norm-check] mean own-d^2={mean_d:.1f} med={med_d:.1f} (want ~C={C}) | "
            f"tau_emp(q{1 - alpha:.2f})={tau_emp:.1f} vs tau_chi2={tau_chi2:.1f} ratio={ratio:.2f} "
            f"(want ~1.0) | chi^2 bar rejects {100 * chi2_reject:.1f}% of knowns "
            f"(target {100 * alpha:.0f}%)"
        )
        if not (0.7 * C <= mean_d <= 1.3 * C):
            logger.warning(
                f"[norm-check] mean own-d^2={mean_d:.1f} is far from C={C}: concepts are NOT "
                f"chi^2-normalised -> chi^2 tau is mis-calibrated. Prefer an empirical tau, or "
                f"retune lda_ridge_gamma (high ridge shrinks distances below C)."
            )
        elif not (0.85 <= ratio <= 1.15):
            side = "too high -> misses novels" if ratio < 1 else "too low -> leaks knowns"
            logger.warning(
                f"[norm-check] tau_emp/tau_chi2={ratio:.2f} (far from 1.0): chi^2 bar is {side}. "
                f"Consider an empirical tau."
            )


def run_estep(labelled_logits: torch.Tensor, labelled_labels: torch.Tensor,
              unlab_logits: torch.Tensor, args,
              prev_assign_unlab: Optional[torch.Tensor] = None,
              frozen_lda: Optional["LDAGaussian"] = None) -> EStepResult:
    # prev_assign_unlab: optional [N_unlab] GLOBAL cluster id each unlab sample had last
    # E-step (known id < K^l, or K^l+novel_id). Used to warm-start the DPMM. None -> cold.
    # frozen_lda: optional pre-fit LDA snapshot. If given, the known means + metric Sigma
    # are FROZEN (reused, not refit) so only the novel means change. None -> refit as usual.
    alpha = getattr(args, "novelty_alpha", 0.05)
    ridge = getattr(args, "lda_ridge_gamma", 0.1)
    dpmm_alpha = getattr(args, "dpmm_alpha", 1.0)
    dpmm_beta = getattr(args, "dpmm_beta", 1.0)
    dpmm_sweeps = getattr(args, "dpmm_sweeps", 30)
    dpmm_max_pts = getattr(args, "dpmm_max_points", 8000)
    dpmm_min_cluster = getattr(args, "dpmm_min_cluster_size", 10)
    min_novel = getattr(args, "dpmm_min_pool", 10)

    # 1. known-class geometry (frozen snapshot -> known protos + metric fixed; else refit)
    if frozen_lda is not None:
        lda = copy.deepcopy(frozen_lda)     # copy so downstream .to(device) can't mutate the snapshot
    else:
        lda = LDAGaussian(ridge_gamma=ridge).fit(labelled_logits, labelled_labels)
    k_known = len(lda.classes)

    # 2. Stage A novelty test
    nov = novelty_test(unlab_logits, lda, alpha=alpha)
    assignments = nov.assignments.clone()          # known ids in 0..K^l-1, -1 for novel

    # 2b. is chi^2 tau trustworthy? (checks concepts are chi^2-normalised; nearly free)
    _check_chi2_normalization(labelled_logits, labelled_labels, lda, alpha, nov.tau)
    # print('_'*80)
    # print('assignments',assignments)
    # print('lda.C,', lda.C)
    # print('_'*80)

    # 3. Stage B DPMM on the novel pool
    k_novel = 0
    novel_means = torch.zeros(0, lda.C, device=lda.means.device)
    if len(nov.novel_idx) >= min_novel:
        pool = unlab_logits[nov.novel_idx]
        m0 = unlab_logits.mean(0)                  # empirical-Bayes base-measure mean

        # warm-start: map each pool point's PREVIOUS novel cluster to a local id
        # (>=K^l -> local = global-K^l; known/unseen -> -1 = no prior cluster).
        init_z = None
        if prev_assign_unlab is not None:
            prev_pool = prev_assign_unlab[nov.novel_idx].cpu()
            init_z = torch.where(prev_pool >= k_known, prev_pool - k_known,
                                 torch.full_like(prev_pool, -1))

        dp = DPMM(alpha=dpmm_alpha, beta=dpmm_beta, n_sweeps=dpmm_sweeps,
                  max_points=dpmm_max_pts, min_cluster_size=dpmm_min_cluster).fit(
                      pool, lda, m0=m0, init_z=init_z)
        k_novel = dp.n_components
        novel_means = dp.means.to(assignments.device if assignments.is_floating_point()
                                  else lda.means.device)
        # novel samples get ids K^l + their DPMM cluster id
        assignments[nov.novel_idx] = k_known + dp.assignments.to(assignments.device)

        sizes = dp.counts.tolist()
        logger.info(f"novel clusters (global ids {k_known}..{k_known + k_novel - 1}) "
                    f"sizes={sizes}")
    else:
        logger.warning(f"Novel pool too small ({len(nov.novel_idx)} < {min_novel}); K^n=0 this round")

    prototypes = torch.cat([lda.means, novel_means.to(lda.means.device)], dim=0)
    logger.info(f"E-step: K^l={k_known}, K^n={k_novel}, total prototypes={prototypes.shape[0]}")

    return EStepResult(lda=lda, prototypes=prototypes, assignments=assignments,
                       k_known=k_known, k_novel=k_novel, novel_idx=nov.novel_idx, tau=nov.tau)
