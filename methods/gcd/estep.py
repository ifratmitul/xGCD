"""
xGCD E-step: known geometry -> novelty test -> DPMM (draft Alg 1, E-step).

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
from methods.gcd.novelty_test import novelty_test, empirical_tau, chi2_threshold
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
    tau_emp: float = 0.0
    tau_chi2: float = 0.0
    lab_reject: float = 0.0
    novel_recall: float = None


def run_estep(labelled_logits: torch.Tensor, labelled_labels: torch.Tensor,
              unlab_logits: torch.Tensor, args, unlab_labels: torch.Tensor = None) -> EStepResult:
    alpha = getattr(args, "novelty_alpha", 0.05)
    ridge = getattr(args, "lda_ridge_gamma", 0.1)
    tau_mode = getattr(args, "tau_mode", "empirical")
    dpmm_alpha = getattr(args, "dpmm_alpha", 1.0)
    dpmm_beta = getattr(args, "dpmm_beta", 1.0)
    dpmm_sweeps = getattr(args, "dpmm_sweeps", 30)
    dpmm_max_pts = getattr(args, "dpmm_max_points", 8000)
    dpmm_min_cluster = getattr(args, "dpmm_min_cluster_size", 10)
    min_novel = getattr(args, "dpmm_min_pool", 10)

    # 1. known-class geometry
    lda = LDAGaussian(ridge_gamma=ridge).fit(labelled_logits, labelled_labels)
    k_known = len(lda.classes)
    # Prototype indexing assumes known classes are the contiguous ids 0..K^l-1 AND that
    # lda.means rows follow that order: mstep_loss uses raw labels as prototype indices,
    # and eval uses (y_true < K^l) for the Old mask. configure_splits guarantees range(K^l);
    # this assert kills the whole class of silent mislabel bugs if a non-contiguous known
    # set (e.g. SSB splits) is ever used without remapping labels to 0..K^l-1.
    assert torch.equal(lda.classes.cpu(), torch.arange(k_known)), (
        f"Known classes must be contiguous 0..{k_known-1}, got {lda.classes.tolist()}. "
        "mstep_loss/eval assume raw labels == prototype indices — remap labels first.")

    # 2. Stage A novelty test — empirical tau (calibrated on labelled) or chi^2
    tau_emp = empirical_tau(labelled_logits, labelled_labels, lda, alpha)
    tau_chi2 = chi2_threshold(lda.C, alpha)
    tau = tau_emp if tau_mode == "empirical" else tau_chi2
    nov = novelty_test(unlab_logits, lda, alpha=alpha, tau=tau)
    assignments = nov.assignments.clone()          # known ids in 0..K^l-1, -1 for novel

    # --- gate calibration diagnostics ---
    # self-test on labelled: lab_reject ~ alpha by construction when tau=tau_emp (it IS
    # that quantile), so this is a sanity check, not an independent measure.
    lab_nov = novelty_test(labelled_logits, lda, alpha=alpha, tau=tau)
    lab_reject = 1.0 - float(lab_nov.is_known.float().mean())
    logger.info(f"[gate] tau_emp={tau_emp:.1f} tau_chi2={tau_chi2:.1f} "
                f"tau_ratio={tau_emp / tau_chi2:.2f} lab_reject={lab_reject:.3f} (target~{alpha})")

    # GT-split distances (diagnosis only; labels never used for training decisions).
    # This is the informative one: how separated are true-novel vs true-known distances.
    novel_recall = None
    if unlab_labels is not None:
        gt_novel = unlab_labels >= k_known
        for name, m in [("lab", lab_nov.min_dist),
                        ("unlab_known", nov.min_dist[~gt_novel]),
                        ("unlab_novel", nov.min_dist[gt_novel])]:
            if len(m) == 0:
                continue
            q = torch.quantile(m, torch.tensor([0.1, 0.5, 0.9], device=m.device))
            logger.info(f"[gate-dist] {name}: p10={q[0]:.1f} med={q[1]:.1f} p90={q[2]:.1f}")
        novel_recall = float((~nov.is_known)[gt_novel].float().mean()) if gt_novel.any() else 0.0
        logger.info(f"[gate] novel_recall={novel_recall:.3f} "
                    f"(fraction of true novels the gate catches)")

        # --- novel pool audit: was the pool correctly created? (diagnosis only) ---
        # separation rule: a sample is put in the pool iff min_k d_k(ell) > tau.
        if len(nov.novel_idx) > 0:
            pool_lbl = unlab_labels[nov.novel_idx]
            n_pool = len(nov.novel_idx)
            n_correct = int((pool_lbl >= k_known).sum())      # true novels -> correctly pooled
            n_wrong = n_pool - n_correct                       # known-class leaks -> wrong
            purity = n_correct / n_pool
            n_true_novel = int((unlab_labels >= k_known).sum())
            logger.info(
                f"[pool] size={n_pool} | correct(novel)={n_correct} ({100*purity:.1f}%) | "
                f"wrong(known-leak)={n_wrong} ({100*(1-purity):.1f}%) | "
                f"caught {n_correct}/{n_true_novel} true novels | rule: min_dist > tau={tau:.1f}")
            uniq, cnts = torch.unique(pool_lbl, return_counts=True)
            comp = ", ".join(
                f"c{int(c)}({'nov' if int(c) >= k_known else 'known'})={int(n)}"
                for c, n in zip(uniq.tolist(), cnts.tolist()))
            logger.info(f"[pool] class composition: {comp}")

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
                       k_known=k_known, k_novel=k_novel, novel_idx=nov.novel_idx, tau=nov.tau,
                       tau_emp=tau_emp, tau_chi2=tau_chi2, lab_reject=lab_reject,
                       novel_recall=novel_recall)
