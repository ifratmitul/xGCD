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
    dpmm_s_applied: float = 1.0   # covariance scale used in Stage B
    dpmm_s_measured: float = 0.0  # covariance scale the data asked for (tr(within)/tr(Sigma))


def _clamp_s(s: float, lo: float = 1.0, hi: float = 10.0) -> float:
    """Clamp the covariance-scale s to [lo, hi]. s<1 would tighten (more splitting);
    unbounded s flattens all likelihoods until everything merges."""
    if s > hi:
        logger.warning(f"[dpmm cov-scale] measured s={s:.2f} hit cap {hi}")
    return float(min(max(s, lo), hi))


def _measure_cov_scale(pool: torch.Tensor, assignments: torch.Tensor, lda) -> float:
    """s = tr(pooled within-cluster scatter) / tr(Sigma), on the FINAL (post-prune)
    clusters, using EMPIRICAL cluster means (measuring spread, not doing inference).
    Naturally size-weighted (pooled estimator: big clusters dominate)."""
    C = pool.shape[1]
    S = torch.zeros(C, C, dtype=pool.dtype)
    total, n_k = 0, 0
    K = int(assignments.max()) + 1 if len(assignments) else 0
    for k in range(K):
        Xk = pool[assignments == k]
        if len(Xk) < 2:
            continue
        d = Xk - Xk.mean(0)                       # empirical cluster mean
        S += d.T @ d
        total += len(Xk); n_k += 1
    within = S / max(total - n_k, 1)              # pooled within-cluster covariance
    return float(torch.trace(within) / torch.trace(lda.cov.cpu()))


def _measure_cov_scale_gt(pool: torch.Tensor, pool_gt: torch.Tensor, k_known: int, lda) -> Optional[float]:
    """GT-based over-spread check (diagnosis only): pooled within-TRUE-novel-class scatter
    (empirical GT-class means) / tr(Sigma) — the s the data would ask for under PERFECT
    novel clustering. Settles the over-spread hypothesis with labels instead of the DPMM's
    own (possibly split/merged) assignments. None if <2 true-novel classes are in the pool."""
    novel_mask = pool_gt >= k_known
    g = pool_gt[novel_mask]
    classes = torch.unique(g).tolist()
    if len(classes) < 2:
        return None
    X = pool.detach().cpu()[novel_mask.cpu()]
    remap = {int(c): i for i, c in enumerate(classes)}
    a = torch.tensor([remap[int(c)] for c in g.tolist()])
    return _measure_cov_scale(X, a, lda)


def _log_cluster_composition(cluster_ids: torch.Tensor, gt_labels: torch.Tensor,
                             k_novel: int, k_known: int, tag: str = ""):
    """Log each DPMM cluster's ground-truth class histogram (diagnosis only).
    Labels each cluster: LEAK (a known class dominates), or novel c<k>; then flags
    splits = novel classes claimed as the dominant class by >1 cluster."""
    pre = f"[dpmm-cluster{'/' + tag if tag else ''}]"
    dominant = {}
    for j in range(k_novel):
        lbls = gt_labels[cluster_ids == j]
        if len(lbls) == 0:
            continue
        uniq, cnts = torch.unique(lbls, return_counts=True)
        order = cnts.argsort(descending=True)
        dom_c, dom_n, N = int(uniq[order[0]]), int(cnts[order[0]]), len(lbls)
        kind = f"LEAK known-c{dom_c}" if dom_c < k_known else f"novel-c{dom_c}"
        top = ", ".join(f"c{int(uniq[o])}={int(cnts[o])}" for o in order[:4])
        logger.info(f"{pre} {j}: size={N} {kind} ({100*dom_n/N:.0f}%) | {top}")
        dominant[j] = dom_c
    from collections import Counter
    dom_counts = Counter(dominant.values())
    splits = sorted(c for c, n in dom_counts.items() if n > 1 and c >= k_known)
    leaks = sorted(j for j, c in dominant.items() if c < k_known)
    logger.info(f"{pre} summary: {k_novel} clusters | "
                f"split novel classes={splits} | leak clusters (idx)={leaks}")


def _absorb_and_merge(dp, pool, lda, m0, k_known, tau, args):
    """Cluster-level cleanup of the raw DPMM output (fix #2, cluster level). Three passes:

      pass 1 absorb: a novel cluster whose CENTROID sits in the known region — nearest
             known prototype within `absorb_scale*tau` AND nearer than any novel peer —
             is a known-class leak. Drop it; its members fall back to that known class.
      pass 2 merge: novel clusters that are one class the DPMM split. COMPLETE linkage
             (every cross-pair centroid distance < `merge_scale*tau`), not single linkage,
             so a fragment can't daisy-chain two real classes through a junk bridge.
      pass 3 re-absorb: a merged group's new centroid may now land on a known prototype
             (e.g. two leak halves that each failed pass 1, or two leaks off one class that
             are nearer each other than the prototype); re-run the absorb test on group
             centroids so those get caught too.

    All tests use the shared-Sigma Mahalanobis metric; tau is the class radius the gate
    calibrated, the natural yardstick. Per-cluster distances + decisions are logged (a
    log-only pass with both flags off still prints the evidence). Returns
    (novel_means [K',C], pool_ids [N_pool]): pool_ids < k_known is an absorbed sample
    routed to that known class; >= k_known is k_known + new novel id.
    """
    dev = lda.means.device
    beta = getattr(args, "dpmm_beta", 1.0)
    do_absorb = getattr(args, "dpmm_absorb", False)
    do_merge = getattr(args, "dpmm_merge", False)
    absorb_tau = getattr(args, "absorb_scale", 1.0) * tau
    merge_tau = getattr(args, "merge_scale", 1.0) * tau

    means = dp.means.to(dev)                              # [K, C] posterior component means
    z = dp.assignments.to(dev)                            # [N_pool]
    pool = pool.to(dev)
    m0 = m0.to(dev)
    K = means.shape[0]

    def group_mask(g):
        m = torch.zeros(len(z), dtype=torch.bool, device=dev)
        for k in g:
            m |= (z == k)
        return m

    def posterior_mean(mask):                            # Eq 11 (s cancels -> cov_scale-free)
        Xk = pool[mask]
        return (m0 + beta * Xk.sum(0)) / (1.0 + beta * len(Xk))

    d_known = lda.mahalanobis_sq(means, lda.means)        # [K, k_known] centroid -> known
    nn_known_d, nn_known_j = d_known.min(1)
    d_nov = lda.mahalanobis_sq(means, means)              # [K, K] centroid -> centroid
    d_nov = d_nov + torch.eye(K, device=dev) * 1e12       # mask self-distance
    nn_nov_d, _ = d_nov.min(1)

    # ---- pass 1: absorb individual leak clusters ----
    absorbed = torch.zeros(K, dtype=torch.bool, device=dev)
    if do_absorb:
        absorbed = (nn_known_d < absorb_tau) & (nn_known_d < nn_nov_d)
    for k in range(K):
        tagk = f"ABSORB->known-c{int(nn_known_j[k])}" if bool(absorbed[k]) else "keep"
        logger.info(f"[absorb] cluster {k}: d2_nearest_known={float(nn_known_d[k]):.1f} "
                    f"(c{int(nn_known_j[k])})  d2_nearest_novel={float(nn_nov_d[k]):.1f}  "
                    f"absorb_tau={absorb_tau:.1f} -> {tagk}")

    # ---- pass 2: complete-linkage merge among survivors ----
    groups = [[k] for k in range(K) if not bool(absorbed[k])]
    if do_merge:
        changed = True
        while changed:
            changed = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    cross = [float(d_nov[a, b]) for a in groups[i] for b in groups[j]]
                    if cross and max(cross) < merge_tau:   # complete linkage
                        logger.info(f"[merge] groups {groups[i]} + {groups[j]} "
                                    f"max_d2={max(cross):.1f} < merge_tau={merge_tau:.1f}")
                        groups[i] = groups[i] + groups[j]
                        groups.pop(j)
                        changed = True
                        break
                if changed:
                    break

    # ---- pass 3: re-absorb merged-group centroids ----
    absorbed_group = {}                                   # group idx -> known class id
    if do_absorb and any(len(g) > 1 for g in groups):
        gcen = torch.stack([posterior_mean(group_mask(g)) for g in groups])
        gd_known = lda.mahalanobis_sq(gcen, lda.means)
        gnn_d, gnn_j = gd_known.min(1)
        gd_nov = lda.mahalanobis_sq(gcen, gcen) + torch.eye(len(groups), device=dev) * 1e12
        gnn_nov = gd_nov.min(1).values if len(groups) > 1 else torch.full((len(groups),), 1e12, device=dev)
        for gi, g in enumerate(groups):
            if len(g) > 1 and bool(gnn_d[gi] < absorb_tau) and bool(gnn_d[gi] < gnn_nov[gi]):
                absorbed_group[gi] = int(gnn_j[gi])
                logger.info(f"[absorb/post-merge] group {g} centroid d2_known={float(gnn_d[gi]):.1f}"
                            f"(c{int(gnn_j[gi])}) d2_novel={float(gnn_nov[gi]):.1f} -> ABSORB")

    # ---- assemble: known ids for absorbed, contiguous novel ids for survivors ----
    final_groups = [g for gi, g in enumerate(groups) if gi not in absorbed_group]
    pool_ids = torch.empty(len(z), dtype=torch.long, device=dev)
    for k in range(K):                                    # pass-1 absorbed singletons
        if bool(absorbed[k]):
            pool_ids[z == k] = int(nn_known_j[k])
    for gi, kn in absorbed_group.items():                 # pass-3 absorbed groups
        for k in groups[gi]:
            pool_ids[z == k] = kn
    for new_id, g in enumerate(final_groups):             # surviving novel groups
        for k in g:
            pool_ids[z == k] = k_known + new_id

    new_means = torch.stack([posterior_mean(group_mask(g)) for g in final_groups]) \
        if final_groups else torch.zeros(0, pool.shape[1], device=dev)

    n_pass1 = int(absorbed.sum())
    n_merged = len([k for g in groups for k in g]) - len(groups)   # clusters folded by merge
    logger.info(f"[absorb-merge] K^n {K} -> {len(final_groups)} (pass1-absorbed {n_pass1}, "
                f"merged {n_merged}, post-merge-absorbed {len(absorbed_group)})")
    return new_means, pool_ids


def _fit_dpmm(pool, lda, m0, args, dpmm_kwargs):
    """Fit Stage-B DPMM applying the covariance-mismatch scale s (fix #3).
    dpmm_cov_scale='auto' iterates fit->measure->refit (|ds|<0.1 or <=cov_iters) since the
    first-pass s is biased low; a float pins s; '1.0' = off. Always measures+logs the s the
    data asked for. Returns (dp, s_applied, s_measured)."""
    mode = str(getattr(args, "dpmm_cov_scale", "1.0"))
    max_iters = getattr(args, "dpmm_cov_iters", 3)

    def one_fit(s):
        return DPMM(**dpmm_kwargs).fit(pool, lda, m0=m0, cov_scale=s)

    if mode == "auto":
        s_applied, s_meas = 1.0, 1.0
        for it in range(max_iters):
            dp = one_fit(s_applied)
            raw = _measure_cov_scale(pool, dp.assignments, lda)   # pre-clamp
            s_meas = _clamp_s(raw)
            logger.info(f"[dpmm cov-scale] iter {it}: applied={s_applied:.2f} "
                        f"measured={s_meas:.2f} raw={raw:.3f} K^n={dp.n_components}")
            if abs(s_meas - s_applied) < 0.1:
                break
            s_applied = s_meas
        return dp, s_applied, s_meas

    s = float(mode)
    dp = one_fit(s)
    raw = _measure_cov_scale(pool, dp.assignments, lda)           # pre-clamp
    s_meas = _clamp_s(raw)
    logger.info(f"[dpmm cov-scale] applied s={s:.2f} | data asked s={s_meas:.2f} "
                f"raw={raw:.3f} | K^n={dp.n_components}")
    return dp, s, s_meas


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
    dpmm_seed = getattr(args, "dpmm_seed", 42)
    dpmm_patience = int(getattr(args, "dpmm_patience", 3))

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

    # 3. Stage B DPMM on the novel pool (with covariance-mismatch scaling, fix #3)
    k_novel = 0
    s_applied, s_measured = 1.0, 0.0
    novel_means = torch.zeros(0, lda.C, device=lda.means.device)
    if len(nov.novel_idx) >= min_novel:
        pool = unlab_logits[nov.novel_idx]
        m0 = unlab_logits.mean(0)                  # empirical-Bayes base-measure mean
        dpmm_kwargs = dict(alpha=dpmm_alpha, beta=dpmm_beta, n_sweeps=dpmm_sweeps,
                           max_points=dpmm_max_pts, min_cluster_size=dpmm_min_cluster,
                           patience=dpmm_patience, seed=dpmm_seed)
        dp, s_applied, s_measured = _fit_dpmm(pool, lda, m0, args, dpmm_kwargs)
        pool_gt = unlab_labels[nov.novel_idx].cpu() if unlab_labels is not None else None

        # GT-based over-spread check (diagnosis only): the s a PERFECT novel clustering would
        # ask for. Closes the cov-mismatch hypothesis with labels, not DPMM assignments.
        if pool_gt is not None:
            raw_gt = _measure_cov_scale_gt(pool, pool_gt, k_known, lda)
            if raw_gt is not None:
                logger.info(f"[cov-scale-gt] within-GT-novel-class scatter / tr(Sigma) = {raw_gt:.3f} "
                            f"(>1 = novel classes broader than the known-fit Sigma)")

        # raw per-cluster GT composition (diagnosis only; labels never used for decisions).
        # Tells which of the K^n clusters are clean-novel / splits (a novel class in >1
        # cluster) / leaks (a known class dominating a "novel" cluster).
        if pool_gt is not None and dp.n_components > 0:
            _log_cluster_composition(dp.assignments.cpu(), pool_gt, dp.n_components, k_known,
                                     tag="raw")

        # cluster-level cleanup: absorb known-leaks, merge DPMM splits (fix #2). ALWAYS run
        # so the [absorb]/[merge] distance tables print; the dpmm_absorb/dpmm_merge flags
        # only control whether decisions fire, so a both-off run is the log-only calibration
        # pass (identity mapping, same K^n as raw).
        if dp.n_components > 0:
            novel_means, pool_ids = _absorb_and_merge(dp, pool, lda, m0, k_known, tau, args)
            k_novel = novel_means.shape[0]
            assignments[nov.novel_idx] = pool_ids.to(assignments.device)
            if pool_gt is not None:
                # absorbed-sample audit: did the leaks route to their TRUE known class?
                absorbed_mask = pool_ids.cpu() < k_known
                n_abs = int(absorbed_mask.sum())
                if n_abs > 0:
                    correct = int((pool_ids.cpu()[absorbed_mask] == pool_gt[absorbed_mask]).sum())
                    logger.info(f"[absorb-acc] {n_abs} samples absorbed->known | "
                                f"{correct}/{n_abs} ({100*correct/n_abs:.1f}%) matched routed known class")
                if k_novel > 0:
                    _log_cluster_composition((pool_ids.cpu() - k_known), pool_gt, k_novel, k_known,
                                             tag="final")
    else:
        logger.warning(f"Novel pool too small ({len(nov.novel_idx)} < {min_novel}); K^n=0 this round")

    prototypes = torch.cat([lda.means, novel_means.to(lda.means.device)], dim=0)
    logger.info(f"E-step: K^l={k_known}, K^n={k_novel}, total prototypes={prototypes.shape[0]} "
                f"| cov-scale s applied={s_applied:.2f} measured={s_measured:.2f}")

    return EStepResult(lda=lda, prototypes=prototypes, assignments=assignments,
                       k_known=k_known, k_novel=k_novel, novel_idx=nov.novel_idx, tau=nov.tau,
                       tau_emp=tau_emp, tau_chi2=tau_chi2, lab_reject=lab_reject,
                       novel_recall=novel_recall,
                       dpmm_s_applied=s_applied, dpmm_s_measured=s_measured)
