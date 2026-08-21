"""Stage B — counting novel classes with a DPMM (xGCD draft Eq 10-11, Appendix A).

The novel pool U_nov (samples Stage A found incompatible with every known class) is
clustered by a Dirichlet-process mixture of Gaussians with a fixed, shared covariance
Sigma (the LDA covariance). Component means are integrated out, so inference is a
collapsed Gibbs sampler over the assignments; the number of occupied components is the
estimate K^n.

Generative model (App A.1):   G ~ DP(alpha, G0),  G0 = N(m0, beta*Sigma),
                              mu_z ~ G,  ell | z ~ N(mu_z, Sigma).

Collapsed assignment conditional (Eq 10):
    p(z_i = k | z_-i, ell) ∝ { N_-i,k * N(ell_i | mu~_k, Sigma~_k)   existing k
                             { alpha  * N(ell_i | m0, (1+beta)Sigma) new component

with the Normal-Normal posterior (Eq 11, Sigma0 = beta*Sigma):
    mu~_k    = (m0 + beta * N_k * ellbar_k) / (1 + beta * N_k)
    Sigma~_k = (1 + beta / (1 + beta * N_k)) * Sigma
and the new-component predictive N(ell_i | m0, (1+beta)Sigma) from marginalising mu.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger


@dataclass
class DPMMResult:
    assignments: torch.Tensor   # [N] novel-cluster id in 0..K^n-1
    means: torch.Tensor         # [K^n, C] posterior component means mu~_k
    counts: torch.Tensor        # [K^n] occupancy N_k
    n_components: int           # K^n


class DPMM:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, n_sweeps: int = 30,
                 max_points: Optional[int] = 8000, patience: int = 3,
                 min_cluster_size: int = 10, seed: int = 0):
        self.alpha = alpha
        self.beta = beta
        self.n_sweeps = n_sweeps
        self.max_points = max_points      # subsample the pool above this (speed)
        self.patience = patience          # stop if #clusters stable this many sweeps
        self.min_cluster_size = min_cluster_size  # dissolve clusters below this (spurious)
        self.seed = seed

    # ------------------------------------------------------------------ fit
    @torch.no_grad()
    def fit(self, logits_pool: torch.Tensor, lda, m0: Optional[torch.Tensor] = None,
            cov_scale: float = 1.0) -> DPMMResult:
        # cov_scale (s) replaces Sigma -> s*Sigma throughout Stage B by multiplying every
        # scalar covariance factor by s (component, base measure, new-table). s scales the
        # prior AND likelihood, so it cancels in the Normal-Normal mean -> means unchanged.
        # The C*log(scale) terms pick up the log-det correction automatically (no extra term).
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        X_full = logits_pool.float().cpu()
        N_full, C = X_full.shape

        # optional subsample for the sweep (K^n estimate is stable under subsampling)
        if self.max_points and N_full > self.max_points:
            idx = torch.randperm(N_full, generator=g)[: self.max_points]
            X = X_full[idx]
        else:
            idx = torch.arange(N_full)
            X = X_full
        N = len(X)

        lda_cpu = _lda_on_cpu(lda)
        m0 = X.mean(0) if m0 is None else m0.float().cpu()

        log_alpha = math.log(self.alpha)
        const = C * math.log(2.0 * math.pi) + float(lda_cpu.logdet)
        new_scale = cov_scale * (1.0 + self.beta)   # new-table predictive: (1+beta)*s*Sigma
        # new-component predictive log N(x | m0, (1+beta)Sigma)
        maha_new = lda_cpu.mahalanobis_sq(X, m0.unsqueeze(0)).squeeze(1)  # [N]
        logp_new = -0.5 * (const + C * math.log(new_scale) + maha_new / new_scale)

        # cluster sufficient stats: counts N_k and sums S_k (sum of members)
        counts = []           # list[int]
        sums = []             # list[Tensor C]
        z = torch.full((N,), -1, dtype=torch.long)

        def cluster_means_and_scales():
            Nk = torch.tensor([float(c) for c in counts])            # [K]
            S = torch.stack(sums) if sums else torch.zeros(0, C)     # [K, C]
            denom = (1.0 + self.beta * Nk).unsqueeze(1)              # [K,1]
            mu = (m0.unsqueeze(0) + self.beta * S) / denom           # [K, C] (Eq 11)
            s = cov_scale * (1.0 + self.beta / (1.0 + self.beta * Nk))  # [K] existing-comp scale
            return mu, s, Nk

        def add(i, k):
            counts[k] += 1
            sums[k] += X[i]
            z[i] = k

        def remove(i):
            k = int(z[i])
            counts[k] -= 1
            sums[k] -= X[i]
            z[i] = -1
            return k

        def gibbs_step(i):
            x = X[i]
            if counts:
                mu, s, Nk = cluster_means_and_scales()
                maha = lda_cpu.mahalanobis_sq(x.unsqueeze(0), mu).squeeze(0)   # [K]
                logp = -0.5 * (const + C * torch.log(s) + maha / s)           # [K]
                logw = torch.log(Nk) + logp                                    # existing
                logw = torch.cat([logw, torch.tensor([log_alpha + float(logp_new[i])])])
            else:
                logw = torch.tensor([log_alpha + float(logp_new[i])])
            w = torch.softmax(logw - logw.max(), dim=0)
            choice = int(torch.multinomial(w, 1, generator=g))
            if choice == len(counts):     # new cluster
                counts.append(0)
                sums.append(torch.zeros(C))
            add(i, choice)

        # ---- sequential CRP init (one pass from empty) ----
        order = torch.randperm(N, generator=g)
        for i in order.tolist():
            gibbs_step(i)

        # ---- Gibbs sweeps to convergence ----
        stable = 0
        prev_k = _num_nonempty(counts)
        for sweep in range(self.n_sweeps):
            for i in torch.randperm(N, generator=g).tolist():
                remove(i)
                _drop_empty_inplace(counts, sums, z)
                gibbs_step(i)
            k_now = _num_nonempty(counts)
            stable = stable + 1 if k_now == prev_k else 0
            prev_k = k_now
            if (sweep + 1) % 5 == 0 or sweep == self.n_sweeps - 1:
                logger.info(f"DPMM sweep {sweep+1}/{self.n_sweeps}: K^n={k_now}")
            if stable >= self.patience:
                logger.info(f"DPMM converged: K^n={k_now} stable for {self.patience} sweeps")
                break

        # relabel to contiguous ids and build final means over the (subsampled) pool
        z, counts_t, means = _finalize(z, X, counts, sums, self.beta, m0)

        # prune spurious sub-threshold clusters (dissolve, reassign to nearest surviving).
        # fix #2: floor min size at 1% of the pool (kills micro-fragments 10 lets through).
        min_size = max(self.min_cluster_size, int(0.01 * N))
        if min_size > 1 and means.shape[0] > 1:
            k_before = means.shape[0]
            z, means, counts_t = _prune_small(
                z, X, means, self.beta, m0, lda_cpu, min_size)
            if means.shape[0] != k_before:
                logger.info(f"DPMM pruned {k_before}->{means.shape[0]} clusters "
                            f"(min_size={min_size})")

        # if we subsampled, assign the held-out pool points to nearest final mean
        if N < N_full:
            z_full = _assign_full(X_full, means, lda_cpu)
            counts_t = torch.bincount(z_full, minlength=means.shape[0])
            result_z = z_full
        else:
            result_z = z

        logger.info(f"DPMM done: K^n={means.shape[0]} over {N_full} novel samples")
        return DPMMResult(result_z, means, counts_t, means.shape[0])


# --------------------------------------------------------------------- helpers
def _num_nonempty(counts):
    return sum(1 for c in counts if c > 0)


def _drop_empty_inplace(counts, sums, z):
    """Remove empty clusters and shift ids in z accordingly."""
    empties = [k for k, c in enumerate(counts) if c == 0]
    if not empties:
        return
    for k in reversed(empties):
        counts.pop(k)
        sums.pop(k)
        z[z > k] -= 1  # shift ids above the removed cluster


def _finalize(z, X, counts, sums, beta, m0):
    keep = [k for k, c in enumerate(counts) if c > 0]
    remap = {k: i for i, k in enumerate(keep)}
    z2 = z.clone()
    for old, new in remap.items():
        z2[z == old] = new
    Nk = torch.tensor([float(counts[k]) for k in keep])
    S = torch.stack([sums[k] for k in keep])
    denom = (1.0 + beta * Nk).unsqueeze(1)
    means = (m0.unsqueeze(0) + beta * S) / denom
    return z2, Nk.long(), means


def _posterior_means(z, X, K, beta, m0):
    C = X.shape[1]
    means = []
    for k in range(K):
        Xk = X[z == k]
        Nk = len(Xk)
        S = Xk.sum(0) if Nk > 0 else torch.zeros(C)
        means.append((m0 + beta * S) / (1.0 + beta * Nk))
    return torch.stack(means)


def _prune_small(z, X, means, beta, m0, lda_cpu, min_size):
    """Iteratively dissolve clusters below `min_size`, reassigning members to the
    nearest surviving cluster (Mahalanobis), until all clusters are large enough."""
    while means.shape[0] > 1:
        counts = torch.bincount(z, minlength=means.shape[0])
        if int(counts.min()) >= min_size:
            break
        keep = counts >= min_size
        if int(keep.sum()) == 0:            # everything tiny: keep the single largest
            keep = torch.zeros_like(counts, dtype=torch.bool)
            keep[int(counts.argmax())] = True
        means = means[keep]
        z = lda_cpu.mahalanobis_sq(X, means).argmin(1)
        means = _posterior_means(z, X, means.shape[0], beta, m0)
    return z, means, torch.bincount(z, minlength=means.shape[0])


def _assign_full(X_full, means, lda_cpu):
    d = lda_cpu.mahalanobis_sq(X_full, means)   # [N_full, K]
    return d.argmin(1)


class _LDAView:
    """Lightweight CPU view of a fitted LDAGaussian for the sampler."""
    def __init__(self, precision, logdet, C):
        self.precision, self.logdet, self.C = precision, logdet, C

    def mahalanobis_sq(self, X, means):
        diff = X.unsqueeze(1) - means.unsqueeze(0)
        return torch.einsum("nkc,cd,nkd->nk", diff, self.precision, diff).clamp_min(0.0)


def _lda_on_cpu(lda):
    return _LDAView(lda.precision.float().cpu(),
                    float(lda.logdet), int(lda.C))
