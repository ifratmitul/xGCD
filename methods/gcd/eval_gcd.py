"""GCD evaluation for xGCD (draft Alg 1 inference).

Assign each unlabelled sample to its max-posterior component (nearest prototype under
the shared Mahalanobis metric — equal-weight, tied-Sigma components make nearest-mean
the MAP assignment), then score with the standard GCD All/Old/New protocol.

Because K^n is *estimated*, the cluster count generally differs from the ground-truth
class count; split_cluster_acc_v2 builds a square-padded contingency and Hungarian-
matches it, so over-/under-counting is penalised honestly. We also report the class-
number error |K^n_hat - K^n|, since count estimation is the headline contribution.
"""
from typing import List, Tuple

import torch

from project_utils.cluster_and_log_utils import split_cluster_acc_v2

def split_cluster_acc_frozen_known(y_true, preds, k_known: int, total_classes: int):
    """All/Old/New accuracy with the KNOWN prototypes FIXED to their classes.

    Known prototype k (id < k_known) always denotes known class k, so the matching can
    never relabel it as a novel class. Only the novel prototypes (id >= k_known) are
    Hungarian-matched to the novel classes. This stops a large novel cluster sitting near
    a known prototype from 'stealing' that known class in the score (the Old-drop artifact).
    """
    y_true = np.asarray(y_true).astype(int)
    preds = np.asarray(preds).astype(int)
    pred_class = preds.copy()                      # known prototypes already == their class id

    novel_ids = sorted(set(preds[preds >= k_known].tolist()))
    novel_classes = list(range(k_known, total_classes))
    if novel_ids and novel_classes:
        C = np.zeros((len(novel_ids), len(novel_classes)), dtype=np.int64)
        for i, c in enumerate(novel_ids):
            m = preds == c
            for j, k in enumerate(novel_classes):
                C[i, j] = int(np.sum(m & (y_true == k)))
        r_ind, c_ind = linear_sum_assignment(-C)   # maximise overlap; matches min(#clusters,#classes)
        matched = {novel_ids[r]: novel_classes[c] for r, c in zip(r_ind, c_ind)}
        for c in novel_ids:
            pred_class[preds == c] = matched.get(c, -1)   # unmatched novel cluster -> counted wrong

    correct = pred_class == y_true
    mask_old = y_true < k_known
    all_acc = float(correct.mean())
    old_acc = float(correct[mask_old].mean()) if mask_old.any() else 0.0
    new_acc = float(correct[~mask_old].mean()) if (~mask_old).any() else 0.0
    return all_acc, old_acc, new_acc

@torch.no_grad()
def assign_to_prototypes(logits: torch.Tensor, prototypes: torch.Tensor, lda) -> torch.Tensor:
    """Nearest prototype under the Mahalanobis metric (whitened -> Euclidean). -> [N]."""
    zt = lda.whiten(logits.to(prototypes.device))
    pt = lda.whiten(prototypes)
    return torch.cdist(zt, pt).argmin(dim=1)


@torch.no_grad()
def evaluate_gcd(logits: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor,
                 lda, num_labeled_classes: int, total_classes: int,
                 frozen_known: bool = False) -> dict:
    """All/Old/New ACC + K^n error over an unlabelled set (logits, GT labels).

    frozen_known=True fixes known prototypes to their classes and only Hungarian-matches
    the novel prototypes (prevents a novel cluster from stealing a known class in the score).
    """
    preds = assign_to_prototypes(logits, prototypes, lda).cpu().numpy()
    y_true = labels.cpu().numpy().astype(int)
    mask_old = y_true < num_labeled_classes                      # True for known classes
    if frozen_known:
        all_acc, old_acc, new_acc = split_cluster_acc_frozen_known(
            y_true, preds, num_labeled_classes, total_classes)
    else:
        all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true, preds, mask_old)

    k_hat = prototypes.shape[0] - num_labeled_classes
    k_true = total_classes - num_labeled_classes
    return {
        "all_acc": all_acc, "old_acc": old_acc, "new_acc": new_acc,
        "k_hat_novel": int(k_hat), "k_novel_true": int(k_true),
        "k_novel_err": int(abs(k_hat - k_true)),
    }



@torch.no_grad()
def explain_prototype(mu_k: torch.Tensor, vocab: List[str], top: int = 5) -> List[Tuple[str, float]]:
    """Top concepts describing a component, read from sigma(mu_k) (draft: no post-hoc)."""
    probs = torch.sigmoid(mu_k)
    k = min(top, len(vocab))
    vals, idx = probs.topk(k)
    return [(vocab[i], float(v)) for i, v in zip(idx.tolist(), vals.tolist())]
