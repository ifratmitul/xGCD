"""Shared feature / concept-logit extraction over a DataLoader.

Reused across stages so no stage re-implements the forward loop:
  * extract_backbone_features -> CLS features z (Stage 1 CBL training)
  * extract_concept_logits    -> concept logits ell (E-step, eval)

Both accept loaders whose batches start with (img, label, uq_idx, ...) — the extra
`mask_lab` field emitted by MergedDataset is preserved when present.
"""
from typing import Tuple

import torch
from tqdm import tqdm


def _unpack(batch):
    img, label, uq = batch[0], batch[1], batch[2]
    mask = batch[3] if len(batch) > 3 else None
    return img, label, uq, mask


@torch.no_grad()
def extract_backbone_features(backbone, loader, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (feats [N, 768], labels [N], uq_idxs [N]) on CPU."""
    backbone.eval()
    feats, labels, uqs = [], [], []
    for batch in tqdm(loader, desc="extract features", leave=False):
        img, label, uq, _ = _unpack(batch)
        f = backbone(img.to(device))
        feats.append(f.detach().cpu())
        labels.append(torch.as_tensor(label))
        uqs.append(torch.as_tensor(uq))
    return torch.cat(feats), torch.cat(labels), torch.cat(uqs)


@torch.no_grad()
def extract_concept_logits(model, loader, device):
    """Return (logits [N, C], labels [N], uq_idxs [N], mask_lab [N] or None) on CPU.

    `model` is a ConceptModel (backbone -> CBL). Used by the E-step and eval.
    """
    model.eval()
    logits, labels, uqs, masks = [], [], [], []
    have_mask = True
    for batch in loader:
        img, label, uq, mask = _unpack(batch)
        ell = model(img.to(device))
        logits.append(ell.detach().cpu())
        labels.append(torch.as_tensor(label))
        uqs.append(torch.as_tensor(uq))
        if mask is None:
            have_mask = False
        else:
            masks.append(torch.as_tensor(mask).reshape(-1))
    mask_lab = torch.cat(masks).bool() if have_mask else None
    return torch.cat(logits), torch.cat(labels), torch.cat(uqs), mask_lab
