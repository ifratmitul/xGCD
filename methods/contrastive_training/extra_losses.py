"""Optional extra losses for Phase 3 -- not part of the core pipeline, opt-in via CLI flag."""
import torch


def known_concept_anchor_loss(current_logits: torch.Tensor, frozen_logits: torch.Tensor,
                              mask_lab: torch.Tensor) -> torch.Tensor:
    """Pull labelled samples' CBL output back toward a frozen (pre-training) snapshot.

    current_logits / frozen_logits: [B, C] concept logits from the live vs. frozen CBL,
    on the SAME batch of images. mask_lab: [B] bool, True for labelled samples -- only
    they get anchored (unlabelled samples are exactly the ones Phase 3 wants to keep
    adapting freely, since their pseudo-labels are what it's trying to refine).
    """
    if not mask_lab.any():
        return current_logits.new_zeros(())
    diff = current_logits[mask_lab] - frozen_logits[mask_lab]
    return (diff ** 2).mean()
