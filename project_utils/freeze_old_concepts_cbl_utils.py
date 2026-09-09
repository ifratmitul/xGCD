"""Freeze the ORIGINAL (pre-growth) CBL concept rows for the rest of phase 3.

Simpler alternative to picking out "important" old weights via some pruning/importance
score: freeze the WHOLE old CBL block instead. Known-class concept detection then stays
byte-for-byte identical to Stage 1 throughout phase 3 -- not just unaffected by the new
concepts, but literally unable to drift from anything CE/BCE do to the newly-grown rows,
since the old rows never update at all. The new rows are still fully trainable (see
combined_fidelity_bce's `novel_only` flag in concept_discovery_utils.py for restricting
their training signal to novel images only, the companion half of this design).

nn.Linear doesn't support partial requires_grad on a slice of one weight tensor, and zeroing
the gradient for that slice isn't sufficient either -- SGD's weight_decay term
(`grad += weight_decay * param`) still nudges a zero-gradient row on every step. So this
snapshots the old rows once and restores them after every optimizer step instead of trying
to prevent the update in the first place -- correct regardless of what the optimizer does
internally (momentum, weight decay, or anything else).
"""
import torch


class FrozenCBLRows:
    """Snapshot of a CBL's first `c_old` rows. Call `.restore(cbl)` after every optimizer
    step (including inside CBL warmup, if used) to pin those rows back to the snapshot."""

    def __init__(self, cbl, c_old: int):
        self.c_old = c_old
        linear = cbl.model[0]
        self.weight = linear.weight[:c_old].detach().clone()
        self.bias = linear.bias[:c_old].detach().clone() if linear.bias is not None else None

    @torch.no_grad()
    def restore(self, cbl):
        linear = cbl.model[0]
        linear.weight[:self.c_old] = self.weight
        if self.bias is not None:
            linear.bias[:self.c_old] = self.bias
