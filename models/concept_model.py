"""ConceptModel = backbone -> CBL, the single object every stage passes around.

  Stage 1 : freeze backbone, train `cbl` with BCE.
  E-step  : `forward` (or `forward_features`) to get concept logits for all data.
  M-step  : `param_groups(backbone_lr, cbl_lr)` -> optimiser (small CBL LR = soft
            concept-drift constraint), then Mahalanobis losses on the logits.
  Eval    : `forward` to assign to prototypes; `concept_activations` to explain.
"""
import os
from typing import List, Tuple

import torch
import torch.nn as nn

from models.backbone import DinoViTBackbone
from models.cbl import ConceptBottleneckLayer


class ConceptModel(nn.Module):
    def __init__(self, backbone: DinoViTBackbone, cbl: ConceptBottleneckLayer):
        super().__init__()
        self.backbone = backbone
        self.cbl = cbl

    @property
    def num_concepts(self) -> int:
        return self.cbl.num_concepts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x images -> concept logits ell [B, C]."""
        return self.cbl(self.backbone(x))

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (backbone CLS feature z [B, 768], concept logits ell [B, C])."""
        z = self.backbone(x)
        return z, self.cbl(z)

    def concept_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid concept activations sigma(ell) — for explanations only."""
        return torch.sigmoid(self.forward(x))

    def param_groups(self, backbone_lr: float, cbl_lr: float) -> List[dict]:
        """Optimiser param groups: trainable backbone blocks at `backbone_lr`,
        the CBL at `cbl_lr` (typically backbone_lr / 100 in the M-step)."""
        return [
            {"params": list(self.backbone.trainable_parameters()), "lr": backbone_lr},
            {"params": list(self.cbl.parameters()), "lr": cbl_lr},
        ]

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.cbl.save(save_dir, name="cbl.pt")
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))
