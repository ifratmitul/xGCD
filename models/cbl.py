"""Concept Bottleneck Layer (CBL).

A linear map from backbone features to concept *logits* `ell = W_phi z`.
Mirrors VLG-CBM's `ConceptLayer` (sample-code-vlg-cbm/cbm.py): a single
`nn.Linear(in_features, num_concepts, bias=True)` when `num_hidden == 0`.

All downstream Gaussian / LDA / DPMM modelling operates on these logits
(unbounded support), never on the sigmoid activations `sigma(ell)`, which are
reserved for human-readable explanations.
"""
import os
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


class ConceptBottleneckLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_concepts: int,
        bias: bool = True,
        num_hidden: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_concepts = num_concepts

        layers = [nn.Linear(in_features, num_concepts, bias=bias)]
        for _ in range(num_hidden):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(num_concepts, num_concepts, bias=bias))
        self.model = nn.Sequential(*layers)
        logger.info(f"CBL: {in_features} -> {num_concepts} (num_hidden={num_hidden})")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, in_features] -> concept logits [B, num_concepts]."""
        return self.model(z)

    def concept_activations(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(z))

    def save(self, save_dir: str, name: str = "cbl.pt"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, name))

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        in_features: int,
        num_concepts: int,
        bias: bool = True,
        num_hidden: int = 0,
        map_location: Optional[str] = "cpu",
    ) -> "ConceptBottleneckLayer":
        model = cls(in_features, num_concepts, bias=bias, num_hidden=num_hidden)
        model.load_state_dict(torch.load(load_path, map_location=map_location))
        return model
