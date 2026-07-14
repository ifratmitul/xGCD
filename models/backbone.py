"""DINO ViT-B/16 backbone wrapper.

Extracts the construction/freezing logic that used to live inline in
`methods/contrastive_training/gmm_sm.py` so every xGCD stage builds the backbone
the same way. Produces the 768-d CLS token used as the input to the CBL.
"""
from typing import Iterator, Optional

import torch
import torch.nn as nn
from loguru import logger

from models import vision_transformer as vits

# Official DINO backbone checkpoints (already clean backbone state_dicts).
# Downloaded on demand via torch.hub and cached under ~/.cache/torch/hub/checkpoints.
DINO_WEIGHT_URLS = {
    ("vit_small", 16): "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
    ("vit_small", 8): "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
    ("vit_base", 16): "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    ("vit_base", 8): "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
}


class DinoViTBackbone(nn.Module):
    """Frozen DINO ViT with only the last `grad_from_block` blocks trainable.

    forward(x) -> [B, feat_dim] CLS token (un-normalized).

    Weights: by default the official DINO checkpoint is downloaded and cached via
    torch.hub (`pretrain="dino"`). Pass `pretrain_path` to load a local file instead,
    or `pretrain=None` to build an un-initialised model (shape/CPU testing only).
    """

    def __init__(
        self,
        arch: str = "vit_base",
        patch_size: int = 16,
        pretrain: Optional[str] = "dino",
        pretrain_path: Optional[str] = None,
        grad_from_block: int = 11,
        mae_checkpoint: bool = False,
    ):
        super().__init__()
        if arch not in vits.__dict__:
            raise ValueError(f"Unknown ViT arch '{arch}'")
        self.vit = vits.__dict__[arch](patch_size=patch_size)
        self.feat_dim = self.vit.embed_dim  # 768 for vit_base
        self.grad_from_block = grad_from_block

        if pretrain_path is not None:
            self._load_local(pretrain_path, mae_checkpoint)
        elif pretrain == "dino":
            self._load_dino(arch, patch_size)
        elif pretrain is None:
            logger.warning(
                "DinoViTBackbone created without pretrained weights "
                "(pretrain=None) — only use this for shape/CPU testing."
            )
        else:
            raise ValueError(f"Unknown pretrain source '{pretrain}'")

        self.set_finetune_blocks(grad_from_block)

    def _load_dino(self, arch: str, patch_size: int):
        key = (arch, patch_size)
        if key not in DINO_WEIGHT_URLS:
            raise ValueError(f"No DINO checkpoint URL for {key}; pass pretrain_path instead.")
        url = DINO_WEIGHT_URLS[key]
        logger.info(f"Downloading/loading DINO weights for {key} from {url}")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        msg = self.vit.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded DINO weights: {msg}")

    def _load_local(self, path: str, mae_checkpoint: bool):
        state_dict = torch.load(path, map_location="cpu")
        if mae_checkpoint:
            state_dict = state_dict["model"]
        msg = self.vit.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded backbone weights from {path}: {msg}")

    def set_finetune_blocks(self, grad_from_block: int):
        """Freeze everything, then unfreeze transformer blocks >= grad_from_block."""
        self.grad_from_block = grad_from_block
        for p in self.vit.parameters():
            p.requires_grad = False
        for name, p in self.vit.named_parameters():
            if name.startswith("blocks") or ".blocks." in name or name.split(".")[0] == "blocks":
                # names look like 'blocks.11.norm1.weight'
                parts = name.split(".")
                try:
                    block_num = int(parts[parts.index("blocks") + 1])
                except (ValueError, IndexError):
                    continue
                if block_num >= grad_from_block:
                    p.requires_grad = True

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.vit.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)  # VisionTransformer.forward returns CLS token x[:, 0]
