"""Shared RAM++ loading + batched tagging.

Used by both N-concept-Gen.py (concept-bag exploration) and concept_discovery.py (phase 3's
novel-concept discovery step). RAM++ needs no class label and no candidate vocabulary to tag
an image -- see N-concept-Gen.py's module docstring for why that matters for xGCD.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # ram_tagging is imported from files that can't use `-m`
sys.path.insert(0, str(REPO_ROOT / "recognize-anything"))

from ram import get_transform as get_ram_transform  # noqa: E402
from ram.models import ram_plus  # noqa: E402

DEFAULT_RAM_PRETRAINED = str(REPO_ROOT / "recognize-anything" / "pretrained" / "ram_plus_swin_large_14m.pth")

# RAM++'s text encoder loads its tokenizer via `BertTokenizer.from_pretrained(text_encoder_type)`,
# which normally hits the Hugging Face Hub for "bert-base-uncased" -- unreachable from a
# no-internet compute node. `from_pretrained` accepts a local folder just as well as a hub id,
# so if vocab.txt/tokenizer_config.json/config.json have been manually downloaded into
# DEFAULT_BERT_TOKENIZER_DIR, use that instead; otherwise fall back to the hub id (e.g. for a
# dev machine that does have internet).
DEFAULT_BERT_TOKENIZER_DIR = str(REPO_ROOT / "recognize-anything" / "pretrained" / "bert-base-uncased")


def load_ram_plus(pretrained: str = DEFAULT_RAM_PRETRAINED, image_size: int = 384, device=None,
                  text_encoder_type: str = None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if text_encoder_type is None:
        text_encoder_type = (DEFAULT_BERT_TOKENIZER_DIR if Path(DEFAULT_BERT_TOKENIZER_DIR).is_dir()
                             else "bert-base-uncased")
    model = ram_plus(pretrained=pretrained, image_size=image_size, vit="swin_l",
                     text_encoder_type=text_encoder_type)
    model.eval().to(device)
    return model, get_ram_transform(image_size=image_size)


class _RawArrayDataset(Dataset):
    """Raw uint8 HWC arrays (e.g. CIFAR's `.data`) through RAM++'s own transform."""

    def __init__(self, arrays: np.ndarray, uq_idxs, transform):
        self.arrays = arrays
        self.uq_idxs = uq_idxs
        self.transform = transform

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, i):
        image = self.transform(Image.fromarray(self.arrays[i]).convert("RGB"))
        return image, int(self.uq_idxs[i])


@torch.no_grad()
def tag_by_uq(model, transform, arrays: np.ndarray, uq_idxs, device,
             batch_size: int = 32, num_workers: int = 4) -> dict:
    """Run RAM++ over raw uint8 HWC arrays. Returns {uq_idx: [tags...]}."""
    ds = _RawArrayDataset(arrays, uq_idxs, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tags_by_uq = {}
    for images, uqs in loader:
        images = images.to(device)
        tag_strs, _ = model.generate_tag(images)
        for uq, tag_str in zip(uqs.tolist(), tag_strs):
            tags_by_uq[int(uq)] = [t.strip() for t in tag_str.split("|") if t.strip()]
    return tags_by_uq
