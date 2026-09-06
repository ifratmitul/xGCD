"""Grounding DINO tagging — for the "did KNOWN samples get updated with any new concept?"
question in concept_discovery.py.

RAM++ (ram_tagging.py) needs no candidate vocabulary, which is exactly why it's the right
model to discover concepts FROM the novel clusters (see N-concept-Gen.py / concept_discovery.py
docstrings). But once a concept is discovered, checking whether it shows up in the *labelled*
set is a closed-vocabulary question (`new_concepts` is a short, known list) — Grounding DINO is
the model already used everywhere else in this repo for exactly that (data/annotations/*.json,
Stage 1's targets), so labelled-set targets for the new concept dims are produced the same way
labelled-set targets always are here, instead of switching models for one step.

NOT independently verified in this environment: the `groundingdino` package (and its config +
checkpoint) is not installed/vendored in this repo the way recognize-anything is — the original
Grounding-DINO annotations under data/annotations/ were produced by a script that lived
elsewhere. Install per https://github.com/IDEA-Research/GroundingDINO and point
--gdino_config / --gdino_checkpoint at your local config/weights before relying on this.
"""
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

try:
    import groundingdino.datasets.transforms as T
    from groundingdino.util.inference import load_model, predict
except ImportError as e:
    raise ImportError(
        "grounding_dino_tagging requires the `groundingdino` package "
        "(pip install, or vendor github.com/IDEA-Research/GroundingDINO the way "
        "recognize-anything/ is vendored) plus a config + checkpoint file."
    ) from e

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GDINO_CONFIG = str(REPO_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
DEFAULT_GDINO_CHECKPOINT = str(REPO_ROOT / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth")

_GDINO_TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_grounding_dino(config_path: str = DEFAULT_GDINO_CONFIG,
                        checkpoint_path: str = DEFAULT_GDINO_CHECKPOINT, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return load_model(config_path, checkpoint_path, device=str(device))


def _phrase_matches(concept: str, phrase: str) -> bool:
    c, p = concept.lower().strip(), phrase.lower().strip()
    return c == p or c in p or p in c


@torch.no_grad()
def ground_by_uq(model, arrays, uq_idxs, concepts: List[str], device,
                 box_threshold: float = 0.35, text_threshold: float = 0.25) -> Dict[int, List[str]]:
    """Run Grounding DINO over raw uint8 HWC arrays with `concepts` as the text queries.

    Returns {uq_idx: [concepts detected above threshold]} — the same shape as
    ram_tagging.tag_by_uq's output, so concept_discovery.py's `_tags_to_o` works unchanged
    on either source.

    One image at a time: Grounding DINO's own `predict()` is not batched.
    """
    caption = " . ".join(concepts) + " ."
    out = {}
    for arr, uq in zip(arrays, uq_idxs):
        image_pil = Image.fromarray(arr).convert("RGB")
        image_tensor, _ = _GDINO_TRANSFORM(image_pil, None)
        _, logits, phrases = predict(
            model=model, image=image_tensor, caption=caption,
            box_threshold=box_threshold, text_threshold=text_threshold, device=str(device),
        )
        hits = {p for p, l in zip(phrases, logits.tolist()) if l > text_threshold}
        out[int(uq)] = [c for c in concepts if any(_phrase_matches(c, p) for p in hits)]
    return out
