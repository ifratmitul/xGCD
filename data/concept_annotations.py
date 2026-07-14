"""Stage D — concept annotation layer (VLG-CBM Grounding-DINO output).

Three pieces, all keyed by CIFAR `uq_idx` (= native torchvision index = JSON filename):

  build_concept_vocabulary(...)  -> the concept vocabulary S~  (VLG-CBM Eq 3),
                                    built from the *labelled known-class* images only.
  ConceptTargetLookup            -> uq_idx -> binary concept target o_i in {0,1}^C
                                    (VLG-CBM Eq 4), a standalone service (no dataset edits).
  get_concept_vocab_and_lookup   -> convenience: build/cache vocab, set args.num_concepts,
                                    return (vocab, train_lookup).

Design notes
------------
* Vocabulary axes come from labelled known classes only, so novel category names never
  leak in. Targets o_i are then computed for *any* image against that fixed axis set;
  supervision (BCE) is applied to labelled images only (masking happens in the trainer).
* A detection contributes concept s_j iff its confidence logit t_j > threshold
  (VLG-CBM Eq 2, default T = 0.15).
* Box coordinates are ignored — o_i is presence/absence only, so the 32->224 resize and
  the absence of crop augmentation are irrelevant here.
"""
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from loguru import logger

DEFAULT_CONF_THRESHOLD = 0.15  # VLG-CBM Appendix B: "We set T = 0.15 in all our experiments."


# ----------------------------------------------------------------------------- #
# Annotation parsing
# ----------------------------------------------------------------------------- #
def annotation_path(ann_dir: str, uq_idx: int) -> str:
    return os.path.join(ann_dir, f"{int(uq_idx)}.json")


def parse_annotation_file(path: str) -> List[tuple]:
    """Return list of (concept_label:str, logit:float) detections for one image.

    The JSON is a list whose first element is metadata ({"img_path": ...}); every
    other element is a detection {"label", "logit", "box"}.
    """
    with open(path, "r") as f:
        records = json.load(f)
    out = []
    for r in records:
        if isinstance(r, dict) and "label" in r and "logit" in r:
            out.append((r["label"], float(r["logit"])))
    return out


def concepts_present(path: str, threshold: float) -> set:
    """Set of concept labels detected in this image above `threshold`."""
    return {lbl for lbl, logit in parse_annotation_file(path) if logit > threshold}


# ----------------------------------------------------------------------------- #
# Vocabulary (VLG-CBM Eq 3)
# ----------------------------------------------------------------------------- #
def build_concept_vocabulary(
    ann_dir: str,
    uq_idxs: Sequence[int],
    threshold: float = DEFAULT_CONF_THRESHOLD,
    min_images: int = 1,
) -> List[str]:
    """Union of concepts grounded (logit > threshold) in >= `min_images` of the given
    images. `uq_idxs` should be the labelled known-class indices. Returns a sorted list
    (deterministic concept -> index mapping)."""
    counts: Dict[str, int] = {}
    missing = 0
    for uq in uq_idxs:
        p = annotation_path(ann_dir, uq)
        if not os.path.exists(p):
            missing += 1
            continue
        for c in concepts_present(p, threshold):
            counts[c] = counts.get(c, 0) + 1
    if missing:
        logger.warning(f"{missing}/{len(uq_idxs)} annotation files missing under {ann_dir}")
    vocab = sorted([c for c, n in counts.items() if n >= min_images])
    logger.info(
        f"Built concept vocabulary: {len(vocab)} concepts "
        f"(threshold={threshold}, min_images={min_images}, from {len(uq_idxs)} labelled imgs)"
    )
    return vocab


def save_vocabulary(vocab: List[str], path: str, meta: Optional[dict] = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"vocab": vocab, "meta": meta or {}}, f, indent=2)
    logger.info(f"Saved vocabulary ({len(vocab)} concepts) -> {path}")


def load_vocabulary(path: str) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)["vocab"]


# ----------------------------------------------------------------------------- #
# uq_idx -> o_i lookup (VLG-CBM Eq 4)
# ----------------------------------------------------------------------------- #
class ConceptTargetLookup:
    """Maps a CIFAR uq_idx to its binary concept target o_i in {0,1}^C.

    Usage:
        lookup = ConceptTargetLookup(ann_dir, vocab, threshold)
        o = lookup[uq_idx]                 # np.float32 [C]
        O = lookup.batch(uq_idxs)          # torch.FloatTensor [B, C]
    Optionally call `precompute(uq_idxs)` to cache targets for fast repeated access.
    """

    def __init__(self, ann_dir: str, vocab: List[str], threshold: float = DEFAULT_CONF_THRESHOLD):
        self.ann_dir = ann_dir
        self.vocab = list(vocab)
        self.threshold = threshold
        self.concept_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.C = len(self.vocab)
        self._cache: Dict[int, np.ndarray] = {}

    def compute(self, uq_idx: int) -> np.ndarray:
        o = np.zeros(self.C, dtype=np.float32)
        p = annotation_path(self.ann_dir, uq_idx)
        if not os.path.exists(p):
            return o  # unseen/missing -> all-zero target
        for c in concepts_present(p, self.threshold):
            j = self.concept_to_idx.get(c)
            if j is not None:  # concepts outside the fixed vocabulary are dropped
                o[j] = 1.0
        return o

    def __getitem__(self, uq_idx: int) -> np.ndarray:
        uq_idx = int(uq_idx)
        if uq_idx in self._cache:
            return self._cache[uq_idx]
        return self.compute(uq_idx)

    def precompute(self, uq_idxs: Sequence[int]):
        for uq in uq_idxs:
            uq = int(uq)
            if uq not in self._cache:
                self._cache[uq] = self.compute(uq)
        logger.info(f"Precomputed {len(self._cache)} concept targets (C={self.C})")
        return self

    def batch(self, uq_idxs) -> torch.Tensor:
        if torch.is_tensor(uq_idxs):
            uq_idxs = uq_idxs.cpu().numpy().tolist()
        return torch.from_numpy(np.stack([self[int(u)] for u in uq_idxs], axis=0))

    def pos_weight(self, uq_idxs: Sequence[int]) -> torch.Tensor:
        """Per-concept BCE pos_weight = #neg / #pos over the given images.

        VLG-CBM notes the concept dataset is heavily negative-skewed and scales the BCE
        loss to balance precision/recall; this returns a ready-to-use `pos_weight` for
        `nn.BCEWithLogitsLoss`.
        """
        O = np.stack([self[int(u)] for u in uq_idxs], axis=0)  # [N, C]
        pos = O.sum(axis=0)
        neg = O.shape[0] - pos
        weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
        return torch.from_numpy(weight.astype(np.float32))


# ----------------------------------------------------------------------------- #
# Convenience wiring
# ----------------------------------------------------------------------------- #
def get_concept_vocab_and_lookup(args, datasets, split_train_subdir=True):
    """Build (or load cached) vocabulary from labelled known-class images, set
    `args.num_concepts`, and return (vocab, train_lookup).

    Expects config to define `concept_annotation_root` and `concept_annotation_dirs`.
    """
    from config import concept_annotation_root, concept_annotation_dirs

    if args.dataset_name not in concept_annotation_dirs:
        raise ValueError(f"No concept annotation dirs configured for '{args.dataset_name}'")
    train_sub, _ = concept_annotation_dirs[args.dataset_name]
    ann_dir = os.path.join(concept_annotation_root, train_sub)

    threshold = getattr(args, "concept_conf_threshold", DEFAULT_CONF_THRESHOLD)
    labelled_uq = np.asarray(datasets["train_labelled"].uq_idxs).astype(int).tolist()

    # cache the vocabulary so runs are reproducible and fast.
    # The split (# labelled/known classes) is part of the key: a different known-class
    # set yields a different vocabulary, so 80/20 and 50/50 must not share a cache file.
    cache_dir = getattr(args, "vocab_cache_dir", os.path.join(concept_annotation_root, "_vocab"))
    n_lab = getattr(args, "num_labeled_classes", len(list(args.train_classes)))
    vocab_path = os.path.join(cache_dir, f"{args.dataset_name}_L{n_lab}_T{threshold}.json")

    if getattr(args, "rebuild_vocab", False) or not os.path.exists(vocab_path):
        vocab = build_concept_vocabulary(ann_dir, labelled_uq, threshold=threshold)
        save_vocabulary(
            vocab, vocab_path,
            meta={"dataset": args.dataset_name, "threshold": threshold,
                  "num_labeled_classes": n_lab,
                  "n_labelled_imgs": len(labelled_uq),
                  "train_classes": list(args.train_classes)},
        )
    else:
        vocab = load_vocabulary(vocab_path)
        logger.info(f"Loaded cached vocabulary ({len(vocab)} concepts) from {vocab_path}")

    args.num_concepts = len(vocab)
    lookup = ConceptTargetLookup(ann_dir, vocab, threshold=threshold)
    return vocab, lookup
